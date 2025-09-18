"""SIMICE remote client using Core.remote_core.

This module subclasses the common RemoteClient base and registers handlers
for 'initialize', 'information', and 'impute'. It keeps legacy entry points
for backward compatibility with existing orchestration code.
"""

import numpy as np
import pandas as pd
import asyncio
from scipy.special import expit
import os
from Core.transfer import (
    write_matrix, write_vector,
    read_string, read_integer, read_vector,
    WebSocketWrapper
)
from Core.remote_core import RemoteClient, run_remote_client_async

class SIMICERemoteClient(RemoteClient):
    """SIMICE algorithm remote client implementation."""

    def __init__(self, data, central_host, central_port, central_proto, site_id, parameters=None):
        super().__init__(data, central_host, central_port, central_proto, site_id, parameters)

        # State used across handlers
        self.D: np.ndarray | None = None  # with intercept
        self.DD: np.ndarray | None = None
        self.p: int | None = None
        self.orig_missing_map: dict[int, np.ndarray] = {}

        # Register handlers
        self.register_handler("initialize", self.handle_initialize)
        self.register_handler("information", self.handle_information)
        self.register_handler("impute", self.handle_impute)

    async def prepare_data(self, mvar: int) -> None:
        # Add intercept column
        self.D = np.column_stack([self.data, np.ones(self.data.shape[0])]).astype(np.float64, copy=False)
        self.p = int(self.D.shape[1])
        # Optional seeding
        seed_env = os.getenv("GLOBAL_SEED")
        if seed_env:
            try:
                np.random.seed(int(seed_env))
                print(f"[{self.site_id}] GLOBAL_SEED set; using seed {int(seed_env)}", flush=True)
            except Exception:
                pass
        print(f"[{self.site_id}] Prepared SIMICE data shape={self.D.shape}", flush=True)

    async def handle_initialize(self, websocket: WebSocketWrapper) -> bool:
        # Receive mvar vector (1-based)
        mvar_vec = await read_vector(websocket)
        mvar = [int(idx) - 1 for idx in mvar_vec]
        self.mvar_list = mvar
        miss = np.isnan(self.D)
        check_cols = [i for i in range(self.p) if i not in mvar and i != self.p - 1]
        valid_rows = np.sum(miss[:, check_cols], axis=1) == 0
        DD = self.D[valid_rows].copy()
        miss_DD = np.isnan(DD)
        self.orig_missing_map = {}
        for j in mvar:
            miss_j = miss_DD[:, j]
            self.orig_missing_map[j] = miss_j.copy()
            if np.any(miss_j):
                DD[miss_j, j] = np.mean(DD[~miss_j, j])
        self.DD = DD
        print(f"[{self.site_id}] Initialize complete mvar={mvar}", flush=True)
        return False

    async def handle_information(self, websocket: WebSocketWrapper) -> bool:
        method = (await read_string(websocket)).lower()
        j_r = await read_integer(websocket)
        j = j_r - 1
        DD = self.DD
        if DD is None:
            raise ValueError("SIMICE: DD not initialized; missing Initialize phase")
        orig_missing = self.orig_missing_map
        if j in orig_missing:
            valid_idx = ~orig_missing[j]
        else:
            valid_idx = ~np.isnan(DD)[:, j]
        X = np.delete(DD[valid_idx], j, axis=1)
        y = DD[valid_idx, j]

        if method == "gaussian":
            n = X.shape[0]
            XX = X.T @ X
            Xy = X.T @ y
            yy = np.sum(y ** 2)
            if np.isnan(XX).any() or np.isinf(XX).any():
                XX = np.nan_to_num(XX)
            if np.isnan(Xy).any() or np.isinf(Xy).any():
                Xy = np.nan_to_num(Xy)
            await write_vector(np.array([float(n)]), websocket)
            await write_matrix(XX, websocket)
            await write_vector(Xy.astype(float), websocket)
            await write_vector(np.array([float(yy)]), websocket)
            print(f"[SIMICE.info.gaussian] Sent stats: n={n}, p={X.shape[1]}", flush=True)
        elif method == "logistic":
            mode = await read_integer(websocket)
            beta = await read_vector(websocket)
            await write_vector(np.array([X.shape[0]], dtype=float), websocket)
            xb = X @ beta
            pr = 1.0 / (1.0 + np.exp(-xb))
            low = pr < 0.5
            high = ~low
            Q = np.sum(y * xb)
            if np.any(low):
                Q += np.sum(np.log(np.maximum(1e-10, 1 - pr[low])))
            if np.any(high):
                Q += np.sum(np.log(np.maximum(1e-10, pr[high])) - xb[high])
            if mode == 1:
                w = pr * (1 - pr)
                H = (X.T * w) @ X
                g = X.T @ (y - pr)
                if np.isnan(H).any() or np.isinf(H).any():
                    H = np.nan_to_num(H)
                if np.isnan(g).any() or np.isinf(g).any():
                    g = np.nan_to_num(g)
                await write_matrix(H, websocket)
                await write_vector(g.astype(float), websocket)
                print(f"[SIMICE.info.logistic] Sent H & g", flush=True)
            await write_vector(np.array([float(Q)]), websocket)
            print(f"[SIMICE.info.logistic] Sent Q={Q}", flush=True)
        else:
            print(f"[{self.site_id}] Unknown method in Information: {method}")
        return False

    async def handle_impute(self, websocket: WebSocketWrapper) -> bool:
        method = (await read_string(websocket)).lower()
        j_r = await read_integer(websocket)
        j = j_r - 1
        DD = self.DD
        if DD is None:
            raise ValueError("SIMICE: DD not initialized; missing Initialize phase")
        orig_missing = self.orig_missing_map
        if j in orig_missing:
            midx = np.where(orig_missing[j])[0]
        else:
            midx = np.where(np.isnan(DD)[:, j])[0]
        nmidx = len(midx)
        if nmidx == 0:
            print(f"[{self.site_id}] No original missing at j={j}; discard payload", flush=True)
            return False
        X = np.delete(DD[midx, :], j, axis=1)
        if method == "gaussian":
            beta = await read_vector(websocket)
            sigv = await read_vector(websocket)
            sig = float(sigv[0])
            self.DD[midx, j] = X @ beta + np.random.normal(0.0, sig, size=nmidx)
            print(f"[SIMICE.impute.gaussian] Imputed {nmidx} cells", flush=True)
        elif method == "logistic":
            alpha = await read_vector(websocket)
            pr = 1.0 / (1.0 + np.exp(-(X @ alpha)))
            self.DD[midx, j] = np.random.binomial(1, pr)
            print(f"[SIMICE.impute.logistic] Imputed {nmidx} cells", flush=True)
        else:
            print(f"[{self.site_id}] Unknown impute method: {method}")
        return False

    async def run(self) -> None:
        # SIMICE supports multiple target columns; central will provide them in Initialize.
        await self.prepare_data(0)
        job_id = self.parameters.get("job_id", "unknown")
        print(
            f"[async:{self.site_id}] Starting SIMICE remote task job_id={job_id} (multi-target)",
            flush=True,
        )
        await super().run()


def run_remote_client(data_file, central_host, central_port, central_proto, site_id=None, remote_port=None, config=None):
    if isinstance(data_file, str):
        D = pd.read_csv(data_file).values
    else:
        D = data_file
    site_id = site_id or "remote1"
    if remote_port is not None:
        print(f"[{site_id}] Ignoring remote_port={remote_port} (no local server)", flush=True)
    asyncio.run(async_run_remote_client(D, central_host, central_port, central_proto, site_id, config or {}))


async def async_run_remote_client(data_file, central_host, central_port, central_proto, site_id=None, config=None):
    if isinstance(data_file, str):
        D = pd.read_csv(data_file).values
    else:
        D = data_file
    site_id = site_id or "remote1"
    await run_remote_client_async(
        SIMICERemoteClient,
        data=D,
        central_host=central_host,
        central_port=central_port,
        central_proto=central_proto,
        site_id=site_id,
        parameters=config or {},
    )

 