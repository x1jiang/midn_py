"""CSLMICE remote client using Core.remote_core.

Implements multi-target CSL within chained equations. The remote behavior
mirrors AVGMMICE remotes for Initialize/Impute phases, but during
Information the central sends a betabar and the remote returns average
gradients g and sample size n, following CSL.
"""

from __future__ import annotations

import os
import asyncio
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from Core.transfer import (
    write_matrix, write_vector, write_string,
    read_string, read_integer, read_vector,
    WebSocketWrapper
)
from Core.remote_core import RemoteClient, run_remote_client_async


class CSLMICERemoteClient(RemoteClient):
    """CSLMICE algorithm remote client implementation (multi-target)."""

    def __init__(self, data, central_host, central_port, central_proto, site_id, parameters=None):
        super().__init__(data, central_host, central_port, central_proto, site_id, parameters)

        # State
        self.D: Optional[np.ndarray] = None   # with intercept
        self.DD: Optional[np.ndarray] = None  # filtered rows
        self.p: Optional[int] = None
        self.mvar_list: list[int] = []        # 0-based
        self.orig_missing_map: dict[int, np.ndarray] = {}

        # Handlers
        self.register_handler("initialize", self.handle_initialize)
        self.register_handler("information", self.handle_information)
        self.register_handler("impute", self.handle_impute)

    async def prepare_data(self, mvar: int) -> None:
        # Multi-target; central will send full list later
        D = np.asarray(self.data, dtype=float)
        self.D = np.column_stack([D, np.ones((D.shape[0],), dtype=float)])
        self.p = int(self.D.shape[1])
        seed_env = os.getenv("GLOBAL_SEED")
        if seed_env:
            try:
                np.random.seed(int(seed_env))
            except Exception:
                pass
        print(f"[{self.site_id}] CSLMICE prepared data shape={self.D.shape}", flush=True)

    async def handle_initialize(self, websocket: WebSocketWrapper) -> bool:
        # Receive vector of 1-based target indices
        vec = await read_vector(websocket)
        mvar_1b = [int(x) for x in vec]
        self.mvar_list = [j - 1 for j in mvar_1b]
        D = self.D
        assert D is not None
        miss = np.isnan(D)
        # Keep rows where all non-target, non-intercept columns are observed
        check_cols = [c for c in range(D.shape[1]) if c not in self.mvar_list and c != D.shape[1] - 1]
        valid_rows = np.sum(miss[:, check_cols], axis=1) == 0
        DD = D[valid_rows].copy()
        miss_DD = np.isnan(DD)
        self.orig_missing_map = {}
        for j in self.mvar_list:
            mj = miss_DD[:, j]
            self.orig_missing_map[j] = mj.copy()
            if np.any(mj):
                # Initialize with column mean (over observed)
                obs = ~mj
                mu = float(np.nanmean(DD[obs, j])) if np.any(obs) else 0.0
                DD[mj, j] = mu
        self.DD = DD
        print(f"[{self.site_id}] Initialize complete targets={self.mvar_list}", flush=True)
        return False

    async def handle_information(self, websocket: WebSocketWrapper) -> bool:
        method = (await read_string(websocket)).lower()
        j_r = await read_integer(websocket)
        j = int(j_r) - 1
        DD = self.DD
        assert DD is not None
        miss_map = self.orig_missing_map
        valid_idx = ~miss_map.get(j, np.isnan(DD)[:, j])
        X = np.delete(DD[valid_idx, :], j, axis=1)
        y = DD[valid_idx, j]
        n = X.shape[0]

        betabar = await read_vector(websocket)
        if method == "gaussian":
            resid = y - X @ betabar
            g = - (X.T @ resid) / float(n)
            await write_vector(np.array([float(n)]), websocket)
            await write_vector(g.astype(float), websocket)
            print(f"[{self.site_id}] SENDING -> INFO[gaussian]: n(number), g(vector)", flush=True)
        elif method == "logistic":
            xb = X @ betabar
            pr = 1.0 / (1.0 + np.exp(-xb))
            g = (X.T @ (y - pr)) / float(n)
            await write_vector(np.array([float(n)]), websocket)
            await write_vector(g.astype(float), websocket)
            print(f"[{self.site_id}] SENDING -> INFO[logistic]: n(number), g(vector)", flush=True)
        else:
            print(f"[{self.site_id}] Unknown method in Information: {method}")
        return False

    async def handle_impute(self, websocket: WebSocketWrapper) -> bool:
        method = (await read_string(websocket)).lower()
        j_r = await read_integer(websocket)
        j = int(j_r) - 1
        DD = self.DD
        assert DD is not None
        miss_map = self.orig_missing_map
        midx = np.where(miss_map.get(j, np.isnan(DD)[:, j]))[0]
        if midx.size == 0:
            return False
        X = np.delete(DD[midx, :], j, axis=1)
        if method == "gaussian":
            # Expect beta and sig
            beta = await read_vector(websocket)
            sigv = await read_vector(websocket)
            sig = float(sigv[0])
            DD[midx, j] = X @ beta + np.random.normal(0.0, sig, size=midx.size)
            print(f"[{self.site_id}] impute.gaussian j={j} n={midx.size}", flush=True)
        elif method == "logistic":
            alpha = await read_vector(websocket)
            pr = 1.0 / (1.0 + np.exp(-(X @ alpha)))
            DD[midx, j] = np.random.binomial(1, pr)
            print(f"[{self.site_id}] impute.logistic j={j} n={midx.size}", flush=True)
        return False

    async def run(self) -> None:
        await self.prepare_data(0)
        job_id = self.parameters.get("job_id", "unknown")
        print(f"[async:{self.site_id}] Starting CSLMICE remote job_id={job_id}", flush=True)
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
    print(f"[async:{site_id}] Loaded data shape={D.shape}", flush=True)
    await run_remote_client_async(
        CSLMICERemoteClient,
        data=D,
        central_host=central_host,
        central_port=central_port,
        central_proto=central_proto,
        site_id=site_id,
        parameters=config or {},
    )
