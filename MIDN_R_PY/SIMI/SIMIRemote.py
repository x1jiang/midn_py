"""SIMI remote client using Core.remote_core.

This module now subclasses the common RemoteClient base and registers
handlers for 'gaussian' and 'logistic'. It retains legacy entry points
for compatibility with existing orchestration code.
"""

import asyncio
import os
import numpy as np
import pandas as pd
from scipy.special import expit
from Core.transfer import (
    write_matrix, write_vector,
    read_integer, read_vector,
    WebSocketWrapper,
)
from Core.remote_core import RemoteClient, validate_parameters, run_remote_client_async


class SIMIRemoteClient(RemoteClient):
    """SIMI algorithm remote client implementation."""

    def __init__(self, data, central_host, central_port, central_proto, site_id, parameters=None):
        super().__init__(data, central_host, central_port, central_proto, site_id, parameters)
        # Validate parameters and prepare key indices
        mvar_py, method = validate_parameters(self.parameters, self.data.shape)
        self.mvar = mvar_py
        self.method = method

        # Placeholders set in prepare_data
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

        # Register algorithm handlers
        self.register_handler("gaussian", self.handle_gaussian)
        self.register_handler("logistic", self.handle_logistic)

    async def prepare_data(self, mvar: int) -> None:
        miss = np.isnan(self.data[:, mvar])
        self.X = np.delete(self.data[~miss], mvar, axis=1)
        self.y = self.data[~miss, mvar]
        print(f"[{self.site_id}] Prepared data X shape={self.X.shape}, y len={self.y.shape[0]}", flush=True)

    async def handle_gaussian(self, websocket: WebSocketWrapper) -> bool:
        assert self.X is not None and self.y is not None
        n = self.X.shape[0]
        XX = self.X.T @ self.X
        Xy = self.X.T @ self.y
        yy = np.sum(self.y ** 2)
        if np.isnan(XX).any() or np.isinf(XX).any():
            XX = np.nan_to_num(XX)
        if np.isnan(Xy).any() or np.isinf(Xy).any():
            Xy = np.nan_to_num(Xy)
        await write_vector(np.array([float(n)]), websocket)
        await write_matrix(XX, websocket)
        await write_vector(Xy.astype(float), websocket)
        await write_vector(np.array([float(yy)]), websocket)
        print(f"[SIMI.gaussian] Sent stats: n={n}, p={self.X.shape[1]}", flush=True)
        return False

    async def handle_logistic(self, websocket: WebSocketWrapper) -> bool:
        assert self.X is not None and self.y is not None
        n = self.X.shape[0]
        await write_vector(np.array([float(n)]), websocket)
        print(f"[SIMI.logistic] Sent sample size n={n}", flush=True)
        while True:
            try:
                mode = await read_integer(websocket)
            except ValueError as e:
                print(f"[SIMI.logistic] Failed to read mode integer: {e}", flush=True)
                await asyncio.sleep(0.1)
                continue
            if mode in (0, -1):
                print(f"[SIMI.logistic] Termination (mode {mode}) received", flush=True)
                break
            beta = await read_vector(websocket)
            xb = self.X @ beta
            pr = expit(xb)
            low = pr < 0.5
            high = ~low
            Q = np.sum(self.y * xb)
            if np.any(low):
                Q += np.sum(np.log(np.maximum(1e-10, 1 - pr[low])))
            if np.any(high):
                Q += np.sum(np.log(np.maximum(1e-10, pr[high])) - xb[high])
            if mode == 1:
                w = pr * (1 - pr)
                H = (self.X.T * w) @ self.X
                g = self.X.T @ (self.y - pr)
                if np.isnan(H).any() or np.isinf(H).any():
                    H = np.nan_to_num(H)
                if np.isnan(g).any() or np.isinf(g).any():
                    g = np.nan_to_num(g)
                await write_matrix(H, websocket)
                await write_vector(g.astype(float), websocket)
                print("[SIMI.logistic] Sent H & g", flush=True)
            await write_vector(np.array([float(Q)]), websocket)
            print(f"[SIMI.logistic] Sent Q={Q}", flush=True)
        return False

    async def run(self) -> None:
        await self.prepare_data(self.mvar)
        job_id = self.parameters.get("job_id")
        print(
            f"[async:{self.site_id}] Starting SIMI remote task job_id={job_id} mvar(0-based)={self.mvar} method={self.method}",
            flush=True,
        )
        await super().run()


def run_remote_client(data, central_host, central_port, central_proto, site_id, remote_port=None, config=None):
    # Legacy entry point
    if isinstance(data, str):
        D = pd.read_csv(data).values
    else:
        D = data
    config = config or {}
    if "mvar" not in config:
        raise ValueError("Config must contain 'mvar' (1-based index) for SIMI remote")
    mvar_py = config["mvar"] - 1
    if remote_port is not None:
        print(f"[{site_id}] Ignoring remote_port={remote_port} (no local server)", flush=True)
    print(f"[{site_id}] Starting SIMI remote with mvar (0-based)={mvar_py}", flush=True)
    # Use the async runner for consistency
    asyncio.run(async_run_remote_client(data, central_host, central_port, central_proto, site_id, config))


async def async_run_remote_client(data, central_host, central_port, central_proto, site_id, parameters):
    # Validate quickly to provide better error messages before constructing the client
    if isinstance(data, str):
        D = pd.read_csv(data).values
    else:
        D = data
    _ = validate_parameters(parameters or {}, D.shape)
    await run_remote_client_async(
        SIMIRemoteClient,
        data=data,
        central_host=central_host,
        central_port=central_port,
        central_proto=central_proto,
        site_id=site_id,
        parameters=parameters or {},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIMI remote client")
    parser.add_argument("--data", required=True, help="Path to data CSV")
    parser.add_argument("--mvar", type=int, required=True, help="1-based index of missing variable")
    parser.add_argument("--central_host", required=True)
    parser.add_argument("--central_port", type=int, required=True)
    parser.add_argument("--site_id", required=True)
    parser.add_argument("--port", type=int, help="Ignored legacy argument")
    args = parser.parse_args()
    cfg = {"mvar": args.mvar}
    run_remote_client(args.data, args.central_host, args.central_port, args.site_id, args.port, cfg)
