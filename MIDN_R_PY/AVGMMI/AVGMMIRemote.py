"""AVGMMI remote client using Core.remote_core.

Implements the AVGMMIRemote.R logic using JSON-only WebSockets:
- For 'Gaussian': compute local OLS beta, iFisher, SSE, n and send.
- For 'logistic': compute local Logit beta, H, n and send.

Follows the same base class pattern as SIMIRemote.py for consistency.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve

from Core.transfer import (
    write_matrix, write_vector,
    WebSocketWrapper,
)
from Core.remote_core import RemoteClient, validate_parameters, run_remote_client_async
from Core.Logit import Logit


class AVGMMIRemoteClient(RemoteClient):
    """AVGMMI algorithm remote client implementation."""

    def __init__(self, data, central_host, central_port, central_proto, site_id, parameters=None):
        super().__init__(data, central_host, central_port, central_proto, site_id, parameters)
        mvar_py, method = validate_parameters(self.parameters, self.data.shape)
        self.mvar = mvar_py
        self.method = method

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

        self.register_handler("gaussian", self.handle_gaussian)
        self.register_handler("logistic", self.handle_logistic)

    async def prepare_data(self, mvar: int) -> None:
        miss = np.isnan(self.data[:, mvar])
        self.X = np.delete(self.data[~miss], mvar, axis=1)
        self.y = self.data[~miss, mvar]
        print(f"[{self.site_id}] AVGMMI prepared data X shape={self.X.shape}, y len={self.y.shape[0]}", flush=True)

    async def handle_gaussian(self, websocket: WebSocketWrapper) -> bool:
        assert self.X is not None and self.y is not None
        X, y = self.X, self.y
        n = X.shape[0]
        p = X.shape[1]

        lam = float(self.parameters.get("lam", 1e-3))
        XX = X.T @ X + lam * np.eye(p)
        Xy = X.T @ y
        yy = float(np.sum(y ** 2))
        try:
            U = cholesky(XX, lower=False)
            iXX = cho_solve((U, False), np.eye(p))
        except Exception:
            iXX = np.linalg.pinv(XX)
        beta = iXX @ Xy
        SSE = yy + float(beta @ (XX @ beta - 2 * Xy))
        iFisher = iXX @ XX @ iXX

        await write_vector(beta.astype(float), websocket)
        await write_matrix(iFisher.astype(float), websocket)
        await write_vector(np.array([float(SSE)]), websocket)
        await write_vector(np.array([float(n)]), websocket)
        print(f"[{self.site_id}.AVGMMI.gaussian] Sent beta({beta.size}), iFisher({iFisher.shape}), SSE, n={n}", flush=True)
        return False

    async def handle_logistic(self, websocket: WebSocketWrapper) -> bool:
        assert self.X is not None and self.y is not None
        X, y = self.X, self.y
        n = X.shape[0]
        lam = float(self.parameters.get("lam", 1e-3))
        maxiter = int(self.parameters.get("maxiter", 100))

        fit = Logit(X, y, lam=lam, maxiter=maxiter)
        beta = fit['beta']
        H = fit['H']

        await write_vector(beta.astype(float), websocket)
        await write_matrix(H.astype(float), websocket)
        await write_vector(np.array([float(n)]), websocket)
        print(f"[{self.site_id}.AVGMMI.logistic] Sent beta({beta.size}), H({H.shape}), n={n}", flush=True)
        return False

    async def run(self) -> None:
        await self.prepare_data(self.mvar)
        job_id = self.parameters.get("job_id")
        print(f"[async:{self.site_id}] Starting AVGMMI remote job_id={job_id} mvar(0-based)={self.mvar} method={self.method}", flush=True)
        await super().run()


def run_remote_client(data, central_host, central_port, central_proto, site_id, remote_port=None, config=None):
    # Legacy-compatible entry point
    if isinstance(data, str):
        D = pd.read_csv(data).values
    else:
        D = data
    config = config or {}
    if "mvar" not in config and "target_column_index" not in config:
        raise ValueError("Config must contain 'mvar' or 'target_column_index' (1-based) for AVGMMI remote")
    if remote_port is not None:
        print(f"[{site_id}] Ignoring remote_port={remote_port} (no local server)", flush=True)
    print(f"[{site_id}] Starting AVGMMI remote with data shape={np.asarray(D).shape}", flush=True)
    asyncio.run(async_run_remote_client(data, central_host, central_port, central_proto, site_id, config))


async def async_run_remote_client(data, central_host, central_port, central_proto, site_id, parameters):
    # Validate ahead of time
    if isinstance(data, str):
        D = pd.read_csv(data).values
    else:
        D = data
    _ = validate_parameters(parameters or {}, np.asarray(D).shape)
    await run_remote_client_async(
        AVGMMIRemoteClient,
        data=data,
        central_host=central_host,
        central_port=central_port,
        central_proto=central_proto,
        site_id=site_id,
        parameters=parameters or {},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AVGMMI remote client")
    parser.add_argument("--data", required=True, help="Path to data CSV")
    parser.add_argument("--mvar", type=int, help="1-based index of missing variable")
    parser.add_argument("--target_column_index", type=int, help="Alternative to --mvar (1-based)")
    parser.add_argument("--central_host", required=True)
    parser.add_argument("--central_port", type=int, required=True)
    parser.add_argument("--site_id", required=True)
    parser.add_argument("--port", type=int, help="Ignored legacy argument")
    parser.add_argument("--method", choices=["gaussian", "logistic"], help="Optional override method")
    args = parser.parse_args()

    cfg = {}
    if args.target_column_index is not None:
        cfg["target_column_index"] = args.target_column_index
    elif args.mvar is not None:
        cfg["mvar"] = args.mvar
    if args.method:
        cfg["method"] = args.method

    run_remote_client(args.data, args.central_host, args.central_port, "ws", args.site_id, args.port, cfg)
