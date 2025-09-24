"""HDMI remote client using Core.remote_core.

Implements the HDMIRemote.R logic with JSON-only WebSockets:
- For 'Gaussian': local probit selection + two-step Heckman; send alpha and vcov.
- For 'logistic': local probit selection + probit outcome; send alpha and vcov.

Parameter packing matches central:
  Gaussian: alpha = [beta_sel(p), beta_out(p-1), log_sigma, atanh(rho)]
  Logistic: alpha = [beta_sel(p), beta_out(p-1), atanh(rho)]
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from Core.transfer import (
    write_matrix, write_vector,
    WebSocketWrapper, read_string
)
from Core.remote_core import RemoteClient, validate_parameters, run_remote_client_async


def _fit_probit(X: np.ndarray, y: np.ndarray, maxiter: int = 50, tol: float = 1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(maxiter):
        eta = X @ beta
        p_hat = np.clip(norm.cdf(eta), 1e-6, 1 - 1e-6)
        phi = norm.pdf(eta)
        W = (phi * phi) / (p_hat * (1 - p_hat))
        W = np.clip(W, 1e-8, None)
        z = eta + (y - p_hat) / np.maximum(phi, 1e-8)
        Xw = X * np.sqrt(W[:, None])
        zw = z * np.sqrt(W)
        try:
            XtX = Xw.T @ Xw
            Xtz = Xw.T @ zw
            beta_new = np.linalg.solve(XtX, Xtz)
        except np.linalg.LinAlgError:
            beta_new = beta
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    # covariance approx
    eta = X @ beta
    p_hat = np.clip(norm.cdf(eta), 1e-6, 1 - 1e-6)
    phi = norm.pdf(eta)
    W = (phi * phi) / (p_hat * (1 - p_hat))
    Xw = X * np.sqrt(W[:, None])
    try:
        cov = np.linalg.pinv(Xw.T @ Xw)
    except Exception:
        cov = np.eye(p) * 1e-2
    return beta, cov


class HDMIRemoteClient(RemoteClient):
    def __init__(self, data, central_host, central_port, central_proto, site_id, parameters=None):
        super().__init__(data, central_host, central_port, central_proto, site_id, parameters)
        mvar_py, method = validate_parameters(self.parameters, self.data.shape)
        self.mvar = mvar_py
        self.method = method

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

        # The server sends method as first string; we map both
        self.register_handler("gaussian", self.handle_gaussian)
        self.register_handler("logistic", self.handle_logistic)

    async def prepare_data(self, mvar: int) -> None:
        self.X = np.delete(self.data, mvar, axis=1)
        self.y = self.data[:, mvar]
        print(f"[{self.site_id}] HDMI prepared full X shape={self.X.shape}, y len={self.y.shape[0]}", flush=True)

    def _local_gaussian(self) -> tuple[np.ndarray, np.ndarray, float]:
        X_full = self.X; y_full = self.y
        assert X_full is not None and y_full is not None
        p = X_full.shape[1]
        sel = (~np.isnan(y_full)).astype(float)
        beta_sel, cov_sel = _fit_probit(X_full, sel)
        obs = sel == 1.0
        X_obs = X_full[obs]
        y_obs = y_full[obs]
        X_out = X_obs[:, :p-1]
        eta_sel_obs = X_obs @ beta_sel
        imr = - norm.pdf(eta_sel_obs) / np.clip(norm.cdf(-eta_sel_obs), 1e-9, None)
        X_ols = np.column_stack([X_out, imr])
        XtX = X_ols.T @ X_ols
        Xty = X_ols.T @ y_obs
        try:
            coef = np.linalg.solve(XtX, Xty)
            resid = y_obs - X_ols @ coef
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(XtX) @ Xty
            resid = y_obs - X_ols @ coef
        dof = max(1, X_ols.shape[0] - X_ols.shape[1])
        sigma_e = float(np.sqrt(np.sum(resid**2) / dof))
        beta_out = coef[:-1]
        delta = float(coef[-1])
        rho = float(np.clip(delta / (sigma_e + 1e-12), -0.999, 0.999))
        log_sigma = float(np.log(max(sigma_e, 1e-8)))
        rho_param = float(np.arctanh(rho))
        alpha = np.concatenate([beta_sel, beta_out, np.array([log_sigma, rho_param])])
        # simple block-diagonal covariance
        try:
            cov_out = np.linalg.pinv(XtX)
        except Exception:
            cov_out = np.eye(X_ols.shape[1]) * 1e-2
        cov_beta_out = cov_out[:-1, :-1]
        cov = np.zeros((2*p + 1, 2*p + 1))
        cov[:p, :p] = cov_sel
        cov[p:2*p-1, p:2*p-1] = cov_beta_out
        cov[2*p-1, 2*p-1] = 0.01
        cov[2*p, 2*p] = 0.01
        return alpha, cov, float(self.X.shape[0])

    def _local_logistic(self) -> tuple[np.ndarray, np.ndarray, float]:
        X_full = self.X; y_full = self.y
        assert X_full is not None and y_full is not None
        p = X_full.shape[1]
        sel = (~np.isnan(y_full)).astype(float)
        beta_sel, cov_sel = _fit_probit(X_full, sel)
        obs = sel == 1.0
        X_obs = X_full[obs]
        y_obs = np.clip(y_full[obs], 0, 1)
        X_out = X_obs[:, :p-1]
        beta_out, cov_out = _fit_probit(X_out, y_obs)
        rho_param = 0.0
        alpha = np.concatenate([beta_sel, beta_out, np.array([rho_param])])
        cov = np.zeros((2*p, 2*p))
        cov[:p, :p] = cov_sel
        cov[p:2*p-1, p:2*p-1] = cov_out
        cov[2*p-1, 2*p-1] = 0.01
        return alpha, cov, float(self.X.shape[0])

    async def handle_gaussian(self, websocket: WebSocketWrapper) -> bool:
        alpha, vcov, n = self._local_gaussian()
        await write_vector(alpha.astype(float), websocket)
        await write_matrix(vcov.astype(float), websocket)
        await write_vector(np.array([n], dtype=float), websocket)
        print(f"[{self.site_id}.HDMI.gaussian] Sent alpha({alpha.size}), vcov({vcov.shape}), n={n}", flush=True)
        return False

    async def handle_logistic(self, websocket: WebSocketWrapper) -> bool:
        alpha, vcov, n = self._local_logistic()
        await write_vector(alpha.astype(float), websocket)
        await write_matrix(vcov.astype(float), websocket)
        await write_vector(np.array([n], dtype=float), websocket)
        print(f"[{self.site_id}.HDMI.logistic] Sent alpha({alpha.size}), vcov({vcov.shape}), n={n}", flush=True)
        return False

    async def run(self) -> None:
        await self.prepare_data(self.mvar)
        job_id = self.parameters.get("job_id")
        print(f"[async:{self.site_id}] Starting HDMI remote job_id={job_id} mvar(0-based)={self.mvar} method={self.method}", flush=True)
        await super().run()


def run_remote_client(data, central_host, central_port, central_proto, site_id, remote_port=None, config=None):
    if isinstance(data, str):
        D = pd.read_csv(data).values
    else:
        D = data
    config = config or {}
    if "mvar" not in config and "target_column_index" not in config:
        raise ValueError("Config must contain 'mvar' or 'target_column_index' (1-based) for HDMI remote")
    if remote_port is not None:
        print(f"[{site_id}] Ignoring remote_port={remote_port} (no local server)", flush=True)
    print(f"[{site_id}] Starting HDMI remote with data shape={np.asarray(D).shape}", flush=True)
    asyncio.run(async_run_remote_client(data, central_host, central_port, central_proto, site_id, config))


async def async_run_remote_client(data, central_host, central_port, central_proto, site_id, parameters):
    if isinstance(data, str):
        D = pd.read_csv(data).values
    else:
        D = data
    _ = validate_parameters(parameters or {}, np.asarray(D).shape)
    await run_remote_client_async(
        HDMIRemoteClient,
        data=data,
        central_host=central_host,
        central_port=central_port,
        central_proto=central_proto,
        site_id=site_id,
        parameters=parameters or {},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HDMI remote client")
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
