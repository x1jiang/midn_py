"""
Python implementation of HDMICentral.R (Heckman-type Data Missingness Imputation)

R reference functions (see HDMICentral.R):
- HDMICentral(D, M, mvar, method, hosts, ports, cent_ports)
- HDMICentralLS(X, y, ...)
- HDMICentralLogit(X, y, ...)

This mirrors the JSON-only WebSocket transport and central runtime used by
other algorithms. Remotes return packed parameter vectors and covariance
matrices for a joint selection/outcome model; central aggregates by n and
imputes by drawing alpha ~ N(beta, vcov).

Notes:
- Selection model uses a probit link on the missingness indicator.
- Outcome model is linear (Gaussian) or probit (for method "logistic").
- Parameters are packed to match the R convention:
  Gaussian: alpha = [beta_sel (p), beta_out (p-1), log_sigma, atanh(rho)]  (len = 2p+1)
  Logistic: alpha = [beta_sel (p), beta_out (p-1), atanh(rho)]             (len = 2p)
  where rho_star = tanh(rho_param) is used in imputation.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from fastapi import WebSocket
from scipy.stats import norm, multivariate_normal
from scipy.linalg import cholesky

from Core.transfer import (
    write_string, read_vector, read_matrix, WebSocketWrapper, get_wrapped_websocket,
)


# Global state similar to other central implementations
remote_websockets: Dict[str, WebSocket] = {}
imputation_running = asyncio.Event()
site_locks: Dict[str, asyncio.Lock] = {}


def _normalize_config(config: Optional[dict]) -> dict:
    cfg = dict(config or {})

    # mvar normalization: accept either 'mvar' (0- or 1-based) or 'target_column_index' (1-based)
    if 'mvar' in cfg:
        mvar = int(cfg['mvar'])
        if cfg.get('one_based', True):
            mvar -= 1
        cfg['mvar'] = mvar
    elif 'target_column_index' in cfg:
        cfg['mvar'] = int(cfg['target_column_index']) - 1
    else:
        raise ValueError("Config must contain 'mvar' or 'target_column_index'")

    # M normalization
    if 'M' not in cfg:
        cfg['M'] = int(cfg.get('imputation_trials', 5))

    # method
    method = cfg.get('method')
    if method is None:
        method = 'logistic' if bool(cfg.get('is_binary', False)) else 'gaussian'
    method = str(method).lower()
    if method not in ("gaussian", "logistic"):
        raise ValueError("method must be 'gaussian' or 'logistic'")
    cfg['method'] = method

    # defaults
    cfg.setdefault('lam', 1e-3)
    cfg.setdefault('maxiter', 100)
    return cfg


# ---- Simple probit via IRLS (for selection and probit outcome) ----
def _fit_probit(X: np.ndarray, y: np.ndarray, maxiter: int = 50, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Fit probit via IRLS. Returns (beta, cov_beta) with cov approx (X'WX)^-1.

    Uses weights W = phi(eta)^2 / (p*(1-p)) and working response z = eta + (y - p)/phi(eta).
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(maxiter):
        eta = X @ beta
        p_hat = norm.cdf(eta)
        # avoid extreme probabilities
        p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
        phi = norm.pdf(eta)
        # weights and working response
        W = (phi * phi) / (p_hat * (1 - p_hat))
        # guard against zeros
        W = np.clip(W, 1e-8, None)
        z = eta + (y - p_hat) / np.maximum(phi, 1e-8)
        # weighted least squares
        Xw = X * np.sqrt(W[:, None])
        zw = z * np.sqrt(W)
        try:
            XtX = Xw.T @ Xw
            Xtz = Xw.T @ zw
            delta = np.linalg.solve(XtX, Xtz) - beta  # solution minus current beta
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


def _local_hdmi_gaussian(D: np.ndarray, mvar: int, lam: float = 1e-3, maxiter: int = 100) -> Dict[str, Any]:
    """Local-site HDMI Gaussian approximation via probit selection + 2-step Heckman.

    Packs alpha = [beta_sel(p), beta_out(p-1), log_sigma, atanh(rho)] and a diagonal-ish
    covariance matrix reasonable for sampling.
    """
    X_full = np.delete(D, mvar, axis=1)
    y_full = D[:, mvar]
    p = X_full.shape[1]
    # selection indicator over all rows (1 if observed y)
    sel = (~np.isnan(y_full)).astype(float)

    # Probit selection on all rows
    beta_sel, cov_sel = _fit_probit(X_full, sel, maxiter=maxiter)

    # Outcome on observed rows using IMR term
    obs_mask = sel == 1.0
    X_obs = X_full[obs_mask]
    y_obs = y_full[obs_mask]
    # exclusion: last column excluded from outcome model
    X_out_obs = X_obs[:, :p-1]
    eta_sel_obs = X_obs @ beta_sel
    # IMR compatible with R imputation formula: -phi / Phi(-eta)
    imr_obs = - norm.pdf(eta_sel_obs) / np.clip(norm.cdf(-eta_sel_obs), 1e-9, None)
    X_ols = np.column_stack([X_out_obs, imr_obs])  # last column is IMR coefficient δ = σρ
    # OLS
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
    delta = float(coef[-1])  # ≈ σ * ρ
    rho = float(np.clip(delta / (sigma_e + 1e-12), -0.999, 0.999))

    # Pack parameters with transforms
    log_sigma = float(np.log(max(sigma_e, 1e-8)))
    rho_param = float(np.arctanh(rho))  # unconstrained param
    alpha = np.concatenate([beta_sel, beta_out, np.array([log_sigma, rho_param])])

    # Covariance: block-diagonal approx (selection, outcome, log_sigma, rho)
    try:
        cov_out = np.linalg.pinv(XtX)
    except Exception:
        cov_out = np.eye(X_ols.shape[1]) * 1e-2
    cov_beta_out = cov_out[:-1, :-1]
    var_log_sigma = np.array([[0.01]])
    var_rho = np.array([[0.01]])
    cov = np.zeros((2*p + 1, 2*p + 1))
    cov[:p, :p] = cov_sel
    cov[p:2*p-1, p:2*p-1] = cov_beta_out
    cov[2*p-1, 2*p-1] = var_log_sigma[0, 0]
    cov[2*p, 2*p] = var_rho[0, 0]

    return {"beta": alpha, "vcov": cov, "n": float(D.shape[0])}


def _local_hdmi_logistic(D: np.ndarray, mvar: int, lam: float = 1e-3, maxiter: int = 100) -> Dict[str, Any]:
    """Local-site HDMI 'logistic' approximation using probit selection and probit outcome.

    Packs alpha = [beta_sel(p), beta_out(p-1), atanh(rho)] with rho≈0 (no dependence) unless
    clear dependence is estimated (not attempted here). Covariance is diagonal-ish.
    """
    X_full = np.delete(D, mvar, axis=1)
    y_full = D[:, mvar]
    p = X_full.shape[1]
    sel = (~np.isnan(y_full)).astype(float)

    # selection probit with all rows
    beta_sel, cov_sel = _fit_probit(X_full, sel, maxiter=maxiter)

    # outcome probit on observed rows, excluding the last column (exclusion restriction)
    obs_mask = sel == 1.0
    X_obs = X_full[obs_mask]
    y_obs = y_full[obs_mask]
    X_out_obs = X_obs[:, :p-1]
    # Replace NaNs in y_obs if any (shouldn't be)
    y_obs_bin = np.clip(y_obs, 0, 1)
    beta_out, cov_out = _fit_probit(X_out_obs, y_obs_bin, maxiter=maxiter)

    rho_param = 0.0  # assume independence by default
    alpha = np.concatenate([beta_sel, beta_out, np.array([rho_param])])

    cov = np.zeros((2*p, 2*p))
    cov[:p, :p] = cov_sel
    cov[p:2*p-1, p:2*p-1] = cov_out
    cov[2*p-1, 2*p-1] = 0.01

    return {"beta": alpha, "vcov": cov, "n": float(D.shape[0])}


def _pbivnorm(x: np.ndarray, y: np.ndarray, rho: float) -> np.ndarray:
    """Vectorized bivariate normal CDF Φ2(x, y; ρ)."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mean = np.array([0.0, 0.0])
    pts = np.column_stack([x, y])
    # multivariate_normal.cdf expects integration up to (x, y)
    return multivariate_normal.cdf(pts, mean=mean, cov=cov)


async def hdmi_central(D: np.ndarray, config: dict | None = None,
                       site_ids: List[str] | None = None,
                       websockets: Dict[str, WebSocket] | None = None) -> List[np.ndarray]:
    if site_ids is None:
        site_ids = []

    cfg = _normalize_config(config)
    M = int(cfg['M'])
    mvar = int(cfg['mvar'])
    method = cfg['method']
    lam = float(cfg.get('lam', 1e-3))
    maxiter = int(cfg.get('maxiter', 100))

    # Provide websockets (established elsewhere)
    global remote_websockets
    if websockets is not None:
        remote_websockets = websockets

    # Create per-site locks if needed
    global site_locks
    for sid in site_ids:
        site_locks.setdefault(sid, asyncio.Lock())

    # Mark running
    imputation_running.set()

    try:
        # Local parameters
        if method == 'gaussian':
            HD = _local_hdmi_gaussian(D, mvar, lam=lam, maxiter=maxiter)
        else:
            HD = _local_hdmi_logistic(D, mvar, lam=lam, maxiter=maxiter)

        # Pre-scale by n to follow R aggregation pattern
        alpha = HD['beta'] * HD['n']
        vcov = HD['vcov'] * (HD['n'] ** 2)
        N = float(HD['n'])

        # Remote aggregation
        for sid in site_ids:
            if sid not in remote_websockets:
                continue
            async with site_locks[sid]:
                ws = get_wrapped_websocket(remote_websockets[sid], pre_accepted=True)
                await write_string("Gaussian" if method == 'gaussian' else "logistic", ws)

                r_beta = await read_vector(ws)
                r_vcov = await read_matrix(ws)
                r_n = await read_vector(ws)
                rn = float(r_n[0])

                alpha = alpha + r_beta * rn
                vcov = vcov + r_vcov * (rn ** 2)
                N += rn

        # Normalize
        if N <= 0:
            raise ValueError("Total sample size N must be positive in HDMI aggregation")
        beta_bar = alpha / N
        vcov_bar = vcov / (N ** 2)

        # Imputation
        n, p_total = D.shape
        miss = np.isnan(D[:, mvar])
        nm = int(np.sum(miss))
        X_full = np.delete(D, mvar, axis=1)
        p = X_full.shape[1]
        imputations: List[np.ndarray] = []
        rng = np.random.default_rng()

        # Ensure symmetric positive-definite vcov for sampling
        try:
            U = cholesky(vcov_bar, lower=False)
        except Exception:
            # jitter
            eps = max(1e-8, 1e-10 * np.trace(vcov_bar) / max(1, vcov_bar.shape[0]))
            U = cholesky(vcov_bar + eps * np.eye(vcov_bar.shape[0]), lower=False)

        for _ in range(M):
            D_imp = D.copy()
            if nm == 0:
                imputations.append(D_imp)
                continue
            Xmiss = X_full[miss, :]

            z = rng.standard_normal(beta_bar.size)
            alpha_draw = beta_bar + U.T @ z

            if method == 'gaussian':
                beta_sel = alpha_draw[:p]
                beta_out = alpha_draw[p:2*p-1]
                log_sigma = alpha_draw[2*p-1]
                rho_param = alpha_draw[2*p]
                sigma_star = float(np.exp(log_sigma))
                rho_star = float(np.tanh(rho_param))

                var_sel = Xmiss
                var_out = var_sel[:, :p-1]
                sel_lin = var_sel @ beta_sel
                out_lin = var_out @ beta_out
                corr_term = sigma_star * rho_star * (- norm.pdf(sel_lin) / np.clip(norm.cdf(-sel_lin), 1e-12, None))
                D_imp[miss, mvar] = out_lin + corr_term + rng.normal(0.0, sigma_star, size=nm)
            else:
                beta_sel = alpha_draw[:p]
                beta_out = alpha_draw[p:2*p-1]
                rho_param = alpha_draw[2*p-1]
                rho_star = float(np.tanh(rho_param))

                var_sel = Xmiss
                var_out = var_sel[:, :p-1]
                sel_lin = var_sel @ beta_sel
                out_lin = var_out @ beta_out
                num = _pbivnorm(out_lin, -sel_lin, -rho_star)
                den = np.clip(norm.cdf(-sel_lin), 1e-9, None)
                p_star = np.clip(num / den, 1e-6, 1 - 1e-6)
                D_imp[miss, mvar] = (rng.random(nm) < p_star).astype(float)

            imputations.append(D_imp)

        return imputations
    finally:
        imputation_running.clear()


if __name__ == "__main__":
    print("This module provides async helpers for HDMI central and isn't meant to be run directly.")
