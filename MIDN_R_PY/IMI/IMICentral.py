"""
Python implementation of IMICentral.R (Independent MI at Central Only)

R reference functions (see IMICentral.R):
- IMICentral(D, M, mvar, method)
- IMICentralLS(X, y, ...)
- IMICentralLogit(X, y, ...)

This module performs central-only estimation and imputation (no networking).
It exposes imi_central(D, config, ...) compatible with central.app.main importer.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

import numpy as np
from scipy.linalg import cholesky, cho_solve

from Core.Logit import Logit


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


def _imi_central_ls(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> Dict[str, Any]:
    """Central OLS estimator and covariance (no networking), R parity.

    R uses XX = X'X (no ridge). For stability we fall back to pinv/regularization
    if Cholesky fails, preserving outputs shape/meaning.
    """
    p = X.shape[1]
    n = X.shape[0]

    XX = X.T @ X  # R parity: no ridge here
    Xy = X.T @ y
    yy = float(np.sum(y ** 2))

    try:
        U = cholesky(XX, lower=False)
        iXX = cho_solve((U, False), np.eye(p))
    except Exception:
        # Add tiny jitter to allow inversion if singular
        eps = max(1e-10 * (np.trace(XX) / p if p > 0 else 1.0), 1e-8)
        try:
            U = cholesky(XX + eps * np.eye(p), lower=False)
            iXX = cho_solve((U, False), np.eye(p))
        except Exception:
            # Final fallback
            iXX = np.linalg.pinv(XX)

    beta = iXX @ Xy
    SSE = yy + float(beta @ (XX @ beta - 2 * Xy))
    return {"beta": beta, "vcov": iXX, "SSE": SSE, "n": float(n)}


def _imi_central_logit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3, maxiter: int = 100) -> Dict[str, Any]:
    """Central logistic estimator and covariance using Core.Logit."""
    fit = Logit(X, y, lam=lam, maxiter=maxiter)
    beta = fit['beta']
    H = fit['H']
    p = X.shape[1]
    try:
        U = cholesky(H, lower=False)
        iH = cho_solve((U, False), np.eye(p))
    except Exception:
        iH = np.linalg.pinv(H)
    return {"beta": beta, "vcov": iH}


async def imi_central(D: np.ndarray, config: dict | None = None, **_kwargs) -> List[np.ndarray]:
    """Central IMI entrypoint (no remotes).

    Args:
      D: np.ndarray full data matrix with missing values in target column.
      config: dict with keys:
        - M: number of imputations
        - mvar: 0-based index of missing variable (or 'target_column_index' 1-based)
        - method: 'gaussian' or 'logistic'
        - lam (optional), maxiter (optional)
      Note: Additional kwargs (site_ids, websockets, debug, ...) are ignored.

    Returns: list of imputed matrices, length M.
    """
    cfg = _normalize_config(config)
    M = int(cfg['M'])
    mvar = int(cfg['mvar'])
    method = cfg['method']
    lam = float(cfg.get('lam', 1e-3))
    maxiter = int(cfg.get('maxiter', 100))

    n, p_total = D.shape
    miss = np.isnan(D[:, mvar])
    nm = int(miss.sum())

    X = np.delete(D[~miss, :], mvar, axis=1)
    y = D[~miss, mvar]
    p = X.shape[1]

    if method == 'gaussian':
        I = _imi_central_ls(X, y, lam=lam)
        vcov = I['vcov']
        SSE = float(I['SSE'])
        N = float(I['n'])
        beta_hat = I['beta']
    else:
        I = _imi_central_logit(X, y, lam=lam, maxiter=maxiter)
        vcov = I['vcov']
        beta_hat = I['beta']

    # Cholesky factor for sampling
    try:
        U = cholesky(vcov, lower=False)
    except Exception:
        eps = max(1e-10 * np.trace(vcov) / vcov.shape[0], 1e-8)
        U = cholesky(vcov + eps * np.eye(p), lower=False)

    imputations: List[np.ndarray] = []
    rng = np.random.default_rng()
    for _ in range(M):
        D_imp = D.copy()
        Xmiss = np.delete(D_imp[miss, :], mvar, axis=1)
        if method == 'gaussian':
            # sigma from Inv-Gamma((n+1)/2, (SSE+1)/2)
            shape = (N + 1.0) / 2.0
            rate = (SSE + 1.0) / 2.0
            gam = rng.gamma(shape, 1.0 / rate)
            sig = float(np.sqrt(1.0 / gam))

            z = rng.standard_normal(p)
            alpha = beta_hat + sig * (U.T @ z)
            D_imp[miss, mvar] = (Xmiss @ alpha).astype(float) + rng.normal(0.0, sig, size=nm)
        else:
            z = rng.standard_normal(p)
            alpha = beta_hat + (U.T @ z)
            pr = 1.0 / (1.0 + np.exp(-(Xmiss @ alpha)))
            D_imp[miss, mvar] = (rng.random(nm) < pr).astype(float)

        imputations.append(D_imp)

    return imputations


if __name__ == "__main__":
    print("This module provides async helpers for IMI central and isn't meant to be run directly.")
