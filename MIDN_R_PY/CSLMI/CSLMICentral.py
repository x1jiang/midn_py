"""
Python implementation of CSLMICentral.R (Communication-Efficient Surrogate Likelihood MI)

R reference functions (see CSLMICentral.R):
- CSLMICentral(D, M, mvar, method, hosts, ports, cent_ports)
- CSLCentralLS(X, y, ...)
- CSLCentralLogit(X, y, ...)

This mirrors the AVGMMI Python structure using the JSON-only WebSocket
transport from Core.transfer and the common central runtime managed externally.
It expects the central server to have already established FastAPI WebSocket
connections to remotes, passing them in via the `websockets` arg.

Key difference from AVGMMI:
- Central sends a local initial estimate beta1 to remotes; remotes return
  average gradients g_k and sample size n_k. Central combines these to form
  the final estimator and imputation covariance.

Returned object from cslmi_central is a list of imputed matrices (np.ndarray),
length M, matching CSLMICentral.R behavior.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Any, Optional

import numpy as np
from fastapi import WebSocket
from scipy.linalg import cholesky, cho_solve, solve_triangular

from Core.transfer import (
    write_string, write_vector, write_integer, read_vector, read_integer,
    WebSocketWrapper, get_wrapped_websocket,
)
from Core.Logit import Logit


# Global state similar to AVGMMI central
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


async def cslmi_central(D: np.ndarray, config: dict | None = None,
                        site_ids: List[str] | None = None,
                        websockets: Dict[str, WebSocket] | None = None) -> List[np.ndarray]:
    """Central CSLMI entrypoint.

    Args:
      D: np.ndarray full data matrix with missing values in target column.
      config: dict with keys:
        - M: number of imputations
        - mvar: 0-based index of missing variable (or 'target_column_index' 1-based)
        - method: 'gaussian' or 'logistic'
        - lam (optional), maxiter (optional)
      site_ids: list of connected remote site IDs (keys for websockets dict)
      websockets: dict mapping site_id -> FastAPI WebSocket

    Returns: list of imputed matrices, length M.
    """
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
        n, p_total = D.shape
        miss = np.isnan(D[:, mvar])
        nm = int(miss.sum())

        X = np.delete(D[~miss, :], mvar, axis=1)
        y = D[~miss, mvar]
        p = X.shape[1]

        if method == 'gaussian':
            CSL = await csl_central_ls(X, y, site_ids, lam=lam)
            # cgram is upper-triangular chol of Gram; use solve_triangular for sampling
        else:
            CSL = await csl_central_logit(X, y, site_ids, lam=lam, maxiter=maxiter)
            # cfisher is upper-triangular chol of Fisher-like matrix

        imputations: List[np.ndarray] = []
        rng = np.random.default_rng()
        for _ in range(M):
            D_imp = D.copy()
            Xmiss = np.delete(D_imp[miss, :], mvar, axis=1)
            if method == 'gaussian':
                # sigma from Inv-Gamma((N+1)/2, (SSE+1)/2)
                alpha_ig = (float(CSL['N']) + 1.0) / 2.0
                beta_ig = (float(CSL['SSE']) + 1.0) / 2.0
                gam = rng.gamma(alpha_ig, 1.0 / beta_ig)
                sig = float(np.sqrt(1.0 / gam))

                z = rng.standard_normal(p)
                delta = solve_triangular(CSL['cgram'], z, lower=False)
                alpha = CSL['beta'] + sig * delta
                D_imp[miss, mvar] = (Xmiss @ alpha).astype(float) + rng.normal(0.0, sig, size=nm)
            else:
                z = rng.standard_normal(p)
                delta = solve_triangular(CSL['cfisher'], z, lower=False)
                alpha = CSL['beta'] + delta
                pr = 1.0 / (1.0 + np.exp(-(Xmiss @ alpha)))
                D_imp[miss, mvar] = (rng.random(nm) < pr).astype(float)

            imputations.append(D_imp)

        return imputations
    finally:
        imputation_running.clear()


async def csl_central_ls(X: np.ndarray, y: np.ndarray, site_ids: List[str],
                         lam: float = 1e-3,
                         websockets: Dict[str, WebSocket] | None = None) -> Dict[str, Any]:
    """Communication-efficient OLS aggregation (CSL style).

    Remote returns average gradient g_k and n_k, central forms final beta and Gram.
    """
    ws_dict = websockets if websockets is not None else remote_websockets

    p = X.shape[1]
    n = X.shape[0]

    XX = X.T @ X + (n * lam) * np.eye(p)
    Xy = X.T @ y
    # Initial estimator
    try:
        U = cholesky(XX, lower=False)
        iXX = cho_solve((U, False), np.eye(p))
    except Exception:
        iXX = np.linalg.pinv(XX)
    beta1 = iXX @ Xy

    N = float(n)
    offset = np.zeros(p, dtype=float)

    # Ask remotes for average gradients at beta1
    for sid in site_ids:
        if sid not in ws_dict:
            continue
        async with site_locks[sid]:
            ws = get_wrapped_websocket(ws_dict[sid], pre_accepted=True)
            await write_string("Gaussian", ws)
            await write_vector(beta1.astype(float), ws)

            nk = await read_integer(ws)
            gk = await read_vector(ws)

            N += float(nk)
            offset += gk

    beta = iXX @ (Xy - n * offset)
    resid = y - X @ beta
    SSE = float(np.sum(resid ** 2) * (N / n))
    gram = XX * (N / n)
    try:
        cgram = cholesky(gram, lower=False)
    except Exception:
        # small jitter if needed
        eps = max(1e-10 * np.trace(gram) / gram.shape[0], 1e-8)
        cgram = cholesky(gram + eps * np.eye(p), lower=False)

    return {"beta": beta, "SSE": SSE, "N": float(N), "cgram": cgram}


async def csl_central_logit(X: np.ndarray, y: np.ndarray, site_ids: List[str],
                            lam: float = 1e-3, maxiter: int = 100,
                            websockets: Dict[str, WebSocket] | None = None) -> Dict[str, Any]:
    """Communication-efficient logistic aggregation (CSL style)."""
    ws_dict = websockets if websockets is not None else remote_websockets

    p = X.shape[1]
    n = X.shape[0]

    fit1 = Logit(X, y, lam=lam, maxiter=maxiter)
    beta1 = fit1['beta']

    N = float(n)
    offset = np.zeros(p, dtype=float)

    for sid in site_ids:
        if sid not in ws_dict:
            continue
        async with site_locks[sid]:
            ws = get_wrapped_websocket(ws_dict[sid], pre_accepted=True)
            await write_string("logistic", ws)
            await write_vector(beta1.astype(float), ws)

            nk = await read_integer(ws)
            gk = await read_vector(ws)
            N += float(nk)
            offset += gk

    fit = Logit(X, y, offset=offset, lam=lam, maxiter=maxiter)
    H = fit['H'] * (N / n)
    try:
        cfisher = cholesky(H, lower=False)
    except Exception:
        eps = max(1e-10 * np.trace(H) / H.shape[0], 1e-8)
        cfisher = cholesky(H + eps * np.eye(p), lower=False)

    return {"beta": fit['beta'], "cfisher": cfisher, "N": float(N)}


if __name__ == "__main__":
    print("This module provides async helpers for CSLMI central and isn't meant to be run directly.")
