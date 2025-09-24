"""
Python implementation of AVGMMICentral.R (Average Model Imputation)

R reference functions (see AVGMMICentral.R):
- AVGMMICentral(D, M, mvar, method, hosts, ports, cent_ports)
- AVGMCentralLS(X, y, ...)
- AVGMCentralLogit(X, y, ...)

This Python module mirrors the SIMI central architecture using the JSON-only
WebSocket transport from Core.transfer and the common central runtime managed
externally. It expects the central server to have already established FastAPI
WebSocket connections to remotes, passing them in via the `websockets` arg.

Differences from SIMI:
- Remotes return condensed sufficient statistics instead of raw XX/Xy/yy for
  Gaussian; and (beta, H, n) for logistic (one-shot, no iterative RPC loop).

Returned object from avgmmi_central is a list of imputed matrices (np.ndarray),
length M, matching AVGMMICentral.R behavior.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Any, Optional

import numpy as np
from fastapi import WebSocket
from scipy.linalg import cholesky, cho_solve

from Core.transfer import (
    write_string, read_vector, read_matrix, WebSocketWrapper, get_wrapped_websocket,
)
from Core.Logit import Logit


# Global state similar to SIMI central
remote_websockets: Dict[str, WebSocket] = {}
imputation_running = asyncio.Event()
site_locks: Dict[str, asyncio.Lock] = {}


def _normalize_config(config: Optional[dict]) -> dict:
    cfg = dict(config or {})

    # mvar normalization: accept either 'mvar' (0- or 1-based) or 'target_column_index' (1-based)
    if 'mvar' in cfg:
        mvar = int(cfg['mvar'])
        # Heuristic: treat mvar as 1-based if >= 1 and caller indicates so; otherwise require 0-based
        # Prefer explicit 'one_based' flag if present
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


async def avgmmi_central(D: np.ndarray, config: dict | None = None,
                         site_ids: List[str] | None = None,
                         websockets: Dict[str, WebSocket] | None = None) -> List[np.ndarray]:
    """Central AVGMMI entrypoint.

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
            AVGM = await avgm_central_ls(X, y, site_ids, lam=lam)
        else:
            AVGM = await avgm_central_logit(X, y, site_ids, lam=lam, maxiter=maxiter)

        # Cholesky of covariance
        try:
            U = cholesky(AVGM['vcov'], lower=False)  # upper U, so U.T @ z ~ N(0, vcov)
        except Exception:
            # Fallback: add tiny jitter
            eps = max(1e-10 * np.trace(AVGM['vcov']) / AVGM['vcov'].shape[0], 1e-8)
            U = cholesky(AVGM['vcov'] + eps * np.eye(p), lower=False)

        imputations: List[np.ndarray] = []
        for _ in range(M):
            D_imp = D.copy()
            if method == 'gaussian':
                # R: sig = sqrt(1/rgamma(1,(n+1)/2,(SSE+1)/2))
                shape = (AVGM['n'] + 1.0) / 2.0
                rate = (AVGM['SSE'] + 1.0) / 2.0
                g = np.random.default_rng().gamma(shape=shape, scale=1.0 / rate)
                sig = float(np.sqrt(1.0 / g))
                alpha = AVGM['beta'] + sig * (U.T @ np.random.normal(size=p))
                D_imp[miss, mvar] = D_imp[miss, :][:, np.arange(p_total) != mvar] @ alpha + \
                                    np.random.normal(loc=0.0, scale=sig, size=nm)
            else:  # logistic
                alpha = AVGM['beta'] + (U.T @ np.random.normal(size=p))
                xb = D_imp[miss, :][:, np.arange(p_total) != mvar] @ alpha
                pr = 1.0 / (1.0 + np.exp(-xb))
                D_imp[miss, mvar] = (np.random.random(size=nm) < pr).astype(float)

            imputations.append(D_imp)

        return imputations
    finally:
        imputation_running.clear()


async def avgm_central_ls(X: np.ndarray, y: np.ndarray, site_ids: List[str],
                          lam: float = 1e-3,
                          websockets: Dict[str, WebSocket] | None = None) -> Dict[str, Any]:
    """Aggregate site-level OLS estimates (AVGMMI style).

    Local site computes beta, iFisher = iXX @ XX @ iXX, SSE; central sums
    beta*n and iFisher*n^2, SSE, and n across sites, then normalizes.
    """
    ws_dict = websockets if websockets is not None else remote_websockets

    p = X.shape[1]
    n = X.shape[0]

    # Local stats
    XX = X.T @ X + lam * np.eye(p)
    Xy = X.T @ y
    yy = float(np.sum(y ** 2))
    U = cholesky(XX, lower=False)
    iXX = cho_solve((U, False), np.eye(p))
    beta = iXX @ Xy
    SSE = yy + float(beta @ (XX @ beta - 2 * Xy))
    iFisher = iXX @ XX @ iXX

    AVGM = {
        'beta': beta * n,
        'vcov': iFisher * (n ** 2),
        'SSE': SSE,
        'n': float(n),
    }

    # Remote aggregation
    for sid in site_ids:
        if sid not in ws_dict:
            continue
        async with site_locks[sid]:
            ws = get_wrapped_websocket(ws_dict[sid], pre_accepted=True)
            await write_string("Gaussian", ws)

            r_beta = await read_vector(ws)
            r_iF = await read_matrix(ws)
            r_SSE = await read_vector(ws)
            r_n = await read_vector(ws)

            rn = float(r_n[0])
            AVGM['beta'] = AVGM['beta'] + r_beta * rn
            AVGM['vcov'] = AVGM['vcov'] + r_iF * (rn ** 2)
            AVGM['SSE'] = AVGM['SSE'] + float(r_SSE[0])
            AVGM['n'] = AVGM['n'] + rn

    # Normalize
    N = float(AVGM['n']) if AVGM['n'] else 1.0
    AVGM['beta'] = AVGM['beta'] / N
    AVGM['vcov'] = AVGM['vcov'] / (N ** 2)
    return AVGM


async def avgm_central_logit(X: np.ndarray, y: np.ndarray, site_ids: List[str],
                             lam: float = 1e-3, maxiter: int = 100,
                             websockets: Dict[str, WebSocket] | None = None) -> Dict[str, Any]:
    """Aggregate site-level logistic estimates (AVGMMI style).

    Local site computes (beta, H); central sums beta*n and inv(H)*n^2 across
    sites, then normalizes.
    """
    ws_dict = websockets if websockets is not None else remote_websockets

    p = X.shape[1]
    n = X.shape[0]

    fit = Logit(X, y, lam=lam, maxiter=maxiter)
    beta_loc = fit['beta']
    H_loc = fit['H']
    # inv(H) via cholesky for stability
    try:
        U = cholesky(H_loc, lower=False)
        iH_loc = cho_solve((U, False), np.eye(p))
    except Exception:
        iH_loc = np.linalg.pinv(H_loc)

    AVGM = {
        'beta': beta_loc * n,
        'vcov': iH_loc * (n ** 2),
        'n': float(n),
    }

    for sid in site_ids:
        if sid not in ws_dict:
            continue
        async with site_locks[sid]:
            ws = get_wrapped_websocket(ws_dict[sid], pre_accepted=True)
            await write_string("logistic", ws)

            r_beta = await read_vector(ws)
            r_H = await read_matrix(ws)
            r_n = await read_vector(ws)

            rn = float(r_n[0])
            # inv(H) for remote
            try:
                U_r = cholesky(r_H, lower=False)
                iH_r = cho_solve((U_r, False), np.eye(p))
            except Exception:
                iH_r = np.linalg.pinv(r_H)

            AVGM['beta'] = AVGM['beta'] + r_beta * rn
            AVGM['vcov'] = AVGM['vcov'] + iH_r * (rn ** 2)
            AVGM['n'] = AVGM['n'] + rn

    # Normalize
    N = float(AVGM['n']) if AVGM['n'] else 1.0
    AVGM['beta'] = AVGM['beta'] / N
    AVGM['vcov'] = AVGM['vcov'] / (N ** 2)
    return AVGM


if __name__ == "__main__":
    print("This module provides async helpers for AVGMMI central and isn't meant to be run directly.")
