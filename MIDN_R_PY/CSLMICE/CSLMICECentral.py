"""
Python implementation of CSLMICECentral.R (CSL within MICE)

CSLMICE combines chained equations with Communication-Efficient Surrogate
Likelihood (CSL) aggregation across sites. It mirrors AVGMMICE's Python
structure but uses:
- Core.LS.CSLLSNet for Gaussian targets
- Core.Logit.CSLLogitNet for logistic targets

Transport: JSON-only WebSockets via Core.transfer; central receives
WebSocket connections from the outer server and exchanges messages directly
with remotes. No local socket servers are used.

Entry point: cslmice_central(D, config, site_ids, websockets)
Returns: list of imputed datasets (np.ndarray), length M
"""

from __future__ import annotations

import os
import time
import asyncio
from typing import Dict, List, Any, Optional

import numpy as np
from fastapi import WebSocket
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular

from Core.transfer import (
    write_string, write_vector, get_wrapped_websocket,
)
from Core.LS import CSLLSNet, ImputeLS
from Core.Logit import CSLLogitNet, ImputeLogit


# Global state used during a run (provided by outer central server)
remote_websockets: Dict[str, WebSocket] = {}
site_locks: Dict[str, asyncio.Lock] = {}
imputation_running = asyncio.Event()


def _parse_index_list(val: Any) -> List[int]:
    """Parse an index list that could be 0-based list, 1-based list, or CSV string."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = list(map(int, list(val)))
    elif isinstance(val, str):
        parts = [p.strip() for p in val.split(',') if p.strip()]
        arr = list(map(int, parts))
    else:
        arr = [int(val)]
    return arr


def _normalize_config(config: Optional[dict]) -> dict:
    raw = dict(config or {})
    norm: Dict[str, Any] = {}

    # mvar: accept 'mvar' (0- or 1-based, list), or 'target_column_indexes' (1-based)
    if 'mvar' in raw:
        mvar = _parse_index_list(raw['mvar'])
        one_based = raw.get('one_based')
        if one_based is None:
            one_based = False
        if one_based:
            mvar = [i - 1 for i in mvar]
        norm['mvar'] = mvar
    elif 'target_column_indexes' in raw:
        mvar = _parse_index_list(raw['target_column_indexes'])
        norm['mvar'] = [i - 1 for i in mvar]
    else:
        raise ValueError("Config must include 'mvar' (0-based) or 'target_column_indexes' (1-based)")

    # type_list: explicit or derive from is_binary_list
    tlist = raw.get('type_list')
    if tlist is not None:
        tl = [str(x).lower() for x in tlist]
    else:
        bins = raw.get('is_binary_list') or []
        bl = [bool(x) for x in bins]
        tl = ["logistic" if b else "gaussian" for b in bl]
    if not tl:
        raise ValueError("type_list or is_binary_list required")
    norm['type_list'] = tl

    # iterations
    if 'iter_val' in raw:
        norm['iter_val'] = int(raw['iter_val'])
    elif 'iteration_between_imputations' in raw:
        norm['iter_val'] = int(raw['iteration_between_imputations'])
    else:
        norm['iter_val'] = 1
    if 'iter0_val' in raw:
        norm['iter0_val'] = int(raw['iter0_val'])
    elif 'iteration_before_first_imputation' in raw:
        norm['iter0_val'] = int(raw['iteration_before_first_imputation'])
    else:
        norm['iter0_val'] = norm['iter_val']

    # M
    if 'M' in raw:
        norm['M'] = int(raw['M'])
    elif 'imputation_trials' in raw:
        norm['M'] = int(raw['imputation_trials'])
    else:
        norm['M'] = 5

    # optional solver params
    norm['lam'] = float(raw.get('lam', 1e-3))
    norm['maxiter'] = int(raw.get('maxiter', 100))

    return norm


async def _initialize_remote_sites(mvar_1based: List[int], site_ids: List[str]) -> None:
    """Send Initialize + mvar vector to all connected remotes."""
    for sid in list(site_ids):
        if sid not in remote_websockets:
            continue
        async with site_locks[sid]:
            ws = get_wrapped_websocket(remote_websockets[sid], pre_accepted=True)
            await write_string("Initialize", ws)
            await write_vector(np.array(mvar_1based, dtype=float), ws)


async def _finalize_remote_sites(site_ids: List[str]) -> None:
    for sid in list(site_ids):
        if sid not in remote_websockets:
            continue
        try:
            async with site_locks[sid]:
                ws = get_wrapped_websocket(remote_websockets[sid], pre_accepted=True)
                await write_string("End", ws)
        except Exception:
            pass


async def cslmice_central(D: np.ndarray,
                           config: Optional[dict] = None,
                           site_ids: Optional[List[str]] = None,
                           websockets: Optional[Dict[str, WebSocket]] = None,
                           debug: bool = False) -> List[np.ndarray]:
    """CSLMICE central algorithm (async).

    Args:
      D: np.ndarray (n x p) raw data matrix with NaNs.
      config: dict with keys {'M','mvar','type_list','iter_val','iter0_val', ...}
      site_ids: list of remote site IDs
      websockets: dict site_id -> WebSocket (already connected)
    Returns: list of imputed datasets (length M)
    """
    cfg = _normalize_config(config)
    mvar = list(cfg['mvar'])
    type_list = list(cfg['type_list'])
    M = int(cfg['M'])
    iter_val = int(cfg['iter_val'])
    iter0_val = int(cfg['iter0_val'])
    lam = float(cfg['lam'])
    maxiter = int(cfg['maxiter'])

    if site_ids is None:
        site_ids = []

    # Provide websockets/locks
    global remote_websockets, site_locks
    if websockets is not None:
        remote_websockets = websockets
    for sid in site_ids:
        site_locks.setdefault(sid, asyncio.Lock())

    # Derive active site list; fallback to all connected if needed
    active_sites = [sid for sid in (site_ids or []) if sid in remote_websockets]
    if not active_sites and remote_websockets:
        active_sites = list(remote_websockets.keys())
        if debug:
            print(f"[CSLMICE] No matching site_ids provided; using all connected: {active_sites}", flush=True)
    if debug:
        print(f"[CSLMICE] Active sites: {active_sites}", flush=True)

    # Start run
    imputation_running.set()
    start = time.time()

    # Build augmented matrix with intercept
    D_aug = np.column_stack([D.astype(float, copy=False), np.ones((D.shape[0],), dtype=float)])
    n, p_aug = D_aug.shape
    miss = np.isnan(D_aug)
    l = len(mvar)

    # Initialize with column means for missing cells in target columns
    for j in mvar:
        idx1 = np.where(miss[:, j])[0]
        if idx1.size == 0:
            continue
        idx0 = np.where(~miss[:, j])[0]
        mu = float(np.nanmean(D_aug[idx0, j])) if idx0.size else 0.0
        D_aug[idx1, j] = mu

    # Tell remotes which columns are targets (1-based indices per legacy wire format)
    if debug:
        print(f"[CSLMICE] Initializing remotes with targets (1-based) {[j+1 for j in mvar]}", flush=True)
    await _initialize_remote_sites([j + 1 for j in mvar], active_sites)

    imputations: List[np.ndarray] = []
    current_iters = iter0_val

    # RNG seed (optional deterministic)
    seed_env = os.getenv("GLOBAL_SEED") or os.getenv("SIMICE_GLOBAL_SEED")
    if seed_env:
        try:
            np.random.seed(int(seed_env))
        except Exception:
            pass

    try:
        for m in range(1, M + 1):
            if debug:
                print(f"[CSLMICE][imputation {m}/{M}] iterations={current_iters}", flush=True)
            for it in range(1, current_iters + 1):
                if debug:
                    print(f"[CSLMICE] Iteration {it}/{current_iters}", flush=True)
                for i, j in enumerate(mvar):
                    if debug:
                        print(f"[CSLMICE] Var j={j} type={type_list[i]}", flush=True)
                    midx = np.where(miss[:, j])[0]
                    if midx.size == 0:
                        continue
                    cidx = np.where(~miss[:, j])[0]
                    if type_list[i].lower() == "gaussian":
                        if debug:
                            print(f"[CSLMICE] Requesting Gaussian info from remotes (CSL)...", flush=True)
                        fit = await CSLLSNet(D_aug, list(cidx), j,
                                              remote_websockets=remote_websockets,
                                              site_locks=site_locks,
                                              site_ids=active_sites,
                                              lam=lam)
                        # Sample sigma and coefficients via cgram
                        shape = (float(fit['N']) + 1.0) / 2.0
                        rate = (float(fit['SSE']) + 1.0) / 2.0
                        g = np.random.default_rng().gamma(shape=shape, scale=1.0 / rate)
                        sig = float(np.sqrt(1.0 / g))
                        cgram = np.asarray(fit['cgram'])
                        z = np.random.normal(size=p_aug - 1)
                        delta = solve_triangular(cgram, z, lower=False)
                        alpha = np.asarray(fit['beta']) + sig * delta
                        # Use all columns except j (including intercept) as regressors
                        Xmiss = np.delete(D_aug[midx, :], j, axis=1)
                        D_aug[midx, j] = Xmiss @ alpha + np.random.normal(0.0, sig, size=midx.size)

                        # Notify remotes
                        if debug:
                            print(f"[CSLMICE] Sending Gaussian impute to remotes...", flush=True)
                        await ImputeLS(j + 1, alpha.astype(np.float64), float(sig),
                                       remote_websockets=remote_websockets,
                                       site_locks=site_locks,
                                       site_ids=active_sites)
                    else:  # logistic
                        if debug:
                            print(f"[CSLMICE] Requesting Logistic info from remotes (CSL)...", flush=True)
                        fit = await CSLLogitNet(D_aug, list(cidx), j,
                                                remote_websockets=remote_websockets,
                                                site_locks=site_locks,
                                                site_ids=active_sites,
                                                lam=lam,
                                                maxiter=maxiter)
                        H = np.asarray(fit['H'])
                        try:
                            cH = cholesky(H, lower=False)
                        except Exception:
                            eps = max(1e-10 * np.trace(H) / max(H.shape[0], 1), 1e-8)
                            cH = cholesky(H + eps * np.eye(H.shape[0]), lower=False)
                        z = np.random.normal(size=p_aug - 1)
                        delta = solve_triangular(cH, z, lower=False)
                        alpha = np.asarray(fit['beta']) + delta
                        Xmiss = np.delete(D_aug[midx, :], j, axis=1)
                        pr = 1.0 / (1.0 + np.exp(-(Xmiss @ alpha)))
                        D_aug[midx, j] = (np.random.random(size=midx.size) < pr).astype(float)

                        if debug:
                            print(f"[CSLMICE] Sending Logistic impute to remotes...", flush=True)
                        await ImputeLogit(j + 1, alpha.astype(np.float64),
                                          remote_websockets=remote_websockets,
                                          site_locks=site_locks,
                                          site_ids=active_sites)

            # Save one imputed copy (drop intercept before returning)
            imputations.append(D_aug[:, : D_aug.shape[1] - 1].copy())
            # After first extraction, switch to regular iteration count
            current_iters = iter_val

        return imputations
    finally:
        await _finalize_remote_sites(site_ids)
        imputation_running.clear()
        if debug:
            dur = time.time() - start
            print(f"[CSLMICE] Completed {M} imputations in {dur:.3f}s", flush=True)
