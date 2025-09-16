"""SIMICE remote client (simplified, no FastAPI/uvicorn).

This refactor removes the previous FastAPI wrapper. The remote now runs a
persistent WebSocket client loop that connects to:
        ws://<central_host>:<central_port>/ws/<site_id>

Protocol commands handled (from central):
    Initialize  -> receive mvar list (1-based), prepare DD, store original missing map
    Information -> receive method + variable index (1-based) and optionally mode/beta
    Impute      -> perform imputation with provided parameters only for originally missing cells
    End         -> terminate
    ping        -> respond with pong

Reconnection with exponential backoff (capped) is preserved. GLOBAL_SEED env
variable seeding retained.
"""

import numpy as np
import pandas as pd
import asyncio
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from scipy.special import expit
import time
from datetime import datetime
import os
import traceback
from Core.transfer import (
        read_matrix, write_matrix, read_vector, write_vector,
        read_string, write_string, read_integer, write_integer,
        read_number,
        WebSocketWrapper, get_wrapped_websocket
)

# Optional verbose protocol debugging (set SIMICE_DEBUG_PROTOCOL=1)
DEBUG_PROTOCOL = os.getenv("SIMICE_DEBUG_PROTOCOL", "0") == "1"

async def _read_mvar_vector_with_retry(wrapped_ws, site_id, max_retries: int = 1):
    """Read the mvar vector, tolerating one stray string frame (mis-ordered message).

    Rationale: Rare race condition observed where a STR/utf8 frame arrives
    where the remote expects the VEC/float64 mvar vector immediately after
    receiving the "Initialize" instruction. We swallow at most one such
    frame and retry reading the vector to re-synchronize.
    """
    try:
        return await read_vector(wrapped_ws)
    except ValueError as e:
        msg = str(e)
        if "Expected VEC/float64" in msg and max_retries > 0:
            if DEBUG_PROTOCOL:
                print(f"[{site_id}] WARN: Expected mvar vector but got STR; attempting resync (1 retry)", flush=True)
            # Consume (already consumed by read_vector), just retry once
            return await read_vector(wrapped_ws)
        raise


async def si_remote_ls(X, y, wrapped_ws, site_id):
    """Implementation of SIRemoteLS"""
    p = X.shape[1]
    n = X.shape[0]

    # Calculate sufficient statistics
    XX = X.T @ X
    Xy = X.T @ y
    yy = np.sum(y**2)

    # Send to central
    await write_vector(np.array([n]), wrapped_ws)
    await write_matrix(XX, wrapped_ws)
    await write_vector(Xy, wrapped_ws)
    await write_vector(np.array([yy]), wrapped_ws)

async def si_remote_logit(X, y, wrapped_ws, beta, site_id, mode=1):
    """Implementation of SIRemoteLogit"""
    p = X.shape[1]
    n = X.shape[0]

    xb = X @ beta
    pr = expit(xb)

    Q = np.sum(y * xb)
    low_pr_mask = pr < 0.5
    Q += np.sum(np.log(1 - pr[low_pr_mask]))
    high_pr_mask = ~low_pr_mask
    Q += np.sum(np.log(pr[high_pr_mask]) - xb[high_pr_mask])

    if mode == 1:
        W = pr * (1 - pr)
        H = X.T @ (X * W[:, np.newaxis])
        g = X.T @ (y - pr)
        await write_matrix(H, wrapped_ws)
        await write_vector(g, wrapped_ws)
    #print(f"[{site_id}][Information][Logistic] n={n}, p={p}, mode={mode}, Q={Q:.4f}", flush=True)
    await write_vector(np.array([Q]), wrapped_ws)

async def remote_kernel(D, central_host, central_port, site_id):
    """Persistent connection loop implementing SIMICE remote behavior (simplified, config ignored)."""
    # Add intercept column
    D = np.column_stack([D, np.ones(D.shape[0])]).astype(np.float64, copy=False)
    p = D.shape[1]

    # Optional seeding
    seed_env = os.getenv("GLOBAL_SEED")
    if seed_env:
        try:
            np.random.seed(int(seed_env))
            print(f"[{site_id}] GLOBAL_SEED set; using seed {int(seed_env)}", flush=True)
        except Exception:
            pass

    state = {
        "connected": False,
        "DD": None,
        "mvar": [],
        "p": p,
        "orig_missing_map": {}
    }

    url = f"ws://{central_host}:{central_port}/ws/{site_id}"
    backoff = 1
    max_backoff = 30

    while True:
        try:
            print(f"[{site_id}] Connecting to central at {url}", flush=True)
            async with websockets.connect(url, ping_interval=None) as websocket:
                wrapped_ws = get_wrapped_websocket(websocket)
                state["connected"] = True
                print(f"[{site_id}] Connected", flush=True)
                backoff = 1
                consecutive_errors = 0
                max_consecutive_errors = 3

                while consecutive_errors < max_consecutive_errors:
                    try:
                        inst = await asyncio.wait_for(read_string(wrapped_ws), timeout=30.0)
                        if inst is None:
                            print(f"[{site_id}] Null instruction; breaking", flush=True)
                            break
                        if inst == "Initialize":
                            try:
                                if DEBUG_PROTOCOL:
                                    print(f"[{site_id}] Initialize received; expecting mvar vector next", flush=True)
                                mvar_vec = await _read_mvar_vector_with_retry(wrapped_ws, site_id)
                                mvar = [int(idx) - 1 for idx in mvar_vec]
                                state["mvar"] = mvar
                                miss = np.isnan(D)
                                check_cols = [i for i in range(p) if i not in mvar and i != p-1]
                                valid_rows = np.sum(miss[:, check_cols], axis=1) == 0
                                DD = D[valid_rows].copy()
                                miss_DD = np.isnan(DD)
                                orig_missing_map = {}
                                for j in mvar:
                                    miss_j = miss_DD[:, j]
                                    orig_missing_map[j] = miss_j.copy()
                                    if np.any(miss_j):
                                        DD[miss_j, j] = np.mean(DD[~miss_j, j])
                                state["DD"] = DD
                                state["orig_missing_map"] = orig_missing_map
                                print(f"[{site_id}] Initialized mvar={mvar}", flush=True)
                                for j in mvar:
                                    print(f"[{site_id}] original miss count j={j} -> {orig_missing_map[j].sum()}")
                            except Exception as e:
                                print(f"[{site_id}] Initialize error: {type(e).__name__}: {e}", flush=True)
                                raise
                        elif inst == "Information":
                            try:
                                if DEBUG_PROTOCOL:
                                    print(f"[{site_id}] Information instruction start", flush=True)
                                method = await read_string(wrapped_ws)
                                j_r = await read_integer(wrapped_ws)
                                j = j_r - 1
                                # Optional consistency warning
                                DD = state["DD"]
                                orig_missing = state.get("orig_missing_map", {})
                                if j in orig_missing:
                                    valid_idx = ~orig_missing[j]
                                else:
                                    valid_idx = ~np.isnan(DD)[:, j]
                                X = np.delete(DD[valid_idx], j, axis=1)
                                y = DD[valid_idx, j]
                                if DEBUG_PROTOCOL:
                                    print(f"[{site_id}] Information method={method} j={j} DD_shape={None if DD is None else DD.shape}", flush=True)
                                if method.lower() == "gaussian":
                                    await si_remote_ls(X, y, wrapped_ws, site_id)
                                elif method.lower() == "logistic":
                                    mode = await read_integer(wrapped_ws)
                                    beta = await read_vector(wrapped_ws)
                                    if DEBUG_PROTOCOL:
                                        print(f"[{site_id}] Logistic info: mode={mode} beta_len={len(beta)} n_local={X.shape[0]}", flush=True)
                                    await write_vector(np.array([X.shape[0]]), wrapped_ws)
                                    await si_remote_logit(X, y, wrapped_ws, beta, site_id, mode)
                                else:
                                    print(f"[{site_id}] Unknown method in Information: {method}")
                            except Exception as e:
                                print(f"[{site_id}] Information error: {type(e).__name__}: {e}")
                                raise
                        elif inst == "Impute":
                            try:
                                method = await read_string(wrapped_ws)
                                print(f"[{site_id}] Impute method={method}", flush=True)
                                j_r = await read_integer(wrapped_ws)
                                j = j_r - 1
                                DD = state["DD"]
                                orig_missing = state.get("orig_missing_map", {})
                                if j in orig_missing:
                                    midx = np.where(orig_missing[j])[0]
                                else:
                                    midx = np.where(np.isnan(DD)[:, j])[0]
                                nmidx = len(midx)
                                if method.lower() == "gaussian":
                                    beta = await read_vector(wrapped_ws)
                                    sigv = await read_vector(wrapped_ws)
                                    sig = float(sigv[0])
                                    if DEBUG_PROTOCOL:
                                        print(f"[{site_id}] Impute Gaussian j={j} nmidx={nmidx} beta_len={len(beta)} sig={sig}", flush=True)
                                elif method.lower() == "logistic":
                                    alpha = await read_vector(wrapped_ws)
                                    if DEBUG_PROTOCOL:
                                        print(f"[{site_id}] Impute Logistic j={j} nmidx={nmidx} alpha_len={len(alpha)}", flush=True)
                                else:
                                    print(f"[{site_id}] Unknown impute method: {method}")
                                    continue
                                if nmidx == 0:
                                    print(f"[{site_id}] No original missing at j={j}; discard payload")
                                    continue
                                X = np.delete(DD[midx, :], j, axis=1)
                                if method.lower() == "gaussian":
                                    DD[midx, j] = X @ beta + np.random.normal(0.0, sig, size=nmidx)
                                else:
                                    pr = 1.0 / (1.0 + np.exp(-(X @ alpha)))
                                    DD[midx, j] = np.random.binomial(1, pr)
                                state["DD"] = DD
                                if DEBUG_PROTOCOL:
                                    print(f"[{site_id}] Impute update complete j={j}", flush=True)
                            except Exception as e:
                                print(f"[{site_id}] Impute error: {type(e).__name__}: {e}")
                        elif inst == "End":
                            # Previously: terminate remote kernel. New behavior: optionally persist and await further instructions.
                            persist_flag = os.getenv("PERSIST_AFTER_END", "1").lower() in ("1", "true", "yes", "on")
                            if persist_flag:
                                # Reset minimal state but keep connection open so a subsequent central job can reuse this remote.
                                print(f"[{site_id}] End received; persisting (set PERSIST_AFTER_END=0 to exit)", flush=True)
                                # Clear imputation-specific state but retain DD so future rounds can decide whether to reinitialize.
                                # If a full reset is desired, clear DD as well.
                                state["mvar"] = []
                                # Do not return; continue waiting for next instruction
                                consecutive_errors = 0
                                continue
                            else:
                                print(f"[{site_id}] End received; exiting (PERSIST_AFTER_END disabled)", flush=True)
                                return
                        elif inst == "ping":
                            await write_string("pong", wrapped_ws)
                        else:
                            print(f"[{site_id}] Unknown instruction: {inst}")
                        consecutive_errors = 0
                    except asyncio.TimeoutError:
                        try:
                            await write_string("ping", wrapped_ws)
                        except Exception:
                            print(f"[{site_id}] Ping failed; breaking", flush=True)
                            break
                        consecutive_errors += 1
                    except ValueError as e:
                        print(f"[{site_id}] Protocol error: {e}")
                        consecutive_errors += 1
                    except Exception as e:
                        print(f"[{site_id}] Processing error: {type(e).__name__}: {e}")
                        consecutive_errors += 1
                        if isinstance(e, (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK)):
                            break
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[{site_id}] Too many consecutive errors ({consecutive_errors}); reconnecting")
                        break
        except asyncio.CancelledError:
            print(f"[{site_id}] Cancelled - shutting down remote kernel", flush=True)
            return
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"[{site_id}] Invalid status code: {e}")
        except (ConnectionRefusedError, OSError) as e:
            msg = str(e).splitlines()[0]
            print(f"[{site_id}] Central not ready (will retry): {msg}")
        except Exception as e:
            print(f"[{site_id}] Connection error: {type(e).__name__}: {e}")
            traceback.print_exc()
        print(f"[{site_id}] Reconnecting in {backoff}s", flush=True)
        await asyncio.sleep(backoff)
        backoff = min(max_backoff, backoff * 2)

def run_remote_client(data_file, central_host, central_port, site_id=None, remote_port=None, config=None):
    """Launch SIMICE remote (remote_port kept for compatibility, config accepted but ignored)."""
    if isinstance(data_file, str):
        D = pd.read_csv(data_file).values
    else:
        D = data_file
    if site_id is None:
        site_id = "remote1"
    if remote_port is not None:
        print(f"[{site_id}] Ignoring remote_port={remote_port} (no local server)", flush=True)
    asyncio.run(remote_kernel(D, central_host, central_port, site_id))


# ---------------------------------------------------------------
# New async-friendly API (for in-process task execution)
# ---------------------------------------------------------------
async def async_run_remote_client(data_file, central_host, central_port, site_id=None, config=None):
    """Async variant of run_remote_client for SIMICE. Config accepted but ignored."""
    if isinstance(data_file, str):
        D = pd.read_csv(data_file).values
    else:
        D = data_file
    if site_id is None:
        site_id = "remote1"
    print(f"[async:{site_id}] Starting SIMICE remote task (config ignored)", flush=True)
    try:
        await remote_kernel(D, central_host, central_port, site_id)
    except asyncio.CancelledError:
        print(f"[async:{site_id}] Cancelled SIMICE remote task", flush=True)
        raise

 