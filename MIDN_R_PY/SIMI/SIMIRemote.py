"""SIMI remote client (simplified).

Refactored to remove the former FastAPI/uvicorn local server. This pure
WebSocket client connects to the central server at:
    ws://<central_host>:<central_port>/ws/<site_id>
and responds to method instructions: "Gaussian", "logistic", "End", "ping".

Public entry point:
    run_remote_client(data, central_host, central_port, site_id, remote_port, config)
`remote_port` is accepted only for backward compatibility (ignored).
"""

import asyncio
import os
import numpy as np
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from scipy.special import expit
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector,
    read_string, write_string, read_integer, write_integer,
    WebSocketWrapper, get_wrapped_websocket
)
import traceback


async def si_remote_ls(X: np.ndarray, y: np.ndarray, websocket: WebSocketWrapper):
    """Compute and send Gaussian sufficient statistics (n, XX, Xy, yy)."""
    n = X.shape[0]
    XX = X.T @ X
    Xy = X.T @ y
    yy = np.sum(y ** 2)

    # Sanitize
    if np.isnan(XX).any() or np.isinf(XX).any():
        XX = np.nan_to_num(XX)
    if np.isnan(Xy).any() or np.isinf(Xy).any():
        Xy = np.nan_to_num(Xy)

    await write_vector(np.array([float(n)]), websocket)
    await write_matrix(XX, websocket)
    await write_vector(Xy.astype(float), websocket)
    await write_vector(np.array([float(yy)]), websocket)
    print(f"[si_remote_ls] Sent stats: n={n}, p={X.shape[1]}", flush=True)


async def si_remote_logit(X: np.ndarray, y: np.ndarray, websocket: WebSocketWrapper):
    """Logistic mode loop; protocol expects initial n then repeated mode cycles.

    Modes:
      -1 : terminate loop
       0 : CSL offset request (recv beta; send offset; return once)
       1 : SI step send Hessian (H) + gradient (g) + Q
       2 : SI step send Q only (beta already provided)
    """
    n = X.shape[0]
    await write_vector(np.array([float(n)]), websocket)
    print(f"[si_remote_logit] Sent sample size n={n}", flush=True)

    while True:
        try:
            mode = await read_integer(websocket)
        except ValueError as e:
            # Protocol desync safeguard: log and continue trying; do NOT consume further envelopes improperly
            print(f"[si_remote_logit] Failed to read mode integer: {e}", flush=True)
            await asyncio.sleep(0.1)
            continue
        print(f"[si_remote_logit] Mode={mode}", flush=True)
        # Central implementation now uses 1,2 active steps and 0 as termination signal.
        if mode == 0:
            print("[si_remote_logit] Termination (mode 0) received", flush=True)
            break
        if mode == -1:
            print("[si_remote_logit] Legacy termination (-1) received", flush=True)
            break

        # For modes 1 and 2 a beta vector follows.
        beta = await read_vector(websocket)
        xb = X @ beta
        pr = expit(xb)

        # Legacy mode 0 offset phase removed; if reintroduced centrally, needs feature flag.

        low = pr < 0.5
        high = ~low
        Q = np.sum(y * xb)
        if np.any(low):
            Q += np.sum(np.log(np.maximum(1e-10, 1 - pr[low])))
        if np.any(high):
            Q += np.sum(np.log(np.maximum(1e-10, pr[high])) - xb[high])

        if mode == 1:
            w = pr * (1 - pr)
            H = (X.T * w) @ X
            g = X.T @ (y - pr)
            if np.isnan(H).any() or np.isinf(H).any():
                H = np.nan_to_num(H)
            if np.isnan(g).any() or np.isinf(g).any():
                g = np.nan_to_num(g)
            await write_matrix(H, websocket)
            await write_vector(g.astype(float), websocket)
            print("[si_remote_logit] Sent H & g", flush=True)

        await write_vector(np.array([float(Q)]), websocket)
        print(f"[si_remote_logit] Sent Q={Q}", flush=True)


async def remote_kernel(D: np.ndarray, mvar: int, central_host: str, central_port: int, site_id: str):
    """Persistent connection loop with the central server."""
    miss = np.isnan(D[:, mvar])
    X = np.delete(D[~miss], mvar, axis=1)
    y = D[~miss, mvar]
    print(f"[{site_id}] Prepared data X shape={X.shape}, y len={y.shape[0]}", flush=True)

    url = f"ws://{central_host}:{central_port}/ws/{site_id}"
    backoff = 1
    max_backoff = 30

    while True:
        try:
            print(f"[{site_id}] Connecting to {url}", flush=True)
            async with websockets.connect(url, ping_interval=None) as ws:
                wrapped = get_wrapped_websocket(ws)
                print(f"[{site_id}] Connected", flush=True)
                backoff = 1
                while True:
                    try:
                        method = await read_string(wrapped)
                    except asyncio.TimeoutError:
                        try:
                            await write_string("ping", wrapped)
                        except Exception:
                            raise
                        continue

                    if method is None:
                        print(f"[{site_id}] Null method (disconnect?)", flush=True)
                        break

                    m = method.lower()
                    if m == "gaussian":
                        print(f"[{site_id}] Gaussian request", flush=True)
                        await si_remote_ls(X, y, wrapped)
                    elif m == "logistic":
                        print(f"[{site_id}] Logistic request", flush=True)
                        await si_remote_logit(X, y, wrapped)
                    elif m == "end":
                        persist_flag = os.getenv("PERSIST_AFTER_END", "1").lower() in ("1", "true", "yes", "on")
                        if persist_flag:
                            print(f"[{site_id}] End received - persisting (set PERSIST_AFTER_END=0 to exit)", flush=True)
                            continue
                        print(f"[{site_id}] End received - stopping (persistence disabled)", flush=True)
                        return
                    elif m == "ping":
                        await write_string("pong", wrapped)
                    else:
                        print(f"[{site_id}] Unknown method '{method}'", flush=True)

        except asyncio.CancelledError:
            print(f"[{site_id}] Cancelled - shutting down remote kernel", flush=True)
            return
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
            print(f"[{site_id}] Connection closed: {type(e).__name__} - retry in {backoff}s", flush=True)
        except (ConnectionRefusedError, OSError) as e:
            # Central likely not ready yet – suppress noisy traceback
            msg = str(e).splitlines()[0]
            print(f"[{site_id}] Central not ready (will retry): {msg} | next attempt in {backoff}s", flush=True)
        except Exception as e:
            print(f"[{site_id}] Kernel error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            print(f"[{site_id}] Reconnecting in {backoff}s", flush=True)
        await asyncio.sleep(backoff)
        backoff = min(max_backoff, backoff * 2)


def run_remote_client(data, central_host, central_port, site_id, remote_port=None, config=None):
    """Entry point used by orchestrator. remote_port ignored (kept for compatibility)."""
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
    asyncio.run(remote_kernel(D, mvar_py, central_host, central_port, site_id))


# ---------------------------------------------------------------
# New async-friendly API (for in-process task execution)
# ---------------------------------------------------------------
async def async_run_remote_client(data, central_host, central_port, site_id, config):
    """Async variant of run_remote_client.

    Args:
        data: path to CSV or numpy-like matrix
        central_host, central_port, site_id: connection metadata
        config: expects {'mvar': <1-based int>} like sync version
    Returns: coroutine that runs until remote kernel exits/cancelled
    """
    if isinstance(data, str):
        D = pd.read_csv(data).values
    else:
        D = data
    if "mvar" not in config:
        raise ValueError("Config must contain 'mvar' (1-based index) for SIMI remote")
    mvar_py = config["mvar"] - 1
    print(f"[async:{site_id}] Starting SIMI remote task with mvar (0-based)={mvar_py}", flush=True)
    try:
        await remote_kernel(D, mvar_py, central_host, central_port, site_id)
    except asyncio.CancelledError:
        print(f"[async:{site_id}] Cancelled SIMI remote task", flush=True)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIMI remote client (no FastAPI)")
    parser.add_argument("--data", required=True, help="Path to data CSV")
    parser.add_argument("--mvar", type=int, required=True, help="1-based index of missing variable")
    parser.add_argument("--central_host", required=True)
    parser.add_argument("--central_port", type=int, required=True)
    parser.add_argument("--site_id", required=True)
    parser.add_argument("--port", type=int, help="Ignored legacy argument")
    args = parser.parse_args()
    cfg = {"mvar": args.mvar}
    run_remote_client(args.data, args.central_host, args.central_port, args.site_id, args.port, cfg)
