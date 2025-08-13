#!/usr/bin/env python3

"""
SIMIRemote.py - SIMI Remote site implementation (wired for FastAPI orchestration)
- Keeps original CLI/websocket server behavior in SIMIRemote_base.py
- Provides reusable functions used by remote FastAPI app to talk to central
"""

import sys
import os
import asyncio
import numpy as np
import websockets
import json
import time
import pandas as pd

async def compute_ls_stats(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute local least-squares sufficient statistics.
    Returns a dict with n, XX, Xy, yy as numpy arrays/scalars.
    """
    n = X.shape[0]
    XX = np.matmul(X.T, X)
    Xy = np.matmul(X.T, y)
    yy = float(np.sum(y ** 2))
    return {
        'n': float(n),
        'XX': XX,
        'Xy': Xy,
        'yy': yy,
    }

async def SIRemoteLS(X, y, websocket):
    """
    Remote component of the SIMI algorithm for least squares.
    Sends a typed payload compatible with central FastAPI service.
    """
    stats = await compute_ls_stats(X, y)
    await websocket.send(json.dumps({
        'type': 'data',
        'n': stats['n'],
        'XX': stats['XX'].tolist(),
        'Xy': stats['Xy'].tolist(),
        'yy': stats['yy']
    }))

async def SIRemoteLogit(X, y, websocket):
    """
    Remote component for logistic regression (interactive with central).
    The message shapes align with SIMI central logistic flow.
    """
    p = X.shape[1]
    n = X.shape[0]

    # Send initial sample size so central can accumulate n across sites
    await websocket.send(json.dumps({'type': 'n', 'n': float(n)}))

    while True:
        msg = await websocket.recv()
        data = json.loads(msg)
        if data.get('type') != 'mode':
            break
        mode = data['mode']
        if mode == 0:
            # termination
            break
        beta = np.array(data['beta'])

        xb = np.matmul(X, beta)
        pos_mask = xb > 0
        neg_mask = ~pos_mask
        pr = np.zeros_like(xb)
        pr[pos_mask] = 1 / (1 + np.exp(-xb[pos_mask]))
        pr[neg_mask] = np.exp(xb[neg_mask]) / (1 + np.exp(xb[neg_mask]))
        pr = np.clip(pr, 1e-15, 1 - 1e-15)

        w = pr * (1 - pr)
        H = np.matmul(X.T * w, X)
        g = np.matmul(X.T, y - pr)
        Q = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))

        await websocket.send(json.dumps({'type': 'H', 'H': H.tolist()}))
        await websocket.send(json.dumps({'type': 'g', 'g': g.tolist()}))
        await websocket.send(json.dumps({'type': 'Q', 'Q': Q}))

async def SIMIRemote(D, mvar, site_id, cent_host="127.0.0.1", cent_port=6000):
    """
    Preserved legacy entrypoint. For FastAPI-based remotes, prefer using
    remote/app/websockets.py which calls SIRemoteLS/SIRemoteLogit directly.
    """
    miss = np.isnan(D[:, mvar])
    X = D[~miss, :]
    X = np.delete(X, mvar, axis=1)
    y = D[~miss, mvar]

    if len(X) == 0:
        raise ValueError(f"No valid data for site after filtering missing values at index {mvar}")

    while True:
        try:
            async with websockets.connect(
                f"ws://{cent_host}:{cent_port}",
                ping_interval=None,
                close_timeout=5,
                open_timeout=5
            ) as websocket:
                await websocket.send(json.dumps({
                    'type': 'REMOTE_SITE',
                    'site_id': site_id
                }))
                msg = await websocket.recv()
                data = json.loads(msg)
                method = data.get('method', '').lower()
                if method == 'gaussian':
                    await SIRemoteLS(X, y, websocket)
                elif method == 'logistic':
                    await SIRemoteLogit(X, y, websocket)
        except Exception:
            await asyncio.sleep(5)

async def main():
    # Legacy CLI preserved; see SIMIRemote_base.py for full details
    if len(sys.argv) < 7:
        print("Usage: python SIMIRemote.py <site_id> <cent_host> <cent_port> <missing_var_index> <method> <data_file>")
        return
    site_id = sys.argv[1]
    cent_host = sys.argv[2]
    cent_port = int(sys.argv[3])
    mvar = int(sys.argv[4])
    method = sys.argv[5]
    data_file = sys.argv[6]
    X = pd.read_csv(data_file).values
    await SIMIRemote(X, mvar, site_id, cent_host, cent_port)

if __name__ == "__main__":
    asyncio.run(main())
