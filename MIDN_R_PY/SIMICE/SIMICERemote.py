# SIMICERemote.py (revised to match R behavior closely)

import numpy as np
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from scipy.special import expit
import time
from datetime import datetime
import os  # HIGHLIGHT: optional env-based seeding at startup (R-style)
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector, 
    read_string, write_string, read_integer, write_integer,
    read_number,
    WebSocketWrapper, get_wrapped_websocket
)

def create_remote_app(D, port, central_host, central_port, site_id=None):
    """Create a FastAPI app for the remote site
    
    Args:
        D: Data matrix
        port: Port to listen on
        central_host: Host of central server
        central_port: Port of central server
        site_id: Site ID to use (default: None, will use "remote{port}")
    """
    # Add a column of 1's for intercept
    D = np.column_stack([D, np.ones(D.shape[0])]).astype(np.float64, copy=False)
    p = D.shape[1]
    
    # Create FastAPI app
    app = FastAPI()
    
    # Store connection status
    connection_status = {"connected": False, "DD": None, "mvar": [], "site_id": site_id or f"remote{port}", "p": p, "orig_missing_map": {}}

    @app.on_event("startup")
    async def startup_event():
        # HIGHLIGHT: Optional R-style one-time seeding
        seed_env = os.getenv("GLOBAL_SEED")         # HIGHLIGHT
        if seed_env:                                # HIGHLIGHT
            try:                                    # HIGHLIGHT
                np.random.seed(int(seed_env))       # HIGHLIGHT
                print(f"[{connection_status['site_id']}] GLOBAL_SEED set; using seed {int(seed_env)}", flush=True)  # HIGHLIGHT
            except Exception:                       # HIGHLIGHT
                pass                                # HIGHLIGHT
        await connect_to_central(central_host, central_port, port, D, connection_status, site_id)
        
    @app.get("/status")
    async def get_status():
        return connection_status
    
    return app

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

async def connect_to_central(central_host, central_port, local_port, D, connection_status, site_id=None):
    """Connect to the central server and handle communication"""
    actual_site_id = site_id if site_id else f"remote{local_port}"
    connection_status["site_id"] = actual_site_id
    connection_status["p"] = D.shape[1]

    central_url = f"ws://{central_host}:{central_port}/ws/{actual_site_id}"
    print(f"Remote site {actual_site_id} will connect to central at: {central_url}", flush=True)

    asyncio.create_task(maintain_connection(central_url, D, connection_status))

async def maintain_connection(central_url, D, connection_status):
    """Maintain connection to central server with automatic reconnection"""
    max_reconnect_delay = 30
    reconnect_delay = 1
    
    while True:
        try:
            async with websockets.connect(central_url, ping_interval=None) as websocket:
                connection_status["connected"] = True
                print(f"Connected to central server at {central_url}")
                wrapped_ws = get_wrapped_websocket(websocket)
                consecutive_errors = 0
                max_consecutive_errors = 3
                
                while consecutive_errors < max_consecutive_errors:
                    try:
                        #print(f"[{connection_status['site_id']}] Waiting for instruction...")
                        inst = await asyncio.wait_for(read_string(wrapped_ws), timeout=30.0)
                        consecutive_errors = 0
                        #print(f"[{connection_status['site_id']}] Received instruction: {inst}", flush=True)
                        if inst == "Initialize":
                            try:
                                mvar = await read_vector(wrapped_ws)
                                mvar = [int(idx) - 1 for idx in mvar]  # 1-based -> 0-based
                                connection_status["mvar"] = mvar
                                p = connection_status["p"]
                                miss = np.isnan(D)
                                check_cols = [i for i in range(p) if i not in mvar and i != p-1]
                                valid_rows = np.sum(miss[:, check_cols], axis=1) == 0
                                DD = D[valid_rows].copy()
                                miss_DD = np.isnan(DD)
                                # HIGHLIGHT: remember which entries were originally missing in each mvar column
                                orig_missing_map = {}                                                # HIGHLIGHT
                                for j in mvar:
                                    miss_j = miss_DD[:, j]
                                    orig_missing_map[j] = miss_j.copy()                              # HIGHLIGHT
                                    if np.any(miss_j):
                                        DD[miss_j, j] = np.mean(DD[~miss_j, j])

                                connection_status["DD"] = DD
                                connection_status["orig_missing_map"] = orig_missing_map             # HIGHLIGHT

                                site_id = connection_status['site_id']
                                print(f"[{site_id}] Initialized with mvar: {mvar}")
                                for j in mvar:                                                       # HIGHLIGHT
                                    print(f"[{site_id}] original miss count for j={j} -> {orig_missing_map[j].sum()}")  # HIGHLIGHT
                                    
 
                            except Exception as e:
                                site_id = connection_status.get('site_id', 'unknown')
                                print(f"[{site_id}] Error during Initialize: {type(e).__name__}: {e}")
                                raise
                        
                        elif inst == "Information":
                            try:
                                start_time = time.time()
                                method = await read_string(wrapped_ws)
                                j_r = await read_integer(wrapped_ws)
                                #print(f"[{connection_status['site_id']}] get Information and method: {method} for Column (0-based) {j_r} at {datetime.now()}" , flush=True)
                                j = j_r - 1
                                DD = connection_status["DD"]
                                # HIGHLIGHT: use original mask; do NOT treat initialized values as observed
                                orig_missing = connection_status.get("orig_missing_map", {})         # HIGHLIGHT
                                if j in orig_missing:                                                # HIGHLIGHT
                                    valid_idx = ~orig_missing[j]                                     # HIGHLIGHT
                                else:                                                                # HIGHLIGHT
                                    valid_idx = ~np.isnan(DD)[:, j]                                  # HIGHLIGHT
                                X = np.delete(DD[valid_idx], j, axis=1)
                                y = DD[valid_idx, j]
                                if method.lower() == "gaussian":
                                    await si_remote_ls(X, y, wrapped_ws, connection_status['site_id'])
                                elif method.lower() == "logistic":
                                    mode = await read_integer(wrapped_ws)
                                    beta = await read_vector(wrapped_ws)
                                    await write_vector(np.array([X.shape[0]]), wrapped_ws)
                                    await si_remote_logit(X, y, wrapped_ws, beta, connection_status['site_id'], mode)
                                # print(f"[{connection_status['site_id']}] Completed Information for j={j} in {time.time() - start_time:.4f} seconds", flush=True)
                            except Exception as e:
                                print(f"Error during Information: {type(e).__name__}: {e}")
                                raise
                        
                        elif inst == "Impute":
                            try:
                                method = await read_string(wrapped_ws)
                                print(f"[{connection_status['site_id']}] get Impute and method: {method}", flush=True)
                                j_r = await read_integer(wrapped_ws)
                                j = j_r - 1

                                site_id = connection_status['site_id']
                                DD = connection_status["DD"]
                                # HIGHLIGHT: impute only the entries that were originally missing
                                orig_missing = connection_status.get("orig_missing_map", {})         # HIGHLIGHT
                                if j in orig_missing:                                                # HIGHLIGHT
                                    midx = np.where(orig_missing[j])[0]                              # HIGHLIGHT
                                else:                                                                # HIGHLIGHT
                                    midx = np.where(np.isnan(DD)[:, j])[0]                           # HIGHLIGHT
                                nmidx = len(midx)

                                # HIGHLIGHT: Always read payload first (R parity)
                                if method.lower() == "gaussian":                         # HIGHLIGHT
                                    beta = await read_vector(wrapped_ws)                 # HIGHLIGHT
                                    sigv = await read_vector(wrapped_ws)                 # HIGHLIGHT
                                    sig = float(sigv[0])                                 # HIGHLIGHT
                                elif method.lower() == "logistic":                       # HIGHLIGHT
                                    alpha = await read_vector(wrapped_ws)                # HIGHLIGHT
                                else:                                                    # HIGHLIGHT
                                    print(f"[{site_id}] Unknown impute method: {method}", flush=True)  # HIGHLIGHT
                                    continue                                             # HIGHLIGHT

                                if nmidx == 0:
                                    # HIGHLIGHT: Keep stream aligned; discard payload cleanly
                                    print(f"[{site_id}][IMPUTE] No original missing for j={j}; payload consumed and discarded")  # HIGHLIGHT
                                    continue                                             # HIGHLIGHT

                                # Now perform imputation (global RNG; no per-call reseed)
                                X = np.delete(DD[midx, :], j, axis=1)
                                if method.lower() == "gaussian":
                                    DD[midx, j] = X @ beta + np.random.normal(0.0, sig, size=nmidx)    # HIGHLIGHT
                                    #print(f"[{site_id}][IMPUTE][Gaussian] miss={nmidx}, "
                                    #    f"beta[min,max]=[{beta.min():.4f},{beta.max():.4f}], sigma={sig:.4f}", flush=True)
                                else:  # logistic
                                    pr = 1.0 / (1.0 + np.exp(-(X @ alpha)))                            # HIGHLIGHT
                                    DD[midx, j] = np.random.binomial(1, pr)                            # HIGHLIGHT
                                    ones = int(np.sum(DD[midx, j] == 1))
                                    zeros = nmidx - ones
                                    #print(f"[{site_id}][IMPUTE][Logistic] miss={nmidx}, "
                                    #    f"alpha[min,max]=[{alpha.min():.4f},{alpha.max():.4f}], "
                                    #    f"counts: 0={zeros}, 1={ones}", flush=True)
                                connection_status["DD"] = DD
        

                            except Exception as e:
                                site_id = connection_status.get('site_id', 'unknown')
                                print(f"[{site_id}] Error during Impute: {type(e).__name__}: {e}", flush=True)
                        
                        elif inst == "End":
                            print("Received End command from central server")
                            return
                        
                        elif inst == "ping":
                            await write_string("pong", wrapped_ws)
                        else:
                            print(f"Unknown instruction: {inst}")
                    
                    except asyncio.TimeoutError:
                        try:
                            await write_string("ping", wrapped_ws)
                        except Exception as e:
                            print(f"[{connection_status['site_id']}] Connection seems dead (error sending ping): {type(e).__name__}: {str(e)}", flush=True)
                            break
                    except ValueError as e:
                        if "too short" in str(e) or "marker" in str(e) or "string" in str(e):
                            print(f"[{connection_status['site_id']}] PROTOCOL ERROR: {e}")
                            print(f"[{connection_status['site_id']}] Attempting to recover by skipping this message")
                            consecutive_errors += 1
                            print(f"[{connection_status['site_id']}] Error count: {consecutive_errors}/{max_consecutive_errors}")
                            try:
                                resync_data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                                if isinstance(resync_data, str) and resync_data in ["Information", "Initialize", "Impute", "End", "ping", "pong"]:
                                    print(f"[{connection_status['site_id']}] Recovered valid command: {resync_data}")
                                    inst = resync_data
                                    consecutive_errors = 0
                                    continue
                                else:
                                    print(f"[{connection_status['site_id']}] Read additional data to resync")
                                await asyncio.sleep(0.5)
                            except Exception as recovery_e:
                                print(f"[{connection_status['site_id']}] Recovery failed: {recovery_e}")
                                consecutive_errors += 1
                        else:
                            print(f"[{connection_status['site_id']}] Error processing message: {type(e).__name__}: {e}")
                            consecutive_errors += 1
                    except asyncio.TimeoutError:
                        print(f"[{connection_status['site_id']}] Timeout waiting for message")
                        consecutive_errors += 1
                        await asyncio.sleep(1)
                    except Exception as e:
                        print(f"[{connection_status['site_id']}] Error processing message: {type(e).__name__}: {e}")
                        consecutive_errors += 1
                        if isinstance(e, (websockets.exceptions.ConnectionClosedError, 
                                          ConnectionResetError, 
                                          websockets.exceptions.ConnectionClosedOK)):
                            print(f"[{connection_status['site_id']}] Connection error, will reconnect")
                            break
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[{connection_status['site_id']}] Too many consecutive errors ({consecutive_errors}), reconnecting")
        except websockets.exceptions.InvalidStatusCode as e:
            connection_status["connected"] = False
            print(f"Invalid status code connecting to {central_url}: {e}")
        except ConnectionRefusedError as e:
            connection_status["connected"] = False
            print(f"Connection refused to {central_url}: {e}")
        except Exception as e:
            connection_status["connected"] = False
            print(f"Error connecting to {central_url}: {type(e).__name__}: {e}")
        print("Waiting to reconnect...")
        await asyncio.sleep(5)

def run_remote_client(data_file, central_host, central_port, site_id=None, remote_port=None, config=None):
    """Run the remote client
    
    Args:
        data_file: Data matrix as numpy array or path to data file
        central_host: Host of central server
        central_port: Port of central server
        site_id: Site ID to use (default: None, will use "remote{remote_port}")
        remote_port: Port for the remote FastAPI server
        config: Configuration dictionary (not used in SIMICE but provided for interface consistency)
    """
    if isinstance(data_file, str):
        import pandas as pd
        D = pd.read_csv(data_file).values
    else:
        D = data_file
    
    # Use default port if not provided
    if remote_port is None:
        if site_id and site_id.endswith('1'):
            remote_port = 8001
        elif site_id and site_id.endswith('2'):
            remote_port = 8002
        else:
            remote_port = 8010
            
    actual_site_id = site_id if site_id else f"remote{remote_port}"
    app = create_remote_app(D, remote_port, central_host, central_port, actual_site_id)
    uvicorn.run(app, host="0.0.0.0", port=remote_port)

 