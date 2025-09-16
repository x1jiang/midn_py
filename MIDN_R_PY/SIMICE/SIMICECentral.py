"""
Python implementation of SIMICECentral.R

Original R function description:
SIMICECentral = function(D,M,mvar,type,iter,iter0,hosts,ports,cent_ports)

Arguments:
D: Data matrix
M: Number of imputations
mvar: a vector of indices of missing variables
type: a vector of "Gaussian" or "logistic" depending on missing variable types
iter: the number of imputation iterations between extracted imputed data
iter0: the number of imputation iterations before the first extracted imputed data
hosts: a vector of hostnames of remote sites
ports: a vector of ports of remote sites
cent_ports: a vector of local listening ports dedicated to corresponding remote sites
"""

import numpy as np
import asyncio
import time
import os                                  # HIGHLIGHT
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict, Tuple, Optional, Any
import scipy.stats as stats
import scipy.linalg as linalg
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.special import expit
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector, 
    write_string, read_string, write_integer, read_integer,
    WebSocketWrapper, get_wrapped_websocket
)
from Core.LS import ImputeLS, SILSNet
from Core.Logit import ImputeLogit, SILogitNet
# Set to True to enable detailed debug information
print_debug_info = False

# Deterministic RNG: global seed from env or random
# If SIMICE_GLOBAL_SEED is not provided, use a random seed
seed_env = os.getenv("GLOBAL_SEED")
if seed_env is not None:
    GLOBAL_SEED = int(seed_env)
else:
    # Use current time for random seed
    import random
    GLOBAL_SEED = random.randint(0, 2**32-1)
    print(f"No GLOBAL_SEED provided, using random seed: {GLOBAL_SEED}", flush=True)

def _seed_combine(*xs: int) -> int:                             # HIGHLIGHT
    s = 0
    for x in xs:
        s = (s * 1315423911 + int(x)) & 0xFFFFFFFF
    return s

# Track iteration counts for debugging
debug_counters = {
    "ls_calls": 0,
    "logit_calls": 0,
    "impute_ls_calls": 0,
    "impute_logit_calls": 0,
    "iteration": 0,
    "imputation": 0
}

# Dictionary to store WebSocket connections for remote sites
remote_websockets: Dict[str, WebSocket] = {}

# Flag to indicate when imputation is running (to avoid concurrent WebSocket access)
imputation_running = asyncio.Event()

# Dictionary to store locks for each site to ensure sequential communication
site_locks = {}

# No local WebSocket endpoint - we use the WebSockets provided from run_imputation.py

# Helper functions to facilitate network communication for SIMICE
async def initialize_remote_sites(mvar_list):
    """Send initialization message to all remote sites with missing variables"""
    # Check if remote sites are still connected
    connected_sites = list(remote_websockets.keys())
    if not connected_sites:
        raise RuntimeError("No remote sites connected. Cannot initialize.")
    
    print(f"Initializing remote sites: {connected_sites}", flush=True)
    
    # Make a copy of the site list to avoid modification during iteration
    site_items = list(remote_websockets.items())
    
    for site_id, websocket in site_items:
        try:
            async with site_locks[site_id]:
                # Mark WebSocket as pre-accepted since it comes from run_imputation.py
                wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                try:
                    # Send initialization commands
                    await write_string("Initialize", wrapped_ws)
                    await write_vector(mvar_list, wrapped_ws)
                    print(f"Initialized remote site {site_id} with mvar (1-based) {mvar_list}", flush=True)
                except Exception as e:
                    print(f"Failed to initialize {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                    # Remove the site from our active connections
                    if site_id in remote_websockets:
                        del remote_websockets[site_id]
        except Exception as e:
            print(f"Error acquiring lock for {site_id}: {type(e).__name__}: {str(e)}", flush=True)
            
    # Verify we still have required sites after initialization
    if not remote_websockets:
        raise RuntimeError("All remote sites disconnected during initialization.")

# The wrapper functions have been removed and their functionality has been 
# integrated directly into the code, calling the imported functions directly

async def finalize_remote_sites(site_ids):
    """Send end message to all remote sites"""
    for site_id in site_ids:
        if site_id in remote_websockets:
            websocket = remote_websockets[site_id]
            try:
                async with site_locks[site_id]:
                    wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                    
                    # Insert small delay to ensure clean channel
                    await asyncio.sleep(0.1)
                    
                    # Send end message
                    cmd_size = await write_string("End", wrapped_ws)
                    print(f"[CENTRAL] Sent 'End' command to {site_id}: {cmd_size} bytes")
            except Exception as e:
                print(f"[CENTRAL] Error sending End message to {site_id}: {type(e).__name__}: {str(e)}", flush=True)

async def simice_central(D, config=None, site_ids=None, websockets=None, debug=False):
    """
    Implement the SIMICE central algorithm with unified interface
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    config : dict
        Configuration parameters including:
        - M: Number of imputations
        - mvar: List of indices of missing variables (0-based)
        - type_list: List of imputation methods ("Gaussian" or "logistic")
        - iter_val: Number of imputation iterations between extracted imputed data
        - iter0_val: Number of imputation iterations before the first extracted imputed data
    site_ids : List[str]
        List of remote site IDs
    websockets : Dict[str, WebSocket]
        Dictionary of WebSocket connections to remote sites
    debug : bool
        Enable debug mode
    
    Returns:
    --------
    List of imputed datasets
    """
    # Normalize raw config (post-move: central server now passes raw DB params directly)
    if config is None:
        raise ValueError("Config dictionary is required")
    raw = dict(config)
    norm: Dict[str, Any] = {}
    # mvar may come as 'mvar' (already 0-based) OR 'target_column_indexes' (1-based, comma string or list)
    def _parse_index_list(val):
        if val is None:
            return []
        if isinstance(val, str):
            parts = [p.strip() for p in val.replace(';',',').split(',') if p.strip()]
        else:
            parts = list(val)
        out = []
        for p in parts:
            try:
                iv = int(p)
                out.append(iv - 1 if iv > 0 else iv)
            except Exception:
                continue
        return out
    if 'mvar' in raw and not isinstance(raw.get('mvar'), (list, tuple)):
        # Single index
        try:
            iv = int(raw['mvar'])
            norm['mvar'] = [iv] if iv < 0 else [iv]
        except Exception:
            norm['mvar'] = raw['mvar'] if isinstance(raw['mvar'], list) else []
    if 'mvar' in raw and isinstance(raw['mvar'], (list, tuple)):
        norm['mvar'] = [int(i) for i in raw['mvar']]
    if 'target_column_indexes' in raw:
        norm['mvar'] = _parse_index_list(raw['target_column_indexes'])
    # Fallback ensure mvar list
    mvar_list = norm.get('mvar') or []
    # Determine type_list: explicit type_list or derive from is_binary_list
    if 'type_list' in raw and isinstance(raw['type_list'], list):
        tlist = raw['type_list']
    else:
        bin_list = raw.get('is_binary_list')
        if isinstance(bin_list, list):
            tlist = ['logistic' if b else 'Gaussian' for b in bin_list]
        elif isinstance(bin_list, bool) and mvar_list:
            tlist = ['logistic' if bin_list else 'Gaussian'] * len(mvar_list)
        else:
            tlist = []
    if tlist:
        norm_map = {
            'gaussian':'Gaussian','g':'Gaussian','cont':'Gaussian','continuous':'Gaussian',
            'logistic':'logistic','bin':'logistic','binary':'logistic'
        }
        norm['type_list'] = [norm_map.get(str(t).lower(), str(t)) for t in tlist]
    # Iterations
    if 'iter_val' in raw:
        norm['iter_val'] = raw['iter_val']
    elif 'iteration_between_imputations' in raw:
        norm['iter_val'] = raw['iteration_between_imputations']
    if 'iter0_val' in raw:
        norm['iter0_val'] = raw['iter0_val']
    elif 'iteration_before_first_imputation' in raw:
        norm['iter0_val'] = raw['iteration_before_first_imputation']
    # M
    if 'M' in raw:
        norm['M'] = raw['M']
    elif 'imputation_trials' in raw:
        norm['M'] = raw['imputation_trials']
    else:
        norm['M'] = 1
    # Copy any other keys not already consumed
    skip = {'target_column_indexes','is_binary_list','iteration_before_first_imputation','iteration_between_imputations','mvar','type_list','iter_val','iter0_val','imputation_trials','M'}
    for k, v in raw.items():
        if k in skip:
            continue
        norm.setdefault(k, v)
    # Replace config with normalized
    config = norm
    M = config.get('M')
    mvar = config.get('mvar')
    type_list = config.get('type_list')
    iter_val = config.get('iter_val')
    iter0_val = config.get('iter0_val')
    if M is None or not mvar or not type_list:
        raise ValueError(f"Missing required parameters after normalization: M={M} mvar={mvar} type_list={type_list}")
    if iter_val is None:
        iter_val = 100
    if iter0_val is None:
        iter0_val = 100
    global print_debug_info
    if debug is not None:
        print_debug_info = debug
    # Use provided websockets if available
    global remote_websockets
    if websockets is not None:
        remote_websockets = websockets
        print(f"Using provided WebSocket connections for {len(remote_websockets)} sites", flush=True)
        
        # Validate that all expected sites are in the provided websockets
        missing = set(site_ids) - set(remote_websockets.keys())
        if missing:
            raise RuntimeError(f"Not all required remote sites are connected. Missing: {sorted(missing)}")
    else:
        # No WebSockets provided - this should never happen in the new architecture
        # as we expect WebSockets to be provided by run_imputation.py
        raise RuntimeError("No WebSockets provided. SIMICE now requires WebSockets to be passed from run_imputation.py.")
        
        
    # Reset debug counters
    global debug_counters
    debug_counters = {
        "ls_calls": 0,
        "logit_calls": 0,
        "impute_ls_calls": 0,
        "impute_logit_calls": 0,
        "iteration": 0,
        "imputation": 0
    }
    
    # Start timing
    start_time = time.time()
    
    # Ensure we have locks for all sites
    for site_id in site_ids:
        if site_id not in site_locks:
            site_locks[site_id] = asyncio.Lock()
            
    # Set the flag to indicate imputation is running
    imputation_running.set()
    print("Imputation started, flag set", flush=True)
    
    if print_debug_info:
        print(f"[CENTRAL] Debug mode ENABLED - detailed logging will be shown", flush=True)
        print(f"[CENTRAL] Starting SIMICE with M={M}, iter={iter_val}, iter0={iter0_val}", flush=True)
        print(f"[CENTRAL] Data shape: {D.shape}, mvar (0-based) : {mvar}", flush=True)
        print(f"[CENTRAL] Variable types: {type_list}", flush=True)
        print(f"[CENTRAL] Expected remote sites: {site_ids}", flush=True)
        
        has_nan = np.isnan(D).any()
        if has_nan:
            nan_cols = np.isnan(D).any(axis=0)
            print(f"[CENTRAL] Columns with NaN: {np.where(nan_cols)[0].tolist()}", flush=True)
            for col in np.where(nan_cols)[0]:
                nan_count = np.isnan(D[:, col]).sum()
                print(f"[CENTRAL] Column {col}: {nan_count} NaN values ({nan_count/D.shape[0]*100:.1f}%)", flush=True)
    
    try:
        # Build mvar_r if you want to log R-style indices; but for computation use 0-based j
        mvar_r = [idx + 1 for idx in mvar]  # 1-based for protocol (optional logging only)
        
        # Add a column of 1's for the intercept term
        D = np.column_stack([D, np.ones(D.shape[0])]).astype(np.float64, copy=False)  # HIGHLIGHT
        p = D.shape[1]
        n = D.shape[0]
        
        # Store original missing value positions and initialize with mean imputation
        original_missing = {}
        for j_idx, j in enumerate(mvar):
            miss_mask = np.isnan(D[:, j])
            original_missing[j] = miss_mask.copy()  # Store the original missing mask
            non_miss_mask = ~miss_mask
            
            if np.any(miss_mask):
                # Initialize with mean imputation
                D[miss_mask, j] = np.mean(D[non_miss_mask, j])
                
        # Initialize remote sites
        await initialize_remote_sites(mvar_r)
        
        # Initialize the list to store imputed datasets
        imp_list = []
        
        # Main imputation loop
        for m in range(M):
            debug_counters["imputation"] += 1
            imp_start_time = time.time()
            
            # Set the number of iterations
            if m == 0:
                iter_current = iter0_val
            else:
                iter_current = iter_val
            
            if print_debug_info:
                print(f"[CENTRAL][IMPUTATION #{m+1}] Starting imputation {m+1}/{M} with {iter_current} iterations", flush=True)
            
            # Imputation iterations
            for it in range(iter_current):
                debug_counters["iteration"] += 1
                iter_start_time = time.time()
                
                if print_debug_info:
                    print(f"[CENTRAL][ITERATION #{debug_counters['iteration']}] Starting iteration {it+1}/{iter_current}", flush=True)
                
                # Iterate through each missing variable
                for i in range(len(mvar)):
                    j = mvar[i]                                  # 0-based for local math (HIGHLIGHT)
                    j_r = j + 1                                  # 1-based for protocol (HIGHLIGHT)
                    
                    var_start_time = time.time()
                    
                    # Use the original missing value mask instead of checking for NaN
                    miss_mask = original_missing[j]
                    non_miss_mask = ~miss_mask
                    
                    cidx = np.where(non_miss_mask)[0]
                    midx = np.where(miss_mask)[0]
                    nmidx = len(midx)
                    
                    if print_debug_info:
                        print(f"[CENTRAL][VAR j={j}] Processing variable {i+1}/{len(mvar)}, type={type_list[i]}", flush=True)
                        print(f"[CENTRAL][VAR j={j}] Missing: {nmidx}, Non-missing: {len(cidx)}", flush=True)
                    
                    if nmidx > 0:
                        # Imputation for continuous (Gaussian) variables
                        if type_list[i].lower() == "gaussian":
                            # Get sufficient statistics from remote sites directly using SILSNet
                            debug_counters["ls_calls"] += 1
                            fit_imp = await SILSNet(
                                D=D,
                                idx=cidx,
                                j=j,                                 # HIGHLIGHT: 0-based into Core/LS
                                remote_websockets=remote_websockets,
                                site_locks=site_locks,
                                site_ids=site_ids
                            )
                            
                            # Deterministic RNG for this (m, j, it)
                            rng = np.random.default_rng(_seed_combine(GLOBAL_SEED, 0, m, j, it))  # HIGHLIGHT
                            sig = np.sqrt(1.0 / rng.gamma((fit_imp["N"] + 1) / 2.0,
                                                         2.0 / (fit_imp["SSE"] + 1.0)))          # HIGHLIGHT
                            z = rng.normal(size=p - 1)                                            # HIGHLIGHT
                            alpha = fit_imp["beta"] + sig * solve_triangular(fit_imp["cgram"], z, lower=False)
                            
                            # Impute at central
                            X_miss = np.delete(D[midx, :], j, axis=1)                             # HIGHLIGHT
                            D[midx, j] = X_miss @ alpha + rng.normal(0.0, sig, size=nmidx)        # HIGHLIGHT
                            
                            if print_debug_info:
                                print(f"[CENTRAL][VAR j={j}] Gaussian imputation: {nmidx} values", flush=True)
                                imputed_data = D[midx, j]
                                print(f"[CENTRAL][VAR j={j}] Imputed values: min={imputed_data.min():.4f}, max={imputed_data.max():.4f}, mean={imputed_data.mean():.4f}", flush=True)
                            
                            # Push to remotes with deterministic seeds
                            seed_base = _seed_combine(GLOBAL_SEED, 1, m, j, it)                   # HIGHLIGHT
                            debug_counters["impute_ls_calls"] += 1
                            await ImputeLS(
                                j=j_r,                        # protocol expects 1-based at remote
                                beta=alpha,
                                sig=sig,
                                remote_websockets=remote_websockets,
                                site_locks=site_locks,
                                site_ids=site_ids
                            )
                        
                        # Imputation for binary (logistic) variables
                        elif type_list[i].lower() == "logistic":
                            # Get sufficient statistics from remote sites directly using SILogitNet
                            debug_counters["logit_calls"] += 1
                            fit_imp = await SILogitNet(
                                D=D,
                                idx=cidx,
                                j=j,                           # HIGHLIGHT: 0-based into Core/Logit
                                remote_websockets=remote_websockets,
                                site_locks=site_locks,
                                site_ids=site_ids
                            )
                            
                            # Sample from posterior distribution using Bayesian logistic regression
                            # We use the Laplace approximation as in the original code
                            rng = np.random.default_rng(_seed_combine(GLOBAL_SEED, 2, m, j, it))  # HIGHLIGHT
                            try:
                                # Calculate upper triangular Cholesky factor (R uses upper by default)
                                cH = cholesky(fit_imp['H'], lower=False)                              # HIGHLIGHT
                                # Use triangular solve (equivalent to R's backsolve)
                                z = rng.normal(size=p - 1)                                            # HIGHLIGHT
                                alpha = fit_imp['beta'] + solve_triangular(cH, z, lower=False)
                                
                                if print_debug_info:
                                    print(f"[CENTRAL][VAR j={j}] Used Cholesky for sampling coefficients", flush=True)
                            except np.linalg.LinAlgError:
                                # Fallback with regularization if Hessian is not positive definite
                                if print_debug_info:
                                    print(f"[CENTRAL][VAR j={j}] WARNING: Hessian not positive definite, using regularization", flush=True)
                                reg_H = fit_imp['H'] + np.eye(fit_imp['H'].shape[0]) * 1e-4
                                # Calculate upper triangular Cholesky factor
                                cH = cholesky(reg_H, lower=False)
                                z = rng.normal(size=p-1)
                                alpha = fit_imp['beta'] + solve_triangular(cH, z, lower=False)
                            
                            # Generate imputed values
                            X_miss = np.delete(D[midx, :], j, axis=1)                             # HIGHLIGHT
                            pr = expit(X_miss @ alpha)  # 1 / (1 + np.exp(-X_miss @ alpha))
                            D[midx, j] = rng.binomial(1, pr, size=nmidx)                          # HIGHLIGHT
                            
                            if print_debug_info:
                                print(f"[CENTRAL][VAR j={j}] Logistic imputation: {nmidx} values", flush=True)
                                print(f"[CENTRAL][VAR j={j}] Probabilities: min={pr.min():.4f}, max={pr.max():.4f}, mean={pr.mean():.4f}", flush=True)
                                imputed_data = D[midx, j]
                                print(f"[CENTRAL][VAR j={j}] Imputed values sum: {imputed_data.sum()} out of {nmidx} ({imputed_data.sum()/nmidx*100:.1f}%)", flush=True)
                            
                            # Update remote sites directly using ImputeLogit
                            seed_base = _seed_combine(GLOBAL_SEED, 3, m, j, it)                   # HIGHLIGHT
                            debug_counters["impute_logit_calls"] += 1
                            await ImputeLogit(
                                j=j_r,                        # 1-based for remote
                                alpha=alpha,
                                remote_websockets=remote_websockets,
                                site_locks=site_locks,
                                site_ids=site_ids
                            )
                            
                    if print_debug_info:
                        var_time = time.time() - var_start_time
                        print(f"[CENTRAL][VAR j={j}] Variable processed in {var_time:.3f}s", flush=True)
                
                if print_debug_info:
                    iter_time = time.time() - iter_start_time
                    print(f"[CENTRAL][ITERATION #{debug_counters['iteration']}] Iteration {it+1}/{iter_current} completed in {iter_time:.3f}s", flush=True)
            
            # Remove the column of 1's before storing the imputed dataset
            #imputed_dataset = D[:, :-1].copy()
            imputed_dataset = D.copy()  # HIGHLIGHT: keep intercept to match R output
            imp_list.append(imputed_dataset)
            
            if print_debug_info:
                imp_time = time.time() - imp_start_time
                print(f"[CENTRAL][IMPUTATION #{m+1}] Imputation {m+1}/{M} completed in {imp_time:.3f}s", flush=True)
                
                # Print summary statistics for imputed variables
                for i, j in enumerate(mvar):
                    var_data = imputed_dataset[:, j]
                    if type_list[i].lower() == "gaussian":
                        print(f"[CENTRAL][IMPUTATION #{m+1}] Variable j={j} stats: min={var_data.min():.4f}, max={var_data.max():.4f}, mean={var_data.mean():.4f}", flush=True)
                    else:  # logistic
                        print(f"[CENTRAL][IMPUTATION #{m+1}] Variable j={j} counts: 0={np.sum(var_data==0)}, 1={np.sum(var_data==1)}", flush=True)
        
        # Finalize remote sites
        await finalize_remote_sites(site_ids)
        
        if print_debug_info:
            total_time = time.time() - start_time
            print(f"[CENTRAL] SIMICE completed: {M} imputations in {total_time:.3f}s", flush=True)
            print(f"[CENTRAL] Debug counters: {debug_counters}", flush=True)
        
        return imp_list
    
    finally:
        imputation_running.clear()
