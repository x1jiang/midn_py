import numpy as np
from scipy import linalg
import asyncio
from scipy.special import expit
from typing import List, Dict, Any, Optional
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector,
    write_string, write_integer, get_wrapped_websocket
)


async def SILogitNet_async(D: np.ndarray, idx: List[int], j: int, 
                          remote_websockets: Dict, site_locks: Dict,
                          site_ids: List[str], 
                          lam: float = 1e-3, 
                          beta0: Optional[np.ndarray] = None, 
                          maxiter: int = 100,
                          debug_info: bool = True,
                          debug_counter: int = 0) -> Dict[str, Any]:
    """
    Asynchronous Python implementation of the R SILogitNet function with WebSocket communication
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    idx : List[int]
        Indices of samples to use
    j : int
        Index of target variable
    remote_websockets : Dict
        Dictionary mapping site_id to websocket connections
    site_locks : Dict
        Dictionary mapping site_id to asyncio locks
    site_ids : List[str]
        List of remote site IDs to communicate with
    lam : float
        Regularization parameter
    beta0 : np.ndarray, optional
        Initial coefficients, defaults to zeros
    maxiter : int
        Maximum number of iterations
    debug_info : bool
        Whether to print debug information
    debug_counter : int
        Counter for debugging purposes
        
    Returns:
    --------
    Dict with keys:
        beta: coefficient vector
        H: Hessian matrix
        iter: number of iterations performed
    """
    iter_num = debug_counter
    start_time = asyncio.get_event_loop().time()
    p = D.shape[1]
    n = len(idx)
    
    if debug_info:
        print(f"[CENTRAL][LOGIT #{iter_num}] Starting logistic regression for j={j-1}, p={p}, n={n}", flush=True)
        print(f"[CENTRAL][LOGIT #{iter_num}] Connected sites: {site_ids}", flush=True)
    
    # Initialize beta to zeros if not provided
    if beta0 is None:
        beta = np.zeros(p-1)
    else:
        beta = beta0.copy()
    
    # Convergence criteria
    delta = np.ones(p-1)
    eps = 1e-6
    iter_count = 0
    
    if debug_info:
        print(f"[CENTRAL][LOGIT #{iter_num}] Starting Newton-Raphson iterations (max={maxiter}, eps={eps})", flush=True)
    
    # Newton-Raphson iteration
    while linalg.norm(delta) > eps and iter_count < maxiter:
        # Initialize the accumulators for this iteration
        # Accumulators
        H_sum = np.zeros((p-1, p-1))
        g_sum = np.zeros(p-1)
        Q_sum = 0
        N_sum = n  # Start with central sample count
        
        nr_iter_start = asyncio.get_event_loop().time()
        
        # Check if we still have any connected sites left
        active_site_ids = [site_id for site_id in site_ids if site_id in remote_websockets]
        if not active_site_ids and len(site_ids) > 0:
            print(f"[CENTRAL][LOGIT #{iter_num}] No connected sites remaining, stopping iterations", flush=True)
            break
            
        if debug_info:
            print(f"[CENTRAL][LOGIT #{iter_num}] NR Iteration {iter_count+1}, delta norm: {linalg.norm(delta):.6f}", flush=True)
            print(f"[CENTRAL][LOGIT #{iter_num}] Connected sites: {active_site_ids}", flush=True)
        
        # Calculate local contributions first
        X = np.delete(D[idx], j, axis=1)
        y = D[idx, j]
        
        if debug_info:
            print(f"[CENTRAL][LOGIT #{iter_num}] Local data: n={len(idx)}, X shape={X.shape}", flush=True)
            print(f"[CENTRAL][LOGIT #{iter_num}] Local data: y values count: 0={np.sum(y==0)}, 1={np.sum(y==1)}", flush=True)
        
        # Compute probabilities using the current beta
        xb = X @ beta
        pr = expit(xb)  # 1 / (1 + exp(-xb))
        
        # Calculate Hessian and gradient
        W = pr * (1 - pr)
        H_local = X.T @ (X * W[:, np.newaxis])
        g_local = X.T @ (y - pr)
        
        # Calculate local log-likelihood
        idx_low = pr < 0.5
        idx_high = ~idx_low
        Q_local = np.sum(y * xb)
        Q_local += np.sum(np.log(1 - pr[idx_low])) if np.any(idx_low) else 0
        Q_local += np.sum(np.log(pr[idx_high]) - xb[idx_high]) if np.any(idx_high) else 0
        
        # Add local contributions to the accumulators
        H_sum = H_local
        g_sum = g_local
        Q_sum = Q_local
        
        # Send information request to all remote sites
        for site_id in site_ids:
            if site_id in remote_websockets:
                websocket = remote_websockets[site_id]
                site_start_time = asyncio.get_event_loop().time()
                try:
                    async with site_locks[site_id]:
                        wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                        
                        if debug_info:
                            print(f"[CENTRAL][LOGIT #{iter_num}] Requesting logistic info from {site_id}, iter {iter_count+1}", flush=True)
                        
                        # Send instruction, method, and current beta to the remote site
                        # Make sure we have a clean channel by inserting small delay
                        await asyncio.sleep(0.1)
                        
                        # Send each message separately to ensure proper handling
                        cmd_size = await write_string("Information", wrapped_ws)
                        print(f"[CENTRAL][LOGIT #{iter_num}] Sent 'Information' command: {cmd_size} bytes")
                        
                        await asyncio.sleep(0.1)  # Small delay to ensure messages are processed in order
                        
                        method_size = await write_string("logistic", wrapped_ws)
                        print(f"[CENTRAL][LOGIT #{iter_num}] Sent 'logistic' method: {method_size} bytes")
                        
                        await asyncio.sleep(0.1)  # Small delay to ensure messages are processed in order
                        
                        await write_integer(j, wrapped_ws)
                        print(f"[CENTRAL][LOGIT #{iter_num}] Sent j={j}")
                        
                        await asyncio.sleep(0.1)  # Small delay to ensure messages are processed in order
                        
                        await write_integer(1, wrapped_ws)  # mode=1 for Newton-Raphson
                        print(f"[CENTRAL][LOGIT #{iter_num}] Sent mode=1")
                        
                        await asyncio.sleep(0.1)  # Small delay to ensure messages are processed in order
                        
                        await write_vector(beta, wrapped_ws)
                        print(f"[CENTRAL][LOGIT #{iter_num}] Sent beta vector (length={len(beta)})")
                        
                        # Read results from the remote site with timeout to avoid deadlocks
                        n_site = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                        N_sum += int(n_site[0])  # Accumulate remote sample count
                        H_site = await asyncio.wait_for(read_matrix(wrapped_ws), timeout=30.0)
                        g_site = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                        Q_site = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                        
                        # Accumulate results
                        H_sum += H_site
                        g_sum += g_site
                        Q_sum += Q_site[0]
                        
                        if debug_info:
                            site_time = asyncio.get_event_loop().time() - site_start_time
                            print(f"[CENTRAL][LOGIT #{iter_num}] Received from {site_id}: n={n_site[0]}, Q={Q_site[0]:.4f}", flush=True)
                            print(f"[CENTRAL][LOGIT #{iter_num}] Site {site_id} took {site_time:.3f}s", flush=True)
                except Exception as e:
                    print(f"[CENTRAL][LOGIT #{iter_num}] Error communicating with {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                    
                    # Check if this is a timeout error
                    if isinstance(e, asyncio.TimeoutError):
                        print(f"[CENTRAL][LOGIT #{iter_num}] Timeout waiting for response from {site_id}", flush=True)
                    
                    # Detailed error information for debugging
                    if "connection closed" in str(e).lower():
                        print(f"[CENTRAL][LOGIT #{iter_num}] Connection to {site_id} was closed unexpectedly", flush=True)
                    elif "cancelled" in str(e).lower():
                        print(f"[CENTRAL][LOGIT #{iter_num}] Operation with {site_id} was cancelled", flush=True)
                    
                    # If there's a communication error, mark the site as problematic
                    print(f"[CENTRAL][LOGIT #{iter_num}] Communication error, skipping site {site_id}", flush=True)
                    
                    # Remove the problematic site from our active connections
                    if site_id in remote_websockets:
                        print(f"[CENTRAL][LOGIT #{iter_num}] Removing {site_id} from active connections", flush=True)
                        del remote_websockets[site_id]
        
        # Apply L2 regularization
        H_sum += np.eye(H_sum.shape[0]) * (N_sum * lam)
        g_sum -= N_sum * lam * beta
        Q_sum -= 0.5 * N_sum * lam * np.sum(beta**2)
        
        if debug_info:
            print(f"[CENTRAL][LOGIT #{iter_num}] After regularization: Q={Q_sum:.4f}, grad norm={linalg.norm(g_sum):.4f}", flush=True)
        
        # Update beta using Newton-Raphson
        try:
            # Check Hessian condition number to detect potential numerical instability
            svd_values = linalg.svd(H_sum, compute_uv=False)
            condition_number = svd_values[0] / svd_values[-1] if svd_values[-1] > 1e-10 else float('inf')
            
            if condition_number > 1e10:
                print(f"[CENTRAL][LOGIT #{iter_num}] WARNING: Poorly conditioned Hessian (cond={condition_number:.1e}), increasing regularization", flush=True)
                # Additional regularization on top of the standard L2 regularization we already added
                H_sum = H_sum + np.eye(H_sum.shape[0]) * (1e-4 * np.trace(H_sum) / H_sum.shape[0])
            
            # Try Cholesky decomposition for better numerical stability
            try:
                # Use scipy.linalg.cholesky and cho_solve
                chol_H = linalg.cholesky(H_sum, lower=True)
                delta = linalg.cho_solve((chol_H, True), g_sum)
                if debug_info:
                    print(f"[CENTRAL][LOGIT #{iter_num}] Used Cholesky decomposition for solving", flush=True)
            except linalg.LinAlgError:
                # Fall back to direct solver using scipy.linalg
                delta = linalg.solve(H_sum, g_sum)
                if debug_info:
                    print(f"[CENTRAL][LOGIT #{iter_num}] Used direct solver as fallback", flush=True)
            
            # Calculate slope for Armijo condition (Newton direction dot gradient)
            slope = np.dot(delta, g_sum)
            
            if debug_info:
                print(f"[CENTRAL][LOGIT #{iter_num}] Newton direction calculated, slope={slope:.6f}", flush=True)
            
            # Basic sanity check for extreme values in delta before line search
            if np.any(np.abs(delta) > 100):
                print(f"[CENTRAL][LOGIT #{iter_num}] WARNING: Very large update detected, delta max={np.max(np.abs(delta)):.2e}", flush=True)
                # Scale down the update if it's extremely large
                scale_factor = min(1.0, 100 / np.max(np.abs(delta)))
                delta = delta * scale_factor
                slope *= scale_factor
                print(f"[CENTRAL][LOGIT #{iter_num}] Pre-scaling delta by factor {scale_factor:.4f} for numerical stability", flush=True)
            
            # Armijo line search
            step = 1.0
            max_ls_iter = 10  # Maximum line search iterations
            ls_iter = 0
            beta_accepted = False
            
            if debug_info:
                print(f"[CENTRAL][LOGIT #{iter_num}] Starting Armijo line search with initial step={step}", flush=True)
            
            while ls_iter < max_ls_iter and not beta_accepted:
                beta_new = beta + step * delta
                
                # Check basic numerical stability before querying remote sites
                if np.any(np.isnan(beta_new)) or np.any(np.isinf(beta_new)) or np.any(np.abs(beta_new) > 1e10):
                    print(f"[CENTRAL][LOGIT #{iter_num}] Step={step} produces unstable beta, reducing", flush=True)
                    step *= 0.5
                    ls_iter += 1
                    continue
                
                # If the change is very small, accept and break
                if np.max(np.abs(beta_new - beta)) < 1e-5:
                    beta_accepted = True
                    beta = beta_new
                    break
                    
                # Evaluate function at new point
                new_Q_sum = 0
                ls_failed = False
                
                # Calculate local contribution to Q at new beta
                X = np.delete(D[idx], j, axis=1)
                y = D[idx, j]
                xb_new = X @ beta_new
                pr_new = expit(xb_new)
                
                # Calculate local log-likelihood at new point
                idx_low = pr_new < 0.5
                idx_high = ~idx_low
                Q_local_new = np.sum(y * xb_new)
                Q_local_new += np.sum(np.log(1 - pr_new[idx_low])) if np.any(idx_low) else 0
                Q_local_new += np.sum(np.log(pr_new[idx_high]) - xb_new[idx_high]) if np.any(idx_high) else 0
                
                new_Q_sum += Q_local_new
                
                # Request Q at new beta from remote sites using mode=2
                for site_id in [s for s in active_site_ids if s in remote_websockets]:
                    try:
                        async with site_locks[site_id]:
                            websocket = remote_websockets[site_id]
                            wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                            
                            # Send instruction to evaluate at new beta
                            await asyncio.sleep(0.1)
                            await write_string("Information", wrapped_ws)
                            await asyncio.sleep(0.1)
                            await write_string("logistic", wrapped_ws)
                            await asyncio.sleep(0.1)
                            await write_integer(j, wrapped_ws)
                            await asyncio.sleep(0.1)
                            await write_integer(2, wrapped_ws)  # mode=2 for line search
                            await asyncio.sleep(0.1)
                            await write_vector(beta_new, wrapped_ws)
                            
                            # Read remote site's contribution to Q
                            n_site = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                            Q_site = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                            new_Q_sum += Q_site[0]
                    except Exception as e:
                        print(f"[CENTRAL][LOGIT #{iter_num}] Line search communication with {site_id} failed: {str(e)}", flush=True)
                        ls_failed = True
                        # Remove the problematic site
                        if site_id in remote_websockets:
                            del remote_websockets[site_id]
                        break
                
                # If any communication failed, reduce step size and try again
                if ls_failed:
                    step *= 0.5
                    ls_iter += 1
                    continue
                
                # Apply regularization penalty to new beta's Q value
                new_Q_sum -= 0.5 * N_sum * lam * np.sum(beta_new**2)
                
                # Check Armijo condition: new_Q - Q > slope * step / 2
                if new_Q_sum - Q_sum > 0.5 * step * slope:
                    beta_accepted = True
                    beta = beta_new
                    break
                
                step *= 0.5
                ls_iter += 1
            
            # If line search failed to find acceptable step
            if not beta_accepted:
                # Accept a minimal step as a fallback
                beta = beta + 1e-6 * delta
                print(f"[CENTRAL][LOGIT #{iter_num}] Line search failed, taking minimal step", flush=True)
            
            if debug_info:
                print(f"[CENTRAL][LOGIT #{iter_num}] Line search complete: step={step}, iter={ls_iter}, accepted={beta_accepted}", flush=True)
                
        except linalg.LinAlgError as e:
            # If Hessian is not invertible, try a more robust regularization approach
            print(f"[CENTRAL][LOGIT #{iter_num}] WARNING: Hessian not invertible: {str(e)}", flush=True)
            
            # Calculate the average diagonal value to scale regularization appropriately
            avg_diag = np.mean(np.diag(np.abs(H_sum))) if np.any(np.diag(H_sum)) else 1.0
            # Add stronger regularization
            H_sum += np.eye(H_sum.shape[0]) * (avg_diag * 0.01)
            try:
                delta = linalg.solve(H_sum, g_sum)
                # Take a conservative step
                beta = beta + 0.1 * delta
            except:
                # If still failing, take a small gradient step
                beta = beta + 0.01 * g_sum / (linalg.norm(g_sum) + 1e-10)
                
        iter_count += 1
        
        # Check for convergence
        if linalg.norm(delta) < eps:
            if debug_info:
                print(f"[CENTRAL][LOGIT #{iter_num}] Converged at iteration {iter_count}", flush=True)
            break
    
    if debug_info:
        total_time = asyncio.get_event_loop().time() - start_time
        print(f"[CENTRAL][LOGIT #{iter_num}] Completed after {iter_count} iterations in {total_time:.3f}s", flush=True)
        print(f"[CENTRAL][LOGIT #{iter_num}] Final beta: min={np.min(beta):.6f}, max={np.max(beta):.6f}", flush=True)
    
    # Return results as a dictionary
    return {
        'beta': beta,
        'H': H_sum,
        'iter': iter_count
    }
