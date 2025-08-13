#!/usr/bin/env python3

"""
SIMICentral.py - SIMI Central site implementation
- Adds helpers usable by FastAPI service for orchestration
"""

import sys
import os
import asyncio
import numpy as np
import websockets
import json
import time
from scipy import stats
import pandas as pd

async def SICentralLS(X, y, expected_sites, expected_site_names=None, port=6000, lam=1e-3):
    """
    Central component of the SIMI algorithm for least squares.
    """
    print("Starting SICentralLS function")
    p = X.shape[1]
    n = X.shape[0]
    
    XX = np.matmul(X.T, X)
    Xy = np.matmul(X.T, y)
    yy = np.sum(y**2)
    
    # Shared state for WebSocket connections
    connected_sites = {}
    remote_sites = 0
    connection_event = asyncio.Event()
    
    # WebSocket handler for incoming connections - websockets v10+ only provides one parameter
    async def handler(websocket):
        nonlocal connected_sites, remote_sites, n, XX, Xy, yy
        
        try:
            # Receive site identification message
            msg = await websocket.recv()
            data = json.loads(msg)
            
            if data['type'] == 'REMOTE_SITE':
                site_id = data['site_id']
                
                # Validate site_id if expected_site_names is provided
                site_valid = True
                if expected_site_names is not None and site_id not in expected_site_names:
                    print(f"WARNING: Unexpected remote site connected with ID: {site_id}")
                    site_valid = False
                else:
                    print(f"Remote site connected with ID: {site_id}")
                
                # Only process if we haven't seen this site before and it's valid
                if site_valid and site_id not in connected_sites:
                    connected_sites[site_id] = True
                    remote_sites += 1
                    
                    # Send instruction to the remote
                    print(f"Sending Gaussian method to site {site_id}")
                    await websocket.send(json.dumps({
                        'type': 'method',
                        'method': 'gaussian'
                    }))
                    
                    # Receive data from the remote site
                    try:
                        # Receive n, XX, Xy, and yy from the remote site
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        n_remote = data['n']
                        XX_remote = np.array(data['XX'])
                        Xy_remote = np.array(data['Xy'])
                        yy_remote = data['yy']
                        
                        n += n_remote
                        XX += XX_remote
                        Xy += Xy_remote
                        yy += yy_remote
                        
                        print(f"Successfully received data from site {site_id}")
                        
                        # Check if we have all expected sites
                        if remote_sites >= expected_sites:
                            if expected_site_names is None or all(site in connected_sites for site in expected_site_names):
                                print("All expected remote sites have connected.")
                                connection_event.set()
                        
                    except Exception as e:
                        print(f"ERROR reading from site {site_id}: {str(e)}")
                        # Remove the site from the connected list if there was an error
                        del connected_sites[site_id]
                        remote_sites -= 1
                        
        except Exception as e:
            print(f"Error in WebSocket handler: {str(e)}")
    
    # Start WebSocket server
    print(f"Creating WebSocket server on 0.0.0.0:{port}")
    server = await websockets.serve(
        handler, 
        '0.0.0.0', 
        port,
        ping_interval=50,  # Send ping every 50 seconds
        ping_timeout=30,   # Wait 30 seconds for pong response
        close_timeout=10   # Wait 10 seconds for close handshake
    )
    
    # If specific site names are provided, report them
    if expected_site_names is not None:
        print(f"Expected remote sites: {', '.join(expected_site_names)}")
    
    print("Waiting up to 60 seconds for all remote sites to connect...")
    try:
        # Wait for either all sites to connect or timeout
        await asyncio.wait_for(connection_event.wait(), timeout=60)
    except asyncio.TimeoutError:
        print("Timed out waiting for remote sites")
        server.close()
        await server.wait_closed()
        raise RuntimeError("Timeout waiting for remote sites")
    
    # All sites connected, compute the result
    beta = np.linalg.solve(XX + lam * np.eye(p), Xy)
    
    # Close the server
    server.close()
    await server.wait_closed()
    
    # Return results
    SI = {
        "beta": beta,
        "vcov": np.linalg.inv(XX),
        "n": n,
        "SSE": yy - 2 * np.sum(Xy * beta) + np.sum(XX * np.outer(beta, beta))
    }
    return SI

async def SICentralLogit(X, y, expected_sites, expected_site_names=None, port=6000, lam=1e-3, maxiter=25):
    """
    Central component of the SIMI algorithm for logistic regression.
    """
    print("Starting SICentralLogit function")
    p = X.shape[1]
    n = X.shape[0]
    
    connected_sites = {}
    remote_sites = 0
    connection_event = asyncio.Event()
    site_responses = {}
    all_done_event = asyncio.Event()  # Event to keep handlers alive

    async def handler(websocket):
        nonlocal connected_sites, remote_sites, site_responses, n
        site_id = None
        try:
            msg = await websocket.recv()
            data = json.loads(msg)
            
            if data['type'] == 'REMOTE_SITE':
                site_id = data['site_id']
                
                if expected_site_names and site_id not in expected_site_names:
                    print(f"WARNING: Unexpected remote site connected: {site_id}")
                    return
                
                if site_id not in connected_sites:
                    print(f"Remote site connected: {site_id}")
                    connected_sites[site_id] = True
                    remote_sites += 1
                    
                    await websocket.send(json.dumps({'type': 'method', 'method': 'logistic'}))
                    
                    site_responses[site_id] = {'websocket': websocket}
                    
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    n_remote = data['n']
                    site_responses[site_id]['n'] = n_remote
                    n += n_remote
                    print(f"Received n={n_remote} from {site_id}")
                    
                    if remote_sites >= expected_sites:
                        print("All expected remote sites have connected.")
                        connection_event.set()

                    # Wait for the main process to signal completion
                    await all_done_event.wait()
        except Exception as e:
            print(f"Error in WebSocket handler for site {site_id or 'unknown'}: {str(e)}")
        finally:
            if site_id and site_id in connected_sites:
                del connected_sites[site_id]
                remote_sites -= 1
                print(f"Connection with {site_id} closed.")

    server = await websockets.serve(handler, '0.0.0.0', port)
    print(f"Waiting for {expected_sites} remote sites to connect...")
    
    try:
        await asyncio.wait_for(connection_event.wait(), timeout=60)
    except asyncio.TimeoutError:
        print("Timed out waiting for remote sites.")
        raise RuntimeError("Timeout waiting for remote sites")

    beta = np.zeros(p)
    Q_old = -np.inf
    
    try:
        for mode in range(1, maxiter + 1):
            print(f"Iteration {mode}")
            
            send_tasks = [
                site['websocket'].send(json.dumps({
                    'type': 'mode',
                    'mode': mode,
                    'beta': beta.tolist()
                }))
                for site in site_responses.values()
            ]
            await asyncio.gather(*send_tasks)
                
            H_total = np.zeros((p, p))
            g_total = np.zeros(p)
            Q_total = 0
            
            async def receive_from_site(site):
                H_str = await site['websocket'].recv()
                g_str = await site['websocket'].recv()
                Q_str = await site['websocket'].recv()
                H = np.array(json.loads(H_str)['H'])
                g = np.array(json.loads(g_str)['g'])
                Q = json.loads(Q_str)['Q']
                return H, g, Q

            receive_tasks = [receive_from_site(site) for site in site_responses.values()]
            results = await asyncio.gather(*receive_tasks)

            for H, g, Q in results:
                H_total += H
                g_total += g
                Q_total += Q
                
            print(f"Iteration {mode} - Q: {Q_total}")
            
            # Add regularization to Hessian for stability
            H_total += lam * np.eye(p)
            
            try:
                # Use more stable solver with regularization
                delta_beta = np.linalg.solve(H_total, g_total)
                
                # Limit step size to avoid numerical issues
                step_norm = np.linalg.norm(delta_beta)
                if step_norm > 1.0:
                    delta_beta *= (1.0 / step_norm)
                
                beta += delta_beta
                
                # Better convergence criterion based on parameter change
                if mode > 1 and np.linalg.norm(delta_beta) < 1e-4:
                    print(f"Converged after {mode} iterations")
                    break
                
                # Ensure we don't diverge
                if mode > 5 and Q_total < Q_old:
                    print("Warning: Log-likelihood decreased. Using smaller step.")
                    beta -= 0.5 * delta_beta  # Take a half-step back
            except np.linalg.LinAlgError:
                print("Warning: Linear algebra error. Using gradient descent instead.")
                # Fall back to simple gradient descent with small step size
                beta += 0.01 * g_total
            
            Q_old = Q_total
            
    finally:
        print("Cleaning up and closing connections...")
        termination_tasks = [
            site['websocket'].send(json.dumps({'type': 'mode', 'mode': 0}))
            for site in site_responses.values()
        ]
        await asyncio.gather(*termination_tasks, return_exceptions=True)
        
        all_done_event.set()
        
        server.close()
        await server.wait_closed()
    
    return {"beta": beta, "vcov": np.linalg.inv(H_total), "n": n}

async def SIMICentral(D, M, mvar, method, expected_sites, expected_site_names=None, port=6000):
    """
    Main function for SIMI central site.
    """
    n, p = D.shape
    miss = np.isnan(D[:, mvar])
    nm = np.sum(miss)
    nc = n - nm
    
    X = D[~miss, :]
    X = np.delete(X, mvar, axis=1)
    y = D[~miss, mvar]
    
    if method.lower() == "gaussian":
        SI = await SICentralLS(X, y, expected_sites, expected_site_names, port)
        cvcov = np.linalg.cholesky(SI["vcov"])
    elif method.lower() == "logistic":
        SI = await SICentralLogit(X, y, expected_sites, expected_site_names, port)
        cvcov = np.linalg.cholesky(SI["vcov"])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    imp = []
    for m in range(M):
        # Create a copy of D for imputation
        D_imp = D.copy()
        
        if method.lower() == "gaussian":
            # In R: sig = sqrt(1/rgamma(1,(SI$n+1)/2,(SI$SSE+1)/2))
            # In Python using scipy.stats:
            shape = (SI["n"] + 1) / 2
            scale = 2 / (SI["SSE"] + 1)  # scale parameter in scipy is 1/rate
            sig = 1 / np.sqrt(stats.gamma.rvs(shape, scale=scale))
            
            # Generate random normal variates for coefficients
            alpha = SI["beta"] + sig * np.matmul(cvcov.T, np.random.normal(0, 1, p-1))
            
            # Generate imputed values
            X_miss = np.delete(D[miss, :], mvar, axis=1)
            D_imp[miss, mvar] = np.matmul(X_miss, alpha) + np.random.normal(0, sig, nm)
            
        elif method.lower() == "logistic":
            # Generate random normal variates for coefficients
            alpha = SI["beta"] + np.matmul(cvcov.T, np.random.normal(0, 1, p-1))
            
            # Generate imputed values
            X_miss = np.delete(D[miss, :], mvar, axis=1)
            
            # More numerically stable calculation of logistic function
            xb = np.matmul(X_miss, alpha)
            # Use a safer approach for the logistic calculation
            pr = np.zeros_like(xb)
            # For large positive values, prob is nearly 1
            pos_mask = xb > 0
            # For large negative values, use exp(x) instead of exp(-x)
            neg_mask = ~pos_mask
            
            # For positive values: 1 / (1 + exp(-x))
            pr[pos_mask] = 1 / (1 + np.exp(-xb[pos_mask]))
            # For negative values: exp(x) / (1 + exp(x))
            pr[neg_mask] = np.exp(xb[neg_mask]) / (1 + np.exp(xb[neg_mask]))
            
            # Ensure probabilities are between 0 and 1
            pr = np.clip(pr, 0, 1)
            D_imp[miss, mvar] = np.random.binomial(1, pr)
        
        imp.append(D_imp)
    
    return imp

def aggregate_ls_stats(local_stats, remote_stats_list):
    """
    Sum sufficient statistics across sites for Gaussian LS.
    local_stats: dict with keys n, XX, Xy, yy (numpy arrays/scalars allowed)
    remote_stats_list: list of dicts in the same shape
    Returns (n, XX, Xy, yy)
    """
    n = float(local_stats['n'])
    XX = np.array(local_stats['XX'])
    Xy = np.array(local_stats['Xy'])
    yy = float(local_stats['yy'])
    for s in remote_stats_list:
        n += float(s['n'])
        XX += np.array(s['XX'])
        Xy += np.array(s['Xy'])
        yy += float(s['yy'])
    return n, XX, Xy, yy

def gaussian_impute(central_df: pd.DataFrame, mvar: int, n: float, XX: np.ndarray, Xy: np.ndarray, yy: float, M: int = 10, lam: float = 1e-3) -> pd.DataFrame:
    """
    Perform Gaussian imputation at central given aggregated stats.
    Returns a single imputed DataFrame (first draw) to match current service behavior.
    """
    p = central_df.shape[1]
    miss = np.isnan(central_df.iloc[:, mvar].values)
    nm = int(np.sum(miss))

    # local design for fit
    X = central_df.loc[~miss, :].drop(central_df.columns[mvar], axis=1).values
    p_minus_1 = X.shape[1]

    beta = np.linalg.solve(XX + lam * np.eye(p_minus_1), Xy)
    vcov = np.linalg.inv(XX)
    SSE = yy - 2 * np.sum(Xy * beta) + np.sum(XX * np.outer(beta, beta))

    cvcov = np.linalg.cholesky(vcov)
    imp = []
    for m in range(M):
        D_imp = central_df.values.copy()
        shape = (n + 1) / 2
        scale = 2 / (SSE + 1)
        sig = 1 / np.sqrt(stats.gamma.rvs(shape, scale=scale))
        alpha = beta + sig * np.matmul(cvcov.T, np.random.normal(0, 1, p_minus_1))
        X_miss = np.delete(D_imp[miss, :], mvar, axis=1)
        D_imp[miss, mvar] = np.matmul(X_miss, alpha) + np.random.normal(0, sig, nm)
        imp.append(D_imp)
    return pd.DataFrame(imp[0], columns=central_df.columns)

async def main():
    # Parse command line arguments
    if len(sys.argv) < 5:
        print("Usage: python SIMICentral.py <method> <missing_var_index> <data_file> <truth_file>")
        print("  method: Gaussian or logistic")
        print("  missing_var_index: 0-based index of the variable with missing values")
        print("  data_file: Path to the CSV data file with missing values")
        print("  truth_file: Path to the CSV truth file with complete data")
        return
    
    method = sys.argv[1]
    mvar = int(sys.argv[2])
    data_file = sys.argv[3]
    truth_file = sys.argv[4]
    
    # Configuration
    expected_sites = 2  # Number of remote sites to wait for
    expected_site_names = ["remote_site_1", "remote_site_2"]  # Names of expected remote sites
    port = 6000  # Port to listen on
    
    # Define base path for data files
    base_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'R_ALL_MODLES', 'SIMI', 'test_data')
    
    print(f"=== SIMI Central Site - {method.capitalize()} Method ===")
    print(f"Listening on port: {port}")
    print(f"Expecting {expected_sites} remote sites: {', '.join(expected_site_names)}")
    print(f"Missing variable index: {mvar}")
    
    # Load data from the provided files
    print(f"Loading data from: {data_file}")
    print(f"Loading truth data from: {truth_file}")
    
    X = pd.read_csv(data_file).values
    X_truth = pd.read_csv(truth_file).values
    
    # Verify that data and truth have compatible shapes
    if X.shape != X_truth.shape:
        print(f"Warning: Data shape ({X.shape}) doesn't match truth shape ({X_truth.shape})")
    
    # Find all missing values in the target column
    missing_mask = np.isnan(X[:, mvar])
    missing_count = np.sum(missing_mask)
    
    # Verify that truth data doesn't have missing values in the target column
    if np.any(np.isnan(X_truth[:, mvar])):
        print("Warning: Truth data contains missing values in the target column")
    
    # For binary data, make sure truth values are properly formatted
    if method.lower() == "logistic":
        # Check if truth data is binary (0/1)
        unique_values = np.unique(X_truth[~np.isnan(X_truth[:, mvar]), mvar])
        if not all(val in [0, 1] for val in unique_values):
            print(f"Warning: Truth data for logistic method contains non-binary values: {unique_values}")
            # Attempt to convert to binary if close to 0/1
            if all(val >= -0.1 and val <= 1.1 for val in unique_values):
                X_truth[:, mvar] = np.round(X_truth[:, mvar])
    
    print(f"Found {missing_count} missing values in column {mvar}")
    print(f"Loaded data with missing values (shape: {X.shape})...")
    
    if method.lower() not in ["gaussian", "logistic"]:
        print(f"Unknown method: {method}. Use 'Gaussian' or 'logistic'.")
        return
    
    print("Waiting for remote sites to connect...")
    
    # Run imputation with M=10 attempts
    M = 100  # Number of imputations to perform
    imp = await SIMICentral(
        D=X,
        M=M,
        mvar=mvar,
        method=method,
        expected_sites=expected_sites,
        expected_site_names=expected_site_names,
        port=port
    )
    
    # Create a mask for all missing values
    all_missing_mask = missing_mask
    
    # Extract truth values for all missing entries
    truth_values = X_truth[all_missing_mask, mvar]
    
    # Extract imputed values for all missing entries across all M imputations
    all_imputed_values = np.array([imp_set[all_missing_mask, mvar] for imp_set in imp])
    
    # Calculate consolidated imputation (average or majority vote)
    consolidated_values = None
    if method.lower() == "gaussian":
        # For continuous data, use average across imputations
        consolidated_values = np.mean(all_imputed_values, axis=0)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((consolidated_values - truth_values) ** 2))
        print(f"\nRMSE (consolidated imputations): {rmse:.4f}")
        
        # Calculate RMSE for each imputation and average
        individual_rmses = [np.sqrt(np.mean((imp_vals - truth_values) ** 2)) for imp_vals in all_imputed_values]
        avg_individual_rmse = np.mean(individual_rmses)
        print(f"Average RMSE across {len(imp)} individual imputations: {avg_individual_rmse:.4f}")
        
        performance_metric = rmse
        metric_name = "RMSE"
        
    elif method.lower() == "logistic":
        # For binary data, use majority vote (mode) across imputations
        print("Using manual mode calculation for binary data")
        # Calculate mode manually by taking the mean and rounding
        consolidated_values = np.round(np.mean(all_imputed_values, axis=0))
        consolidated_values = consolidated_values.astype(int)
        
        # Ensure truth values are binary 0/1 integers for comparison
        truth_values = truth_values.astype(int)
        
        # Safety check - handle any non-binary values if they still exist
        if np.any((truth_values != 0) & (truth_values != 1)):
            print("Warning: Found non-binary values in truth data. Converting to binary 0/1.")
            truth_values = (truth_values > 0).astype(int)
        
        # Print debug information
        print(f"\nDebug - Truth values summary:")
        print(f"  Range: [{np.min(truth_values)}, {np.max(truth_values)}]")
        print(f"  Mean: {np.mean(truth_values)}")
        print(f"  Unique values: {np.unique(truth_values)}")
        
        print(f"\nDebug - Imputed values summary:")
        print(f"  Range: [{np.min(consolidated_values)}, {np.max(consolidated_values)}]") 
        print(f"  Mean: {np.mean(consolidated_values)}")
        print(f"  Unique values: {np.unique(consolidated_values)}")
        
        # Calculate accuracy with proper type comparison
        matches = np.abs(consolidated_values - truth_values) < 0.01  # Allow small numerical differences
        accuracy = np.mean(matches)
        print(f"\nAccuracy (consolidated imputations): {accuracy:.4f}")
        
        # Calculate accuracy for each imputation and average
        individual_accuracies = []
        for imp_vals in all_imputed_values:
            imp_matches = np.abs(imp_vals.astype(float) - truth_values) < 0.01
            individual_accuracies.append(np.mean(imp_matches))
        
        avg_individual_accuracy = np.mean(individual_accuracies)
        print(f"Average accuracy across {len(imp)} individual imputations: {avg_individual_accuracy:.4f}")
        
        performance_metric = accuracy
        metric_name = "Accuracy"
    
    # Save results to JSON file
    result_dir = os.path.join(base_data_path, 'result')
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f'imp_{method.lower()}_PY.json')
    
    # Prepare results dictionary
    results = {
        "method": method,
        "missing_var_index": mvar,
        "missing_count": int(missing_count),
        "imputations": M,
        "consolidated_values": consolidated_values.tolist() if hasattr(consolidated_values, 'tolist') else consolidated_values,
        "truth_values": truth_values.tolist() if hasattr(truth_values, 'tolist') else truth_values,
        "performance_metric": float(performance_metric),
        "metric_name": metric_name,
        "individual_imputations": [imp_vals.tolist() for imp_vals in all_imputed_values]
    }
    
    # Create a custom JSON encoder to handle NaN values properly
    class NaNJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, float) and np.isnan(obj):
                return "NaN"
            return super().default(obj)
    
    # Write the results to a JSON file
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NaNJSONEncoder)
    
    print(f"\nImputation results saved to: {result_file}")
    
    # Also save a simpler CSV with just the truth and imputed values for easier analysis
    csv_file = os.path.join(result_dir, f'imp_{method.lower()}_PY_comparison.csv')
    comparison_df = pd.DataFrame({
        'truth': truth_values,
        'imputed': consolidated_values
    })
    comparison_df.to_csv(csv_file, index=False)
    print(f"Comparison CSV saved to: {csv_file}")
    
    print(f"\nPerformance summary:")
    print(f"{metric_name}: {performance_metric:.4f}")
    print(f"Missing values analyzed: {missing_count}")
    
    print("\n=== SIMI Central Site Completed ===")

if __name__ == "__main__":
    asyncio.run(main())
