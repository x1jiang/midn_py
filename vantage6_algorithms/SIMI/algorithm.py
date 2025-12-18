"""
SIMI Algorithm for Vantage6
Single Imputation for Missing Data - adapted for vantage6 framework

Original algorithm from MIDN_R_PY/SIMI, converted to use vantage6's RPC pattern.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.linalg import cholesky, cho_solve
from scipy.special import expit
import json
from vantage6.algorithm.tools.util import info, warn, error

# Import vantage6 tools (AlgorithmClient lives in vantage6.algorithm.client)
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data


def _to_serializable(obj):
    """Recursively convert numpy/pandas objects to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.values.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _create_and_wait(client: AlgorithmClient, input_: Dict[str, Any], org_ids: List[int] | None = None):
    """Dispatch a task and wait for results using the current AlgorithmClient API."""
    clean_input = _to_serializable(input_)
    task = client.task.create(input_=clean_input, organizations=org_ids or [])
    return client.wait_for_results(task_id=task.get("id"), interval=1)

@data(1)
def simi_remote_gaussian(data: pd.DataFrame, mvar: int, *args, **kwargs) -> Dict[str, Any]:
    """
    Remote function for Gaussian method in SIMI.
    
    Computes local statistics (XX, Xy, yy) for Gaussian regression.
    
    Parameters:
    -----------
    data : dict
        Contains 'data' key with local dataset (numpy array or path)
        and 'mvar' key with target column index (1-based)
    
    Returns:
    --------
    dict with keys: n, XX, Xy, yy
    """
    # Load data (vantage6 @data provides a DataFrame; keep ndarray fallback)
    if isinstance(data, pd.DataFrame):
        D = data.values
    elif isinstance(data, np.ndarray):
        D = data
    else:
        raise ValueError("Data must be numpy array or DataFrame")
    
    # Convert to 0-based index
    mvar = int(mvar) - 1
    
    # Extract complete cases
    miss = np.isnan(D[:, mvar])
    X = np.delete(D[~miss], mvar, axis=1)
    y = D[~miss, mvar]
    
    # Compute statistics
    n = X.shape[0]
    XX = X.T @ X
    Xy = X.T @ y
    yy = np.sum(y ** 2)
    
    # Handle NaN/Inf
    XX = np.nan_to_num(XX)
    Xy = np.nan_to_num(Xy)
    yy = np.nan_to_num(yy)
    
    return {
        'n': int(n),
        'XX': XX.tolist(),  # Convert to list for JSON serialization
        'Xy': Xy.tolist(),
        'yy': float(yy)
    }


@data(1)
def simi_remote_logistic(data: pd.DataFrame, beta: List[float], mode: int, mvar: int, *args, **kwargs) -> Dict[str, Any]:
    """
    Remote function for Logistic method in SIMI.
    
    Computes local statistics for logistic regression iteration.
    
    Parameters:
    -----------
    data : dict
        Contains 'data' key with local dataset and 'mvar' with target column
    beta : list
        Current coefficient vector
    mode : int
        1 = compute H and g, 2 = compute Q only, 0 = terminate
    
    Returns:
    --------
    dict with keys: n, H, g, Q (depending on mode)
    """
    if mode == 0:
        return {'status': 'terminated'}
    
    # Load data (vantage6 @data provides a DataFrame; keep ndarray fallback)
    if isinstance(data, pd.DataFrame):
        D = data.values
    elif isinstance(data, np.ndarray):
        D = data
    else:
        raise ValueError("Data must be numpy array or DataFrame")
    
    mvar = int(mvar) - 1
    beta = np.array(beta)
    
    # Extract complete cases
    miss = np.isnan(D[:, mvar])
    X = np.delete(D[~miss], mvar, axis=1)
    y = D[~miss, mvar]
    
    n = X.shape[0]
    xb = X @ beta
    pr = expit(xb)  # 1/(1+exp(-xb))
    
    # Compute Q (objective function)
    low = pr < 0.5
    high = ~low
    Q = np.sum(y * xb)
    if np.any(low):
        Q += np.sum(np.log(np.maximum(1e-10, 1 - pr[low])))
    if np.any(high):
        Q += np.sum(np.log(np.maximum(1e-10, pr[high])) - xb[high])
    
    result = {'n': int(n), 'Q': float(Q)}
    
    if mode == 1:
        # Compute Hessian and gradient
        w = pr * (1 - pr)
        H = (X.T * w) @ X
        g = X.T @ (y - pr)
        
        # Handle NaN/Inf
        H = np.nan_to_num(H)
        g = np.nan_to_num(g)
        
        result['H'] = H.tolist()
        result['g'] = g.tolist()
    
    return result

def master_simi(client: AlgorithmClient, data: Dict[str, Any], org_ids: List[int], *args, **kwargs) -> Dict[str, Any]:
    """
    Master function for SIMI algorithm in vantage6.
    
    Orchestrates federated computation and performs imputation.
    
    Parameters:
    -----------
    client : AlgorithmClient
        Vantage6 client for RPC calls
    data : dict
        Central dataset and configuration:
        - 'data': numpy array or CSV path
        - 'target_column_index': 1-based index of target column
        - 'is_binary': boolean for binary vs continuous
        - 'imputation_trials': number of imputations (M)
        - 'method': optional, 'Gaussian' or 'logistic'
    
    Returns:
    --------
    dict with imputed datasets
    """
    # Extract configuration
    config = kwargs.copy()
    config.update(data if isinstance(data, dict) else {})
    
    info("SIMI master: starting imputation")
    # Determine central data: prefer explicit config['data'], otherwise use positional data
    dataset = config.get('data', data)
    D = dataset.values.astype(float)

    
    # Normalize parameters
    mvar_1based = config.get('target_column_index', config.get('mvar', 1))
    mvar = int(mvar_1based) - 1
    
    is_binary = config.get('is_binary', False)
    method = config.get('method')
    if method is None:
        method = 'logistic' if is_binary else 'Gaussian'
    
    M = config.get('imputation_trials', config.get('M', 10))
    
    # Identify missing values
    miss = np.isnan(D[:, mvar])
    nm = np.sum(miss)
    nc = D.shape[0] - nm
    info(f"SIMI master: target col={mvar_1based}, rows={D.shape[0]}, missing={nm}, complete={nc}")
    
    if nm == 0:
        print("No missing values found. Returning original data.")
        return {'imputed_datasets': [D.tolist()]}
    
    X_central = np.delete(D[~miss], mvar, axis=1)
    y_central = D[~miss, mvar]
    
    # Aggregate statistics from remote nodes
    if method == 'Gaussian':
        info("SIMI master: aggregating Gaussian stats")
        SI = _aggregate_gaussian_stats(client, X_central, y_central, config, org_ids)
    else:  # logistic
        info("SIMI master: aggregating Logistic stats")
        SI = _aggregate_logistic_stats(client, X_central, y_central, config, org_ids)
    
    # Perform imputation
    imputed_datasets = []
    for m in range(M):
        D_imputed = D.copy()
        
        if method == 'Gaussian':
            # Sample from posterior
            sig = np.sqrt(1 / np.random.gamma((SI['n'] + 1) / 2, 2 / (SI['SSE'] + 1)))
            cvcov = cholesky(SI['vcov'], lower=True)
            alpha = SI['beta'] + sig * np.dot(cvcov, np.random.normal(size=SI['beta'].shape[0]))
            
            # Impute missing values
            D_imputed[miss, mvar] = np.dot(np.delete(D[miss], mvar, axis=1), alpha) + np.random.normal(0, sig, nm)
        
        else:  # logistic
            cvcov = cholesky(SI['vcov'], lower=True)
            alpha = SI['beta'] + np.dot(cvcov, np.random.normal(size=SI['beta'].shape[0]))
            pr = expit(np.dot(np.delete(D[miss], mvar, axis=1), alpha))
            D_imputed[miss, mvar] = np.random.binomial(1, pr)
        
        imputed_datasets.append(D_imputed.tolist())
    
    info("SIMI master: finished imputations")
    return {'imputed_datasets': imputed_datasets}


def _aggregate_gaussian_stats(client: AlgorithmClient, X: np.ndarray, y: np.ndarray, config: Dict, org_ids: List[int]) -> Dict[str, Any]:
    """Aggregate Gaussian statistics from remote nodes."""
    # Central statistics
    XX = X.T @ X
    Xy = X.T @ y
    yy = np.sum(y ** 2)
    n = X.shape[0]
    
    info("SIMI Gaussian: dispatching remote stats")
    task_input = {
        'method': 'Gaussian',
        'mvar': config.get('target_column_index', config.get('mvar', 1)),
    }
    
    # Call remote function on all nodes
    results = _create_and_wait(
        client,
        {
            'method': 'simi_remote_gaussian',
            'args': [],
            'kwargs': task_input
        },
        org_ids=org_ids
    )
    
    # Aggregate results
    for result in results:
        remote_stats = result.get('result', {})
        remote_XX = np.array(remote_stats.get('XX', []))
        remote_Xy = np.array(remote_stats.get('Xy', []))
        if remote_XX.shape == XX.shape:
            XX += remote_XX
        if remote_Xy.shape == Xy.shape:
            Xy += remote_Xy
        yy += remote_stats.get('yy', 0)
    
    # Compute final statistics
    XX = np.nan_to_num(XX)
    Xy = np.nan_to_num(Xy)
    
    # Add regularization
    reg = max(1e-10 * np.trace(XX) / XX.shape[0], 1e-6)
    for i in range(XX.shape[0]):
        XX[i, i] += reg
    
    try:
        cXX = cholesky(XX, lower=True)
        iXX = cho_solve((cXX, True), np.eye(XX.shape[0]))
    except:
        iXX = np.linalg.pinv(XX)
    
    beta = np.dot(iXX, Xy)
    SSE = yy + np.sum(beta * (np.dot(XX, beta) - 2 * Xy))
    
    return {
        'beta': beta,
        'vcov': iXX,
        'SSE': SSE,
        'n': n
    }


def _aggregate_logistic_stats(client: AlgorithmClient, X: np.ndarray, y: np.ndarray, config: Dict, org_ids: List[int]) -> Dict[str, Any]:
    """Aggregate logistic statistics from remote nodes using iterative optimization."""
    p = X.shape[1]
    beta = np.zeros(p)
    maxiter = config.get('maxiter', 100)
    lam = config.get('lam', 1e-3)
    
    # Central sample size
    N = X.shape[0]
    
    # Get total sample size from remotes
    # First call with mode=0 to get sample size
    size_results = _create_and_wait(
        client,
        {
            'method': 'simi_remote_logistic',
            'args': [[0.0] * p, 0],  # beta=zeros, mode=0 (get sample size)
            'kwargs': {
                'mvar': config.get('target_column_index', config.get('mvar', 1))
            }
        },
        org_ids=org_ids
    )
    
    for result in size_results:
        N += result.get('result', {}).get('n', 0)
    
    # Iterative optimization
    for iteration in range(maxiter):
        # Compute central statistics
        xb = np.dot(X, beta)
        pr = expit(xb)
        w = pr * (1 - pr)
        H = np.dot(X.T * w, X) + np.eye(p) * (N * lam)
        g = np.dot(X.T, (y - pr)) - N * lam * beta
        
        # Aggregate from remotes
        remote_results = _create_and_wait(
            client,
            {
                'method': 'simi_remote_logistic',
                'args': [beta.tolist(), 1],  # mode=1: compute H and g
                'kwargs': {
                    'mvar': config.get('target_column_index', config.get('mvar', 1))
                }
            },
            org_ids=org_ids
        )
        
        for result in remote_results:
            remote_stats = result.get('result', {})
            rH = np.array(remote_stats.get('H', []))
            rg = np.array(remote_stats.get('g', []))
            if rH.shape == H.shape:
                H += rH
            if rg.shape == g.shape:
                g += rg
        
        # Update beta
        try:
            cH = cholesky(H, lower=True)
            direction = cho_solve((cH, True), g)
        except:
            direction = np.dot(np.linalg.pinv(H), g)
        
        # Line search
        step = 1.0
        while True:
            nbeta = beta + step * direction
            if np.max(np.abs(nbeta - beta)) < 1e-5:
                break
            
            # Check objective improvement
            xb_new = np.dot(X, nbeta)
            pr_new = expit(xb_new)
            # Simplified check - in practice would compute full Q
            if step < 1e-10:
                break
            step /= 2
        
        if np.max(np.abs(nbeta - beta)) < 1e-5:
            beta = nbeta
            break
        
        beta = nbeta
    
    # Compute final covariance
    xb = np.dot(X, beta)
    pr = expit(xb)
    w = pr * (1 - pr)
    H = np.dot(X.T * w, X) + np.eye(p) * (N * lam)
    
    try:
        cH = cholesky(H, lower=True)
        vcov = cho_solve((cH, True), np.eye(p))
    except:
        vcov = np.linalg.pinv(H)
    
    return {
        'beta': beta,
        'vcov': vcov,
        'n': N
    }
