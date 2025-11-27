"""
SIMICE Algorithm for Vantage6
Single Imputation for Multiple Columns - adapted for vantage6 framework

Original algorithm from MIDN_R_PY/SIMICE, converted to use vantage6's RPC pattern.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.linalg import cholesky, cho_solve
from scipy.special import expit
import json

# Import vantage6 tools
try:
    from vantage6.algorithm.tools import AlgorithmClient
    from vantage6.algorithm.tools.decorators import algorithm_client
except ImportError:
    print("Warning: vantage6 not installed. Using mock interface.")
    class AlgorithmClient:
        def __init__(self, *args, **kwargs):
            pass
        def get_data(self, *args, **kwargs):
            return None
        def create_new_task(self, *args, **kwargs):
            return None


def RPC_simice_remote_initialize(*args, **kwargs) -> Dict[str, Any]:
    """
    Remote function to initialize SIMICE for multiple target columns.
    
    Parameters (via kwargs):
    -----------
    data : dict with 'data' key containing dataset
    mvar_list : list of 1-based target column indices
    
    Returns:
    --------
    dict with initialization status
    """
    # Extract from kwargs (vantage6 passes everything via kwargs)
    data = kwargs.get('data', {})
    mvar_list = kwargs.get('mvar_list', [])
    
    if isinstance(data, dict):
        dataset = data.get('data', data)
    else:
        dataset = data
    
    if isinstance(dataset, str):
        D = pd.read_csv(dataset).values
    elif isinstance(dataset, np.ndarray):
        D = dataset
    else:
        raise ValueError("Data must be numpy array or CSV path")
    
    # Convert to 0-based
    mvar_py = [int(m) - 1 for m in mvar_list]
    
    # Prepare data (add intercept column)
    D_aug = np.column_stack([D.astype(float), np.ones((D.shape[0],), dtype=float)])
    
    # Initialize missing values with column means
    miss = np.isnan(D_aug)
    for j in mvar_py:
        mj = miss[:, j]
        if np.any(mj):
            obs = ~mj
            mu = float(np.nanmean(D_aug[obs, j])) if np.any(obs) else 0.0
            D_aug[mj, j] = mu
    
    return {'status': 'initialized', 'n': D_aug.shape[0]}


def RPC_simice_remote_statistics(*args, **kwargs) -> Dict[str, Any]:
    """
    Remote function to compute statistics for SIMICE iteration.
    
    Parameters (via kwargs):
    -----------
    data : dict with dataset
    mvar_list : list of target column indices (1-based)
    beta_list : list of coefficient vectors (one per target)
    method_list : list of methods ('Gaussian' or 'logistic')
    
    Returns:
    --------
    dict with aggregated statistics
    """
    # Extract from kwargs (vantage6 passes everything via kwargs)
    data = kwargs.get('data', {})
    mvar_list = kwargs.get('mvar_list', [])
    beta_list = kwargs.get('beta_list', [])
    method_list = kwargs.get('method_list', [])
    
    if isinstance(data, dict):
        dataset = data.get('data', data)
    else:
        dataset = data
    
    if isinstance(dataset, str):
        D = pd.read_csv(dataset).values
    elif isinstance(dataset, np.ndarray):
        D = dataset
    else:
        raise ValueError("Data must be numpy array or CSV path")
    
    D_aug = np.column_stack([D.astype(float), np.ones((D.shape[0],), dtype=float)])
    mvar_py = [int(m) - 1 for m in mvar_list]
    
    results = {}
    for idx, (mvar, beta, method) in enumerate(zip(mvar_py, beta_list, method_list)):
        beta = np.array(beta)
        X = np.delete(D_aug, mvar, axis=1)
        y = D_aug[:, mvar]
        
        if method.lower() == 'gaussian':
            XX = X.T @ X
            Xy = X.T @ y
            yy = np.sum(y ** 2)
            results[f'col_{idx}'] = {
                'XX': np.nan_to_num(XX).tolist(),
                'Xy': np.nan_to_num(Xy).tolist(),
                'yy': float(np.nan_to_num(yy))
            }
        else:  # logistic
            xb = X @ beta
            pr = expit(xb)
            w = pr * (1 - pr)
            H = (X.T * w) @ X
            g = X.T @ (y - pr)
            results[f'col_{idx}'] = {
                'H': np.nan_to_num(H).tolist(),
                'g': np.nan_to_num(g).tolist()
            }
    
    return results


def master_simice(client: AlgorithmClient, data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Master function for SIMICE algorithm in vantage6.
    
    Handles multiple target columns with iterative imputation.
    
    Parameters:
    -----------
    client : AlgorithmClient
        Vantage6 client for RPC calls
    data : dict with:
        - 'data': central dataset
        - 'target_column_indexes': list of 1-based indices
        - 'is_binary_list': list of booleans per column
        - 'imputation_trials': number of imputations
        - 'iteration_before_first_imputation': iterations before first imputation
        - 'iteration_between_imputations': iterations between imputations
    
    Returns:
    --------
    dict with imputed datasets
    """
    config = kwargs.copy()
    config.update(data if isinstance(data, dict) else {})
    
    # Load central data
    if isinstance(config.get('data'), str):
        D = pd.read_csv(config['data']).values.astype(float)
    elif isinstance(config.get('data'), np.ndarray):
        D = config['data'].astype(float)
    else:
        raise ValueError("Central data must be provided")
    
    # Extract parameters
    mvar_list_1b = config.get('target_column_indexes', config.get('mvar', []))
    is_binary_list = config.get('is_binary_list', [])
    method_list = []
    for is_bin in is_binary_list:
        method_list.append('logistic' if is_bin else 'Gaussian')
    
    M = config.get('imputation_trials', config.get('M', 10))
    iter0 = config.get('iteration_before_first_imputation', config.get('iter0_val', 0))
    iter_val = config.get('iteration_between_imputations', config.get('iter_val', 0))
    
    # Initialize
    D_aug = np.column_stack([D, np.ones((D.shape[0],), dtype=float)])
    mvar_py = [int(m) - 1 for m in mvar_list_1b]
    
    # Initialize missing values
    miss = np.isnan(D_aug)
    for j in mvar_py:
        mj = miss[:, j]
        if np.any(mj):
            obs = ~mj
            mu = float(np.nanmean(D_aug[obs, j])) if np.any(obs) else 0.0
            D_aug[mj, j] = mu
    
    # Initialize remote nodes
    task_input = {
        'data': config.get('data'),
        'mvar_list': mvar_list_1b
    }
    
    init_results = client.create_new_task(
        input_={
            'method': 'simice_remote_initialize',
            'args': [],
            'kwargs': task_input
        },
        organization_ids=[]
    )
    
    # Verify initialization
    if not init_results:
        print("Warning: No remote nodes responded to initialization")
    
    # Iterations before first imputation
    beta_list = [np.zeros(D_aug.shape[1] - 1) for _ in mvar_py]
    for _ in range(iter0):
        beta_list = _simice_iteration(client, D_aug, mvar_py, beta_list, method_list, config)
    
    # Generate imputations
    imputed_datasets = []
    for m in range(M):
        D_imputed = D_aug.copy()
        
        # Iterations between imputations
        for _ in range(iter_val):
            beta_list = _simice_iteration(client, D_imputed, mvar_py, beta_list, method_list, config)
        
        # Impute missing values
        for idx, (j, method) in enumerate(zip(mvar_py, method_list)):
            mj = miss[:, j]
            if not np.any(mj):
                continue
            
            beta = beta_list[idx]
            X_miss = np.delete(D_imputed[mj], j, axis=1)
            
            if method == 'Gaussian':
                # Sample from Gaussian posterior
                # Simplified: use point estimate + noise
                pred = X_miss @ beta
                D_imputed[mj, j] = pred + np.random.normal(0, 0.1, np.sum(mj))
            else:  # logistic
                pr = expit(X_miss @ beta)
                D_imputed[mj, j] = np.random.binomial(1, pr)
        
        imputed_datasets.append(D_imputed[:, :-1].tolist())  # Remove intercept column
    
    return {'imputed_datasets': imputed_datasets}


def _simice_iteration(client: AlgorithmClient, D: np.ndarray, mvar_py: List[int],
                     beta_list: List[np.ndarray], method_list: List[str], config: Dict) -> List[np.ndarray]:
    """Perform one iteration of SIMICE."""
    mvar_list_1b = [m + 1 for m in mvar_py]
    
    # Request statistics from remotes
    task_input = {
        'data': config.get('data'),
        'mvar_list': mvar_list_1b,
        'beta_list': [b.tolist() for b in beta_list],
        'method_list': method_list
    }
    
    results = client.create_new_task(
        input_={
            'method': 'simice_remote_statistics',
            'args': [],
            'kwargs': task_input
        },
        organization_ids=[]
    )
    
    if not results:
        print("Warning: No remote nodes responded to statistics request")
        return beta_list  # Return unchanged if no remotes
    
    # Aggregate and update betas
    new_beta_list = []
    for idx, (j, method) in enumerate(zip(mvar_py, method_list)):
        # Aggregate central + remote statistics
        X = np.delete(D, j, axis=1)
        y = D[:, j]
        
        if method == 'Gaussian':
            XX = X.T @ X
            Xy = X.T @ y
            
            for result in results:
                remote_stats = result.get('result', {}).get(f'col_{idx}', {})
                remote_XX = remote_stats.get('XX', [])
                remote_Xy = remote_stats.get('Xy', [])
                if remote_XX and len(remote_XX) > 0:
                    XX += np.array(remote_XX)
                if remote_Xy and len(remote_Xy) > 0:
                    Xy += np.array(remote_Xy)
            
            # Solve
            reg = 1e-6 * np.eye(XX.shape[0])
            beta = np.linalg.solve(XX + reg, Xy)
        else:  # logistic
            beta = beta_list[idx]
            xb = X @ beta
            pr = expit(xb)
            w = pr * (1 - pr)
            H = (X.T * w) @ X
            g = X.T @ (y - pr)
            
        for result in results:
            remote_stats = result.get('result', {}).get(f'col_{idx}', {})
            remote_H = remote_stats.get('H', [])
            remote_g = remote_stats.get('g', [])
            if remote_H and len(remote_H) > 0:
                H += np.array(remote_H)
            if remote_g and len(remote_g) > 0:
                g += np.array(remote_g)
            
            # Update
            try:
                cH = cholesky(H, lower=True)
                direction = cho_solve((cH, True), g)
            except:
                direction = np.dot(np.linalg.pinv(H), g)
            
            beta = beta + 0.1 * direction  # Simple step
        
        new_beta_list.append(beta)
    
    return new_beta_list


# Vantage6 entry point
if __name__ == "__main__":
    from vantage6.algorithm.tools import AlgorithmClient
    from vantage6.algorithm.tools.decorators import algorithm_client
    
    @algorithm_client
    def main(client: AlgorithmClient, data: Dict[str, Any], *args, **kwargs):
        """Main entry point for vantage6."""
        return master_simice(client, data, *args, **kwargs)

