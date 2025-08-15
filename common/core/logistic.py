"""
Logistic regression core functions - Python equivalent of R Core/Logit.R
Provides centralized Newton-Raphson logistic regression with proper regularization.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.linalg import cholesky, solve_triangular, LinAlgError


def Logit(X: np.ndarray, y: np.ndarray, offset: Optional[np.ndarray] = None, 
          beta0: Optional[np.ndarray] = None, lam: float = 1e-3, maxiter: int = 100) -> Dict[str, Any]:
    """
    Newton-Raphson logistic regression - Python equivalent of R Logit() function.
    
    Args:
        X: Design matrix (n x p)
        y: Binary response vector (n,)
        offset: Offset vector (p,), defaults to zeros
        beta0: Initial coefficient vector (p,), defaults to zeros
        lam: Regularization parameter
        maxiter: Maximum number of iterations
        
    Returns:
        Dictionary containing:
        - beta: Coefficient vector
        - H: Hessian matrix at convergence
        - converged: Whether algorithm converged
        - iterations: Number of iterations used
    """
    p = X.shape[1]
    n = X.shape[0]
    
    if offset is None:
        offset = np.zeros(p)
    if beta0 is None:
        beta0 = np.zeros(p)
    
    beta = beta0.copy()
    
    for iteration in range(maxiter):
        # Linear predictor
        xb = X @ beta
        # Clip to prevent overflow
        xb = np.clip(xb, -500, 500)
        
        # Probabilities
        pr = 1.0 / (1.0 + np.exp(-xb))
        
        # Hessian and gradient
        w = pr * (1 - pr)
        w = np.maximum(w, 1e-8)  # Prevent zero weights
        H = X.T @ (X * w[:, np.newaxis]) + n * lam * np.eye(p)
        g = X.T @ (y - pr) + n * offset - n * lam * beta
        
        # Log-likelihood (for line search)
        Q = (np.sum(y * xb) + 
             np.sum(np.log(1 - pr[pr < 0.5])) + 
             np.sum(np.log(pr[pr >= 0.5]) - xb[pr >= 0.5]) +
             np.sum(n * offset * beta) - n * lam * np.sum(beta**2) / 2)
        
        # Newton direction
        try:
            L = cholesky(H, lower=True)
            dir_vec = solve_triangular(L, g, lower=True)
            dir_vec = solve_triangular(L.T, dir_vec, lower=False)
        except LinAlgError:
            # Fallback to pseudo-inverse
            dir_vec = np.linalg.pinv(H) @ g
        
        # Line search (simplified version of R implementation)
        m = np.dot(dir_vec, g)
        step = 1.0
        
        while True:
            nbeta = beta + step * dir_vec
            
            if np.max(np.abs(nbeta - beta)) < 1e-5:
                break
                
            # Evaluate new log-likelihood
            xb_new = X @ nbeta
            xb_new = np.clip(xb_new, -500, 500)
            pr_new = 1.0 / (1.0 + np.exp(-xb_new))
            
            nQ = (np.sum(y * xb_new) +
                  np.sum(np.log(1 - pr_new[pr_new < 0.5])) +
                  np.sum(np.log(pr_new[pr_new >= 0.5]) - xb_new[pr_new >= 0.5]) +
                  np.sum(n * offset * nbeta) - n * lam * np.sum(nbeta**2) / 2)
            
            if nQ - Q > m * step / 2:
                break
                
            step = step / 2
            
        # Check convergence
        if np.max(np.abs(nbeta - beta)) < 1e-5:
            return {
                'beta': nbeta,
                'H': H,
                'converged': True,
                'iterations': iteration + 1
            }
            
        beta = nbeta
    
    return {
        'beta': beta,
        'H': H,
        'converged': False,
        'iterations': maxiter
    }


def SILogitNet(D: np.ndarray, idx: np.ndarray, yidx: int,
               remote_stats: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Federated logistic regression coordinator - Python equivalent of R SILogitNet() function.
    
    Args:
        D: Local data matrix  
        idx: Indices of complete observations
        yidx: Index of target variable (0-based)
        remote_stats: List of Hessian and gradient statistics from remote sites
        
    Returns:
        Aggregated statistics for federated logistic regression
    """
    p = D.shape[1] - 1  # Assuming intercept column
    n = len(idx)
    
    # Local computation
    X = D[idx, :]
    X = np.delete(X, yidx, axis=1)  # Remove target column
    y = D[idx, yidx]
    
    # Use local Logit function to get converged parameters
    local_result = Logit(X, y)
    
    # Start with local Hessian and compute gradient
    beta = local_result['beta']
    H_total = local_result['H'].copy()
    
    # Compute local gradient at converged beta
    xb = X @ beta
    xb = np.clip(xb, -500, 500)
    pr = 1.0 / (1.0 + np.exp(-xb))
    g_total = X.T @ (y - pr)
    
    N = n
    
    # Aggregate remote statistics
    if remote_stats:
        for stats in remote_stats:
            H_remote = np.array(stats.get('H', np.zeros((p, p))))
            g_remote = np.array(stats.get('g', np.zeros(p)))
            n_remote = stats.get('n', 0)
            
            H_total += H_remote
            g_total += g_remote
            N += n_remote
    
    # Solve aggregated system
    lam = 1e-3
    H_reg = H_total + N * lam * np.eye(p)
    
    try:
        # Final beta from aggregated statistics
        L = cholesky(H_reg, lower=True)
        beta_final = solve_triangular(L, g_total, lower=True)
        beta_final = solve_triangular(L.T, beta_final, lower=False)
    except LinAlgError:
        beta_final = np.linalg.pinv(H_reg) @ g_total
    
    return {
        'beta': beta_final,
        'H': H_total,
        'N': N
    }


def ImputeLogit(yidx: int, alpha: np.ndarray, manager=None, participants: List[str] = None):
    """
    Send logistic imputation parameters to remote sites - Python equivalent of R ImputeLogit().
    
    Args:
        yidx: Target variable index (0-based)
        alpha: Logistic regression coefficients
        manager: Connection manager for sending messages  
        participants: List of participant site IDs
    """
    if manager and participants:
        message = {
            'type': 'impute_logistic',
            'target_column_index': yidx + 1,  # Convert to 1-based
            'alpha': alpha.tolist(),
            'method': 'logistic'
        }
        
        for participant_id in participants:
            manager.send_to_site(message, participant_id)
    
    return {
        'status': 'parameters_sent',
        'yidx': yidx,
        'n_sites': len(participants) if participants else 0
    }
