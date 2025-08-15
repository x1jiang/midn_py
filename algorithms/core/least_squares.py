"""
Least squares core functions - Python equivalent of R Core/LS.R
Provides centralized implementations for federated least squares computations.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy.linalg import cholesky, solve_triangular, LinAlgError


def LS(X: np.ndarray, y: np.ndarray, offset: Optional[np.ndarray] = None, lam: float = 1e-3) -> Dict[str, Any]:
    """
    Basic least squares with regularization - Python equivalent of R LS() function.
    
    Args:
        X: Design matrix (n x p)
        y: Response vector (n,)
        offset: Offset vector (p,), defaults to zeros
        lam: Regularization parameter
        
    Returns:
        Dictionary containing:
        - beta: Coefficient vector
        - n: Sample size
        - SSE: Sum of squared errors
        - cgram: Cholesky factor of X'X + lambda*I
    """
    p = X.shape[1]
    n = X.shape[0]
    
    if offset is None:
        offset = np.zeros(p)
    
    # Compute sufficient statistics
    XX = X.T @ X + lam * np.eye(p)
    Xy = X.T @ y
    yy = np.sum(y**2)
    
    # Cholesky decomposition
    try:
        cXX = cholesky(XX, lower=True)
        iXX = solve_triangular(cXX, np.eye(p), lower=True)
        iXX = iXX.T @ iXX  # (L^-1)^T L^-1 = (LL^T)^-1 = XX^-1
    except LinAlgError:
        # Fallback to pseudo-inverse
        iXX = np.linalg.pinv(XX)
        cXX = None
    
    # Compute coefficients
    beta = iXX @ (Xy - n * offset)
    
    # Compute SSE
    e = y - X @ beta
    SSE = np.sum(e**2)
    
    return {
        'beta': beta,
        'n': n,
        'SSE': SSE,
        'cgram': cXX
    }


def SILSNet(D: np.ndarray, idx: np.ndarray, yidx: int, 
            lam: float = 1e-3, remote_stats: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Federated least squares coordinator - Python equivalent of R SILSNet() function.
    Aggregates local and remote statistics for federated regression.
    
    Args:
        D: Local data matrix
        idx: Indices of complete observations
        yidx: Index of target variable (0-based in Python)
        lam: Regularization parameter
        remote_stats: List of statistics from remote sites
        
    Returns:
        Aggregated statistics for federated least squares
    """
    p = D.shape[1] - 1  # Assuming intercept column
    n = len(idx)
    
    # Local computation
    X = D[idx, :]
    X = np.delete(X, yidx, axis=1)  # Remove target column
    y = D[idx, yidx]
    
    XX = X.T @ X
    Xy = X.T @ y
    yy = np.sum(y**2)
    
    # Aggregate remote statistics
    N = n
    if remote_stats:
        for stats in remote_stats:
            N += stats.get('n', 0)
            XX += np.array(stats.get('XTX', np.zeros((p, p))))
            Xy += np.array(stats.get('XTy', np.zeros(p)))
            yy += stats.get('yTy', 0.0)
    
    # Add regularization
    XX_reg = XX + N * lam * np.eye(p)
    
    try:
        cXX = cholesky(XX_reg, lower=True)
        iXX = solve_triangular(cXX, np.eye(p), lower=True)
        iXX = iXX.T @ iXX
    except LinAlgError:
        iXX = np.linalg.pinv(XX_reg)
        cXX = None
    
    # Compute aggregated coefficients
    beta = iXX @ Xy
    
    # Compute aggregated SSE
    SSE = yy - Xy.T @ beta
    
    return {
        'beta': beta,
        'N': N,
        'SSE': SSE,
        'cgram': cXX
    }


def ImputeLS(yidx: int, beta: np.ndarray, sig: float, manager=None, participants: List[str] = None):
    """
    Send Gaussian imputation parameters to remote sites - Python equivalent of R ImputeLS().
    
    Args:
        yidx: Target variable index (0-based)
        beta: Regression coefficients
        sig: Error standard deviation
        manager: Connection manager for sending messages
        participants: List of participant site IDs
    """
    if manager and participants:
        from ...communication.messages import create_message
        
        message = create_message(
            "impute_gaussian",
            target_column_index=yidx + 1,  # Convert to 1-based for UI compatibility
            beta=beta.tolist(),
            sigma=sig,
            method="gaussian"
        )
        
        for participant_id in participants:
            # Send imputation parameters to each site
            manager.send_to_site(message, participant_id)
    
    return {
        'status': 'parameters_sent',
        'yidx': yidx,
        'n_sites': len(participants) if participants else 0
    }
