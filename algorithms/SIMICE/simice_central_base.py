"""
SIMICE base classes for central and remote implementations.
Provides base functionality common to both SIMICE implementations.
"""

import numpy as np
import pandas as pd
import asyncio
import websockets
import json
from typing import Dict, Any, List, Optional, Union
from scipy import stats


class SIMICEBase:
    """
    Base class for SIMICE algorithm implementations.
    Contains common functionality for both central and remote sites.
    """
    
    def __init__(self):
        self.target_columns = []
        self.is_binary = []
        self.iteration_before_first_imputation = 5
        self.iteration_between_imputations = 5
        self.imputation_count = 10
    
    def _validate_parameters(self, target_column_indexes: List[int], is_binary: List[bool]):
        """
        Validate input parameters.
        
        Args:
            target_column_indexes: List of 1-based column indices
            is_binary: List of boolean flags for each target column
        """
        if len(target_column_indexes) != len(is_binary):
            raise ValueError("Length of target_column_indexes must match length of is_binary")
        
        if not all(idx >= 1 for idx in target_column_indexes):
            raise ValueError("All target column indexes must be >= 1 (1-based indexing)")
    
    def _prepare_predictor_matrix(self, data: np.ndarray, target_col_idx: int, 
                                include_intercept: bool = True) -> np.ndarray:
        """
        Prepare predictor matrix by excluding the target column and optionally adding intercept.
        
        Args:
            data: Input data matrix
            target_col_idx: Index of target column to exclude
            include_intercept: Whether to add intercept column
            
        Returns:
            Predictor matrix
        """
        # Get all columns except target
        predictor_cols = [i for i in range(data.shape[1]) if i != target_col_idx]
        X = data[:, predictor_cols]
        
        if include_intercept:
            X = np.column_stack([X, np.ones(X.shape[0])])
        
        return X
    
    def _clip_probabilities(self, probs: np.ndarray, min_val: float = 1e-8, 
                           max_val: float = 1 - 1e-8) -> np.ndarray:
        """
        Clip probabilities to prevent numerical issues.
        """
        return np.clip(probs, min_val, max_val)
    
    def _safe_cholesky(self, matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
        """
        Compute Cholesky decomposition with fallback to regularized version.
        
        Args:
            matrix: Square positive definite matrix
            regularization: Regularization parameter for numerical stability
            
        Returns:
            Lower triangular Cholesky factor
        """
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            # Add regularization and try again
            regularized_matrix = matrix + regularization * np.eye(matrix.shape[0])
            try:
                return np.linalg.cholesky(regularized_matrix)
            except np.linalg.LinAlgError:
                # Final fallback: use SVD-based pseudo-square-root
                U, s, Vt = np.linalg.svd(matrix)
                s_sqrt = np.sqrt(np.maximum(s, regularization))
                return U @ np.diag(s_sqrt)
    
    def _solve_linear_system(self, A: np.ndarray, b: np.ndarray, 
                           regularization: float = 1e-6) -> np.ndarray:
        """
        Solve linear system Ax = b with fallback options.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            regularization: Regularization parameter
            
        Returns:
            Solution vector
        """
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Try with regularization
            try:
                A_reg = A + regularization * np.eye(A.shape[0])
                return np.linalg.solve(A_reg, b)
            except np.linalg.LinAlgError:
                # Final fallback: least squares
                return np.linalg.lstsq(A, b, rcond=None)[0]
    
    def _compute_inverse(self, matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
        """
        Compute matrix inverse with fallback options.
        
        Args:
            matrix: Square matrix to invert
            regularization: Regularization parameter
            
        Returns:
            Inverse matrix
        """
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            try:
                # Try with regularization
                reg_matrix = matrix + regularization * np.eye(matrix.shape[0])
                return np.linalg.inv(reg_matrix)
            except np.linalg.LinAlgError:
                # Final fallback: pseudo-inverse
                return np.linalg.pinv(matrix)


async def SIMICECentralLS(X: np.ndarray, y: np.ndarray, expected_sites: int = 1,
                         expected_site_names: Optional[List[str]] = None,
                         port: int = 6000, lam: float = 1e-3) -> Dict[str, Any]:
    """
    Central component of SIMICE algorithm for least squares (Gaussian) imputation.
    
    Args:
        X: Local predictor matrix
        y: Local target variable
        expected_sites: Number of expected remote sites
        expected_site_names: List of expected site identifiers
        port: Port for WebSocket connections
        lam: Regularization parameter
        
    Returns:
        Dictionary containing aggregated statistics
    """
    print(f"Starting SIMICECentralLS with {expected_sites} expected sites")
    
    p = X.shape[1]
    n = X.shape[0]
    
    # Local sufficient statistics
    XX = X.T @ X
    Xy = X.T @ y
    yy = y.T @ y
    
    # Shared state for WebSocket connections
    connected_sites = {}
    remote_sites = 0
    connection_event = asyncio.Event()
    
    async def handler(websocket):
        nonlocal connected_sites, remote_sites, n, XX, Xy, yy
        
        try:
            # Receive site identification
            msg = await websocket.recv()
            data = json.loads(msg)
            
            if data['type'] == 'REMOTE_SITE':
                site_id = data['site_id']
                
                # Validate site_id if expected
                if expected_site_names is not None and site_id not in expected_site_names:
                    print(f"WARNING: Unexpected remote site: {site_id}")
                
                connected_sites[site_id] = websocket
                remote_sites += 1
                
                print(f"Remote site {site_id} connected. Total: {remote_sites}/{expected_sites}")
                
                if remote_sites == expected_sites:
                    connection_event.set()
                
                # Request data from remote site
                await websocket.send(json.dumps({
                    'type': 'REQUEST_DATA',
                    'method': 'gaussian'
                }))
                
                # Receive remote statistics
                response = await websocket.recv()
                remote_data = json.loads(response)
                
                if remote_data['type'] == 'DATA_RESPONSE':
                    stats = remote_data['statistics']
                    
                    # Aggregate statistics
                    XX += np.array(stats['XTX'])
                    Xy += np.array(stats['XTy'])
                    yy += stats['yTy']
                    n += stats['n']
                    
                    print(f"Aggregated data from site {site_id}")
                
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed for site")
        except Exception as e:
            print(f"Error in handler: {e}")
    
    # Start WebSocket server
    server = await websockets.serve(handler, "localhost", port)
    print(f"Central server listening on port {port}")
    
    # Wait for all sites to connect
    await connection_event.wait()
    print("All remote sites connected and data aggregated")
    
    # Solve for beta and compute statistics
    base = SIMICEBase()
    beta = base._solve_linear_system(XX + lam * np.eye(p), Xy)
    SSE = yy - Xy.T @ beta
    vcov = base._compute_inverse(XX + lam * np.eye(p))
    
    # Close server
    server.close()
    await server.wait_closed()
    
    return {
        'beta': beta,
        'vcov': vcov,
        'SSE': SSE,
        'n': n,
        'XTX': XX
    }


async def SIMICECentralLogit(X: np.ndarray, y: np.ndarray, expected_sites: int = 1,
                           expected_site_names: Optional[List[str]] = None,
                           port: int = 6000, max_iter: int = 10) -> Dict[str, Any]:
    """
    Central component of SIMICE algorithm for logistic imputation.
    
    Args:
        X: Local predictor matrix  
        y: Local binary target variable
        expected_sites: Number of expected remote sites
        expected_site_names: List of expected site identifiers
        port: Port for WebSocket connections
        max_iter: Maximum iterations for logistic regression
        
    Returns:
        Dictionary containing aggregated statistics
    """
    print(f"Starting SIMICECentralLogit with {expected_sites} expected sites")
    
    p = X.shape[1]
    n = X.shape[0]
    
    # Initialize beta
    beta = np.zeros(p)
    
    # Shared state
    connected_sites = {}
    remote_sites = 0
    connection_event = asyncio.Event()
    
    async def handler(websocket):
        nonlocal connected_sites, remote_sites, beta
        
        try:
            msg = await websocket.recv()
            data = json.loads(msg)
            
            if data['type'] == 'REMOTE_SITE':
                site_id = data['site_id']
                
                if expected_site_names is not None and site_id not in expected_site_names:
                    print(f"WARNING: Unexpected remote site: {site_id}")
                
                connected_sites[site_id] = websocket
                remote_sites += 1
                
                print(f"Remote site {site_id} connected. Total: {remote_sites}/{expected_sites}")
                
                if remote_sites == expected_sites:
                    connection_event.set()
                
        except Exception as e:
            print(f"Error in logistic handler: {e}")
    
    # Start server
    server = await websockets.serve(handler, "localhost", port)
    print(f"Central logistic server listening on port {port}")
    
    # Wait for connections
    await connection_event.wait()
    
    # Iterative logistic regression
    base = SIMICEBase()
    
    for iteration in range(max_iter):
        # Send current beta to all sites and collect Hessian contributions
        H_total = np.zeros((p, p))
        gradient_total = np.zeros(p)
        
        # Local contribution
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -500, 500)
        probs = 1 / (1 + np.exp(-linear_pred))
        probs = base._clip_probabilities(probs)
        
        W = probs * (1 - probs)
        H_local = X.T @ np.diag(W) @ X
        gradient_local = X.T @ (y - probs)
        
        H_total += H_local
        gradient_total += gradient_local
        
        # Collect from remote sites
        for site_id, websocket in connected_sites.items():
            try:
                await websocket.send(json.dumps({
                    'type': 'REQUEST_GRADIENT',
                    'beta': beta.tolist(),
                    'iteration': iteration
                }))
                
                response = await websocket.recv()
                remote_data = json.loads(response)
                
                if remote_data['type'] == 'GRADIENT_RESPONSE':
                    H_total += np.array(remote_data['H'])
                    gradient_total += np.array(remote_data['gradient'])
                    
            except Exception as e:
                print(f"Error communicating with site {site_id}: {e}")
        
        # Update beta
        try:
            delta = base._solve_linear_system(H_total, gradient_total)
            beta += delta
        except Exception as e:
            print(f"Error updating beta in iteration {iteration}: {e}")
            break
    
    # Final Hessian for variance computation
    H_final = np.zeros((p, p))
    
    # Local Hessian
    linear_pred = X @ beta
    linear_pred = np.clip(linear_pred, -500, 500)
    probs = 1 / (1 + np.exp(-linear_pred))
    probs = base._clip_probabilities(probs)
    W = probs * (1 - probs)
    H_final += X.T @ np.diag(W) @ X
    
    # Collect final Hessian from remote sites
    for site_id, websocket in connected_sites.items():
        try:
            await websocket.send(json.dumps({
                'type': 'FINAL_HESSIAN',
                'beta': beta.tolist()
            }))
            
            response = await websocket.recv()
            remote_data = json.loads(response)
            
            if remote_data['type'] == 'HESSIAN_RESPONSE':
                H_final += np.array(remote_data['H'])
                
        except Exception as e:
            print(f"Error getting final Hessian from site {site_id}: {e}")
    
    vcov = base._compute_inverse(H_final)
    
    # Close server
    server.close()
    await server.wait_closed()
    
    return {
        'beta': beta,
        'vcov': vcov,
        'H': H_final
    }
