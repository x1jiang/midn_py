"""
SIMICE algorithm implementation for the remote site.
Implements the RemoteAlgorithm interface for multiple imputation with chained equations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from common.algorithm.base import RemoteAlgorithm


class SIMICERemoteAlgorithm(RemoteAlgorithm):
    """
    Remote implementation of the SIMICE algorithm.
    Multiple Imputation using Chained Equations for multiple target columns.
    """
    
    def __init__(self):
        """
        Initialize the algorithm instance.
        """
        self.target_columns = []
        self.is_binary = []
        self.missing_masks = {}
        self.current_data = None
        self.original_data = None
    
    @classmethod
    def get_algorithm_name(cls) -> str:
        return "SIMICE"
    
    @classmethod
    def get_supported_methods(cls) -> List[str]:
        return ["gaussian", "logistic"]
    
    def _initialize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Initialize missing values with mean imputation for continuous variables
        and mode imputation for binary variables.
        """
        data_init = data.copy()
        
        for i, col_idx in enumerate(self.target_columns):
            missing_mask = np.isnan(data_init[:, col_idx])
            
            if missing_mask.any():
                non_missing_values = data_init[~missing_mask, col_idx]
                
                if len(non_missing_values) > 0:
                    if self.is_binary[i]:
                        # Mode imputation for binary variables
                        unique_vals, counts = np.unique(non_missing_values, return_counts=True)
                        mode_value = unique_vals[np.argmax(counts)]
                        data_init[missing_mask, col_idx] = mode_value
                    else:
                        # Mean imputation for continuous variables
                        mean_value = np.mean(non_missing_values)
                        data_init[missing_mask, col_idx] = mean_value
                else:
                    # If no non-missing values, use default
                    data_init[missing_mask, col_idx] = 0 if self.is_binary[i] else 0.0
        
        return data_init
    
    async def prepare_data(self, data: np.ndarray, target_column_indexes: List[int],
                         is_binary: List[bool]) -> Dict[str, Any]:
        """
        Prepare data for initial transmission to central.
        
        Args:
            data: Data array
            target_column_indexes: List of 1-based indices of columns to impute
            is_binary: List of booleans indicating if each target column is binary
            
        Returns:
            Data to send to the central site
        """
        # Store parameters (convert to 0-based indexing)
        self.target_columns = [idx - 1 for idx in target_column_indexes]
        self.is_binary = is_binary
        self.original_data = data.copy()
        
        # Create missing value masks
        for col_idx in self.target_columns:
            self.missing_masks[col_idx] = np.isnan(data[:, col_idx])
        
        # Initialize data with simple imputation
        self.current_data = self._initialize_data(data)
        
        # Filter to complete cases for other variables (not target columns)
        other_cols = [i for i in range(data.shape[1]) if i not in self.target_columns]
        complete_case_mask = ~np.any(np.isnan(data[:, other_cols]), axis=1)
        
        # Prepare summary statistics for initial communication
        n_complete = np.sum(complete_case_mask)
        n_total = data.shape[0]
        
        return {
            "n_observations": n_total,
            "n_complete_cases": n_complete,
            "target_columns": target_column_indexes,  # Send back 1-based
            "is_binary": is_binary,
            "status": "initialized"
        }
    
    async def compute_local_statistics(self, target_col_idx: int, method: str) -> Dict[str, Any]:
        """
        Compute local statistics for a specific target column.
        
        Args:
            target_col_idx: 0-based index of the target column
            method: "gaussian" or "logistic"
            
        Returns:
            Local statistics for this target column
        """
        # Get observations where this target column is not missing
        missing_mask = self.missing_masks[target_col_idx]
        non_missing_mask = ~missing_mask
        
        if not non_missing_mask.any():
            # No observations for this column
            p = self.current_data.shape[1]  # Including intercept will be added
            if method.lower() == "gaussian":
                return {
                    'XTX': np.zeros((p, p)),
                    'XTy': np.zeros(p),
                    'yTy': 0.0,
                    'n': 0,
                    'method': method
                }
            else:
                return {
                    'beta': np.zeros(p),
                    'H': np.zeros((p, p)),
                    'method': method
                }
        
        # Prepare predictors (all columns except current target)
        predictor_cols = [i for i in range(self.current_data.shape[1]) if i != target_col_idx]
        X = self.current_data[non_missing_mask][:, predictor_cols]
        # Add intercept
        X = np.column_stack([X, np.ones(X.shape[0])])
        
        # Get target values
        y = self.current_data[non_missing_mask, target_col_idx]
        
        if method.lower() == "gaussian":
            return await self._compute_gaussian_statistics(X, y)
        else:
            return await self._compute_logistic_statistics(X, y)
    
    async def _compute_gaussian_statistics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute sufficient statistics for Gaussian regression following R implementation.
        R code computes: XX (XTX), Xy (XTy), yy (yTy), n
        """
        XTX = X.T @ X
        XTy = X.T @ y  
        yTy = float(y.T @ y)  # Ensure scalar
        n = len(y)
        
        print(f"🔢 SIMICE Gaussian: n={n}, XTX shape={XTX.shape}, XTy shape={XTy.shape}, yTy={yTy:.4f}")
        
        return {
            'XTX': XTX.tolist(),  # Convert to list for JSON serialization
            'XTy': XTy.tolist(),
            'yTy': yTy,
            'n': n,
            'method': 'gaussian'
        }
    
    async def _compute_logistic_statistics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute statistics for logistic regression following R Newton-Raphson implementation.
        R code implements full Newton-Raphson with proper Hessian computation.
        """
        p = X.shape[1]
        beta = np.zeros(p)
        max_iter = 25  # From R implementation
        tol = 1e-6
        
        print(f"🔄 SIMICE Logistic: Starting Newton-Raphson, n={len(y)}, p={p}")
        
        for iteration in range(max_iter):
            try:
                # Compute linear combination  
                eta = X @ beta
                
                # Compute probabilities with numerical stability
                eta_clipped = np.clip(eta, -500, 500)
                mu = 1 / (1 + np.exp(-eta_clipped))
                
                # Compute weights (variance of Bernoulli)
                w = mu * (1 - mu)
                w = np.maximum(w, 1e-8)  # Avoid division by zero
                
                # Compute Hessian (X^T W X)
                W_diag = np.diag(w)
                H = X.T @ W_diag @ X
                
                # Compute gradient (score vector)
                residual = y - mu
                g = X.T @ residual
                
                # Add regularization to ensure invertibility
                lam = 1e-6
                H_reg = H + lam * np.eye(p)
                
                # Newton-Raphson update
                delta = np.linalg.solve(H_reg, g)
                beta_new = beta + delta
                
                # Check convergence
                if np.linalg.norm(delta) < tol:
                    print(f"✅ SIMICE Logistic: Converged at iteration {iteration + 1}")
                    beta = beta_new
                    break
                    
                beta = beta_new
                
            except np.linalg.LinAlgError as e:
                print(f"⚠️ SIMICE Logistic: Numerical error at iteration {iteration}: {e}")
                break
        
        # Compute final Hessian and gradient for transmission
        try:
            eta = X @ beta
            eta_clipped = np.clip(eta, -500, 500)
            mu = 1 / (1 + np.exp(-eta_clipped))
            w = mu * (1 - mu)
            w = np.maximum(w, 1e-8)
            W_diag = np.diag(w)
            H_final = X.T @ W_diag @ X
            # Compute gradient for central aggregation
            residual = y - mu
            g_final = X.T @ residual
        except:
            H_final = np.eye(p)  # Fallback to identity
            g_final = np.zeros(p)  # Fallback to zero gradient
            
        print(f"🔢 SIMICE Logistic: Final beta shape={beta.shape}, H shape={H_final.shape}")
        
        return {
            'H': H_final.tolist(),  # Hessian matrix for aggregation
            'g': g_final.tolist(),  # Gradient vector for aggregation  
            'n': len(y),
            'method': 'logistic'
        }
    
    async def update_imputed_values(self, target_col_idx: int, imputed_values: np.ndarray) -> None:
        """
        Update the local data with new imputed values from central site.
        
        Args:
            target_col_idx: 0-based index of the target column
            imputed_values: New imputed values for missing observations
        """
        missing_mask = self.missing_masks[target_col_idx]
        if missing_mask.any() and len(imputed_values) == np.sum(missing_mask):
            self.current_data[missing_mask, target_col_idx] = imputed_values
    
    async def process_message(self, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from the central site.
        
        Args:
            message_type: Type of the message
            payload: Message payload
            
        Returns:
            Response payload (if any)
        """
        if message_type == "initialize":
            target_column_indexes = payload.get("target_column_indexes", [])
            is_binary = payload.get("is_binary", [])
            
            # Initialize with provided parameters
            self.target_columns = [idx - 1 for idx in target_column_indexes]
            self.is_binary = is_binary
            
            return {"status": "initialized", "ready": True}
        
        elif message_type == "request_statistics":
            target_col_idx = payload.get("target_column_index") - 1  # Convert to 0-based
            method = payload.get("method", "gaussian")
            
            # Compute and return local statistics
            stats = await self.compute_local_statistics(target_col_idx, method)
            return {"statistics": stats, "status": "computed"}
        
        elif message_type == "update_imputations":
            target_col_idx = payload.get("target_column_index") - 1  # Convert to 0-based
            imputed_values = np.array(payload.get("imputed_values", []))
            
            # Update local data
            await self.update_imputed_values(target_col_idx, imputed_values)
            return {"status": "updated"}
        
        elif message_type == "iteration_complete":
            return {"status": "acknowledged", "ready_for_next": True}
        
        elif message_type == "get_final_data":
            # Return final imputed data for result collection
            final_data = await self.get_final_imputed_data()
            return {"final_data": final_data, "status": "completed"}
        
        else:
            return {"error": f"Unknown message type: {message_type}"}
    
    async def update_imputations(self, target_col_idx: int, global_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update local imputations using global parameters.
        
        Args:
            target_col_idx: 0-based index of the target column
            global_parameters: Global parameters computed from aggregated statistics
            
        Returns:
            Status information
        """
        print(f"🔄 SIMICE Remote: Updating imputations for column {target_col_idx}")
        
        if target_col_idx not in self.missing_masks:
            print(f"⚠️ SIMICE Remote: Column {target_col_idx} not in target columns")
            return {"status": "error", "message": "Column not in target columns"}
        
        missing_mask = self.missing_masks[target_col_idx]
        method = global_parameters.get("method", "gaussian")
        beta = np.array(global_parameters.get("beta", []))
        
        if not missing_mask.any():
            print(f"ℹ️ SIMICE Remote: No missing values in column {target_col_idx}")
            return {"status": "completed", "message": "No missing values to impute"}
        
        # Prepare predictors (all columns except current target)
        predictor_cols = [i for i in range(self.current_data.shape[1]) if i != target_col_idx]
        X_missing = self.current_data[missing_mask][:, predictor_cols]
        
        # Add intercept
        X_missing = np.column_stack([X_missing, np.ones(X_missing.shape[0])])
        
        # Generate new imputations using global parameters
        if method.lower() == "gaussian":
            # For Gaussian: y = X*beta + noise
            predictions = X_missing @ beta
            # Add some noise (simplified)
            noise = np.random.normal(0, 0.1, predictions.shape)
            new_values = predictions + noise
            
        else:  # logistic
            # For logistic: p = sigmoid(X*beta)
            logits = X_missing @ beta
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))  # Clip to prevent overflow
            
            # Sample from Bernoulli distribution
            new_values = np.random.binomial(1, probabilities).astype(float)
        
        # Update the current data with new imputations
        self.current_data[missing_mask, target_col_idx] = new_values
        
        print(f"✅ SIMICE Remote: Updated {np.sum(missing_mask)} missing values in column {target_col_idx}")
        print(f"   Method: {method}, Mean imputed value: {np.mean(new_values):.4f}")
        
        return {
            "status": "completed",
            "n_imputed": int(np.sum(missing_mask)),
            "mean_value": float(np.mean(new_values)),
            "method": method
        }
    
    async def get_final_imputed_data(self) -> Dict[str, Any]:
        """
        Get the final imputed datasets for all imputations.
        Returns a dictionary with imputation datasets.
        """
        try:
            import pandas as pd
            
            # For now, return the current imputed data as a single imputation
            # In a full SIMICE implementation, you would store multiple imputations
            # during the algorithm execution
            
            if self.current_data is not None:
                # Convert to DataFrame for easier handling
                df = pd.DataFrame(self.current_data)
                
                # Return just the final imputed data - the central server will replicate it
                # based on the imputation_trials parameter
                result = {
                    "final_data": df.copy()
                }
                
                print(f"📊 SIMICE Remote: Returning final imputed dataset")
                return result
            else:
                print("⚠️ SIMICE Remote: No current data available")
                return {}
                
        except Exception as e:
            print(f"💥 SIMICE Remote: Error getting final data: {e}")
            return {}
