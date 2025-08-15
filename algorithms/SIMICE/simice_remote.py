"""
SIMICE algorithm implementation for the remote site.
Implements the RemoteAlgorithm interface for multiple imputation with chained equations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from common.algorithm.base import RemoteAlgorithm
from ..core.least_squares import LS
from ..core.logistic import Logit
from ..core.transfer import package_gaussian_stats, package_logistic_stats


class SIMICERemoteAlgorithm(RemoteAlgorithm):
    """
    Remote implementation of the SIMICE algorithm.
    Multiple Imputation using Chained Equations for multiple target columns.
    Uses core statistical functions for R-compliant computations.
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
        
        # Add intercept column like R reference: D = cbind(D,1)
        data_with_intercept = np.column_stack([data, np.ones(data.shape[0])])
        
        # Create missing value masks for target columns only
        for col_idx in self.target_columns:
            self.missing_masks[col_idx] = np.isnan(data_with_intercept[:, col_idx])
        
        # Initialize data with simple imputation (following R reference)
        self.current_data = self._initialize_data(data_with_intercept)
        
        # Filter to complete cases for non-target variables (like R: idx = rowSums(miss[,-mvar]) == 0)
        non_target_cols = [i for i in range(data_with_intercept.shape[1]) if i not in self.target_columns]
        complete_case_mask = ~np.any(np.isnan(data_with_intercept[:, non_target_cols]), axis=1)
        
        # Apply complete case filter
        self.current_data = self.current_data[complete_case_mask]
        
        # Update missing masks to reflect filtered data
        for col_idx in self.target_columns:
            original_missing = np.isnan(data_with_intercept[:, col_idx])
            self.missing_masks[col_idx] = original_missing[complete_case_mask]
        
        # Prepare summary statistics for initial communication
        n_complete = np.sum(complete_case_mask)
        n_total = data.shape[0]
        
        print(f"📊 SIMICE Remote: Prepared data - {n_total} total obs, {n_complete} complete cases")
        print(f"🎯 SIMICE Remote: Target columns: {self.target_columns} (0-based)")
        print(f"📈 SIMICE Remote: Data shape with intercept: {self.current_data.shape}")
        
        return {
            "n_observations": n_total,
            "n_complete_cases": n_complete,
            "target_columns": target_column_indexes,  # Send back 1-based for consistency
            "is_binary": is_binary,
            "status": "initialized"
        }
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
        Following R reference: X = matrix(DD[!miss[,yidx],-yidx],ncol=p-1)
        
        Args:
            target_col_idx: 0-based index of the target column
            method: "gaussian" or "logistic"
            
        Returns:
            Local statistics for this target column
        """
        if target_col_idx not in self.missing_masks:
            return {"error": f"Target column {target_col_idx} not initialized"}
        
        # Get observations where this target column is not missing (like R: !miss[,yidx])
        missing_mask = self.missing_masks[target_col_idx]
        non_missing_mask = ~missing_mask
        
        if not non_missing_mask.any():
            # No observations for this column - return zero statistics
            p = self.current_data.shape[1] - 1  # Exclude target column
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
        
        # Prepare predictors: all columns except current target (like R: -yidx)
        all_cols = list(range(self.current_data.shape[1]))
        predictor_cols = [i for i in all_cols if i != target_col_idx]
        
        # Get non-missing observations for this target column
        X = self.current_data[non_missing_mask][:, predictor_cols]
        y = self.current_data[non_missing_mask, target_col_idx]
        
        print(f"🔢 SIMICE Statistics: Column {target_col_idx}, X shape: {X.shape}, y shape: {y.shape}")
        print(f"   Non-missing observations: {np.sum(non_missing_mask)}/{len(missing_mask)}")
        
        if method.lower() == "gaussian":
            return await self._compute_gaussian_statistics(X, y)
        else:
            return await self._compute_logistic_statistics(X, y)
    
    async def _compute_gaussian_statistics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute sufficient statistics for Gaussian regression using core LS function.
        This now uses the centralized LS() function equivalent to R Core/LS.R
        """
        # Use core LS function for R-compliant computation
        ls_result = LS(X, y, lam=1e-3)
        
        # Extract the components needed for federated aggregation
        # Following R pattern: compute XTX, XTy, yTy, n
        XTX = X.T @ X  # We need raw statistics for aggregation
        XTy = X.T @ y
        yTy = float(y.T @ y)
        n = len(y)
        
        print(f"🔢 SIMICE Gaussian (using core LS): n={n}, XTX shape={XTX.shape}, XTy shape={XTy.shape}, yTy={yTy:.4f}")
        print(f"🎯 SIMICE Gaussian: Core LS computed beta=[{', '.join(f'{x:.3f}' for x in ls_result['beta'][:3])}...]")
        
        return {
            'XTX': XTX.tolist(),  # Raw statistics for central aggregation
            'XTy': XTy.tolist(),
            'yTy': yTy,
            'n': n,
            'method': 'gaussian'
        }
    
    async def _compute_logistic_statistics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute statistics for logistic regression using core Logit function.
        This now uses the centralized Logit() function equivalent to R Core/Logit.R
        """
        n, p = X.shape
        
        # Use core Logit function for R-compliant Newton-Raphson computation
        logit_result = Logit(X, y, lam=1e-3, maxiter=25)
        
        # Extract converged parameters
        beta = logit_result['beta']
        converged = logit_result['converged']
        iterations = logit_result['iterations']
        
        # Compute final Hessian and gradient at converged beta for federated aggregation
        # This matches the R implementation's approach
        xb = X @ beta
        xb = np.clip(xb, -500, 500)
        pr = 1 / (1 + np.exp(-xb))
        w = pr * (1 - pr)
        w = np.maximum(w, 1e-8)
        
        # Final Hessian and gradient (what central needs for aggregation)
        H = X.T @ (X * w[:, np.newaxis]) + n * 1e-3 * np.eye(p)  # Include regularization
        g = X.T @ (y - pr)
        
        # Compute log-likelihood for monitoring
        log_lik = np.sum(y * xb - np.log(1 + np.exp(xb)))
        
        print(f"🔄 SIMICE Logistic (using core Logit): Converged={converged} after {iterations} iterations")
        print(f"🔢 SIMICE Logistic: n={n}, p={p}, log_lik={log_lik:.4f}")
        print(f"🎯 SIMICE Logistic: Core Logit beta=[{', '.join(f'{x:.3f}' for x in beta[:3])}...]")
        print(f"🔢 SIMICE Logistic: H shape={H.shape}, g shape={g.shape}")
        
        return {
            'H': H.tolist(),      # Hessian matrix for aggregation
            'g': g.tolist(),      # Gradient vector for aggregation  
            'n': n,               # Sample size
            'log_lik': float(log_lik),  # Log-likelihood for monitoring
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
        Following R reference: X = matrix(DD[midx,-yidx],ncol=p-1)
        
        Args:
            target_col_idx: 0-based index of the target column
            global_parameters: Global parameters (should contain coefficients and method)
            
        Returns:
            Status information
        """
        print(f"🔄 SIMICE Remote: Updating imputations for column {target_col_idx}")
        
        if target_col_idx not in self.missing_masks:
            print(f"⚠️ SIMICE Remote: Column {target_col_idx} not in target columns")
            return {"status": "error", "message": "Column not in target columns"}
        
        missing_mask = self.missing_masks[target_col_idx]
        method = global_parameters.get("method", "gaussian")
        
        if not missing_mask.any():
            print(f"ℹ️ SIMICE Remote: No missing values in column {target_col_idx}")
            return {"status": "completed", "message": "No missing values to impute"}
        
        # Get indices of missing values (like R: midx = which(miss[,yidx]))
        missing_indices = np.where(missing_mask)[0]
        
        # Prepare predictors for missing observations: all columns except current target (like R: -yidx)
        all_cols = list(range(self.current_data.shape[1]))
        predictor_cols = [i for i in all_cols if i != target_col_idx]
        
        # Get predictor data for missing observations (like R: DD[midx,-yidx])
        X_missing = self.current_data[missing_indices][:, predictor_cols]
        
        print(f"🔢 SIMICE Imputation: Missing indices: {len(missing_indices)}, X shape: {X_missing.shape}")
        
        # Generate new imputations using global parameters
        if method.lower() == "gaussian":
            # For Gaussian: D[midx,j] = D[midx,-j] %*% alpha + rnorm(nmidx,0,sig)
            beta = np.array(global_parameters.get("beta", []))
            sigma = global_parameters.get("sigma", 0.1)
            
            if len(beta) != X_missing.shape[1]:
                print(f"💥 SIMICE: Beta dimension mismatch. Expected {X_missing.shape[1]}, got {len(beta)}")
                return {"status": "error", "message": f"Beta dimension mismatch"}
            
            predictions = X_missing @ beta
            noise = np.random.normal(0, sigma, predictions.shape)
            new_values = predictions + noise
            
        else:  # logistic
            # For logistic: pr = 1 / (1 + exp(-D[midx,-j] %*% alpha)); D[midx,j] = rbinom(nmidx,1,pr)
            alpha = np.array(global_parameters.get("alpha", []))
            
            if len(alpha) != X_missing.shape[1]:
                print(f"💥 SIMICE: Alpha dimension mismatch. Expected {X_missing.shape[1]}, got {len(alpha)}")
                return {"status": "error", "message": f"Alpha dimension mismatch"}
            
            logits = X_missing @ alpha
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))  # Clip to prevent overflow
            new_values = np.random.binomial(1, probabilities).astype(float)
        
        # Update the current data with new imputations
        self.current_data[missing_indices, target_col_idx] = new_values
        
        print(f"✅ SIMICE Remote: Updated {len(missing_indices)} missing values in column {target_col_idx}")
        print(f"   Method: {method}, Mean imputed value: {np.mean(new_values):.4f}")
        
        return {
            "status": "completed",
            "n_imputed": len(missing_indices),
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
