"""
SIMICE algorithm implementation for the central site.
Implements the CentralAlgorithm interface for multiple imputation with chained equations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import asyncio

from common.algorithm.base import CentralAlgorithm


class SIMICECentralAlgorithm(CentralAlgorithm):
    """
    Central implementation of the SIMICE algorithm.
    Multiple Imputation using Chained Equations for multiple target columns.
    """
    
    def __init__(self):
        """
        Initialize the algorithm instance.
        """
        self.target_columns = []
        self.is_binary = []
        self.missing_masks = {}
        self.non_missing_masks = {}
        self.iteration_before_first_imputation = 5
        self.iteration_between_imputations = 5
        self.imputation_count = 10
    
    @classmethod
    def get_algorithm_name(cls) -> str:
        return "SIMICE"
    
    @classmethod
    def get_supported_methods(cls) -> List[str]:
        return ["gaussian", "logistic"]
    
    def _initialize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize missing values with mean imputation for continuous variables
        and mode imputation for binary variables.
        """
        data_init = data.copy()
        
        for i, col_idx in enumerate(self.target_columns):
            col_name = data.columns[col_idx]
            missing_mask = pd.isna(data_init[col_name])
            
            if missing_mask.any():
                non_missing = data_init.loc[~missing_mask, col_name]
                
                if self.is_binary[i]:
                    # Mode imputation for binary variables
                    mode_value = non_missing.mode().iloc[0] if len(non_missing) > 0 else 0
                    data_init.loc[missing_mask, col_name] = mode_value
                else:
                    # Mean imputation for continuous variables
                    mean_value = non_missing.mean() if len(non_missing) > 0 else 0
                    data_init.loc[missing_mask, col_name] = mean_value
        
        return data_init
    
    async def aggregate_data(self, local_data: Any, remote_data_list: List[Any]) -> Any:
        """
        Aggregate local and remote data for a specific target column.
        
        Args:
            local_data: Data from the central site (dict with X, y, method)
            remote_data_list: List of data from remote sites
            
        Returns:
            Aggregated sufficient statistics
        """
        method = local_data.get('method', 'gaussian')
        
        if method.lower() == "gaussian":
            return await self._aggregate_gaussian(local_data, remote_data_list)
        else:  # logistic
            return await self._aggregate_logistic(local_data, remote_data_list)
    
    async def _aggregate_gaussian(self, local_data: Dict, remote_data_list: List[Dict]) -> Dict:
        """
        Aggregate data for Gaussian (continuous) imputation.
        """
        # Initialize with local data
        XTX = local_data['XTX'].copy()
        XTy = local_data['XTy'].copy()
        yTy = local_data['yTy']
        n = local_data['n']
        
        # Aggregate from all remote sites
        for remote_data in remote_data_list:
            XTX += remote_data['XTX']
            XTy += remote_data['XTy']
            yTy += remote_data['yTy']
            n += remote_data['n']
        
        # Solve for beta
        try:
            beta = np.linalg.solve(XTX, XTy)
            SSE = yTy - XTy.T @ beta
        except np.linalg.LinAlgError:
            # Handle singular matrix
            beta = np.linalg.lstsq(XTX, XTy, rcond=None)[0]
            SSE = yTy - XTy.T @ beta
        
        # Compute variance-covariance matrix
        try:
            vcov = np.linalg.inv(XTX)
        except np.linalg.LinAlgError:
            vcov = np.linalg.pinv(XTX)
        
        return {
            'beta': beta,
            'vcov': vcov,
            'SSE': SSE,
            'n': n,
            'XTX': XTX
        }
    
    async def _aggregate_logistic(self, local_data: Dict, remote_data_list: List[Dict]) -> Dict:
        """
        Aggregate data for logistic (binary) imputation.
        """
        # Start with local data
        beta = local_data['beta'].copy()
        H = local_data['H'].copy()
        
        # Simple aggregation for logistic regression (can be improved with proper distributed optimization)
        for remote_data in remote_data_list:
            H += remote_data['H']
        
        # Compute final variance-covariance matrix
        try:
            vcov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            vcov = np.linalg.pinv(H)
        
        return {
            'beta': beta,
            'vcov': vcov,
            'H': H
        }
    
    async def impute(self, data: pd.DataFrame, target_column_indexes: List[int], 
                   is_binary: List[bool], iteration_before_first_imputation: int,
                   iteration_between_imputations: int, imputation_count: int = 10) -> List[pd.DataFrame]:
        """
        Perform multiple imputations using chained equations.
        
        Args:
            data: DataFrame with missing values to impute
            target_column_indexes: List of 1-based indices of columns to impute
            is_binary: List of booleans indicating if each target column is binary
            iteration_before_first_imputation: Iterations before first imputation
            iteration_between_imputations: Iterations between imputations
            imputation_count: Number of complete imputations to generate
            
        Returns:
            List of DataFrames with imputed values
        """
        # Store parameters
        self.target_columns = [idx - 1 for idx in target_column_indexes]  # Convert to 0-based
        self.is_binary = is_binary
        self.iteration_before_first_imputation = iteration_before_first_imputation
        self.iteration_between_imputations = iteration_between_imputations
        self.imputation_count = imputation_count
        
        # Create missing value masks
        for col_idx in self.target_columns:
            col_name = data.columns[col_idx]
            self.missing_masks[col_idx] = pd.isna(data[col_name])
            self.non_missing_masks[col_idx] = ~self.missing_masks[col_idx]
        
        # Initialize data with simple imputation
        current_data = self._initialize_data(data)
        
        imp_list = []
        
        for m in range(imputation_count):
            # Set initial iterations (more before first, less between subsequent)
            iterations = self.iteration_before_first_imputation if m == 0 else self.iteration_between_imputations
            
            # Run iterations for this imputation
            for it in range(iterations):
                # Cycle through all target columns
                for i, col_idx in enumerate(self.target_columns):
                    col_name = data.columns[col_idx]
                    missing_mask = self.missing_masks[col_idx]
                    
                    if missing_mask.any():
                        # Prepare data for this column
                        predictor_cols = [c for c in range(len(data.columns)) if c != col_idx]
                        X = current_data.iloc[:, predictor_cols].values
                        # Add intercept column
                        X = np.column_stack([X, np.ones(X.shape[0])])
                        
                        # Get non-missing observations for this target
                        non_missing_idx = ~missing_mask
                        if non_missing_idx.sum() > 0:
                            X_obs = X[non_missing_idx]
                            y_obs = current_data.loc[non_missing_idx, col_name].values
                            
                            # Determine method based on variable type
                            method = "logistic" if self.is_binary[i] else "gaussian"
                            
                            # Get aggregated model from remote sites
                            local_data = await self._prepare_local_data(X_obs, y_obs, method)
                            # Note: In practice, we would collect remote_data_list from actual remote sites
                            remote_data_list = []  # Placeholder for remote site data
                            
                            aggregated_data = await self.aggregate_data(local_data, remote_data_list)
                            
                            # Generate imputed values
                            if method == "gaussian":
                                imputed_values = await self._gaussian_impute_column(
                                    X, missing_mask, aggregated_data
                                )
                            else:  # logistic
                                imputed_values = await self._logistic_impute_column(
                                    X, missing_mask, aggregated_data
                                )
                            
                            # Update current data with imputed values
                            current_data.loc[missing_mask, col_name] = imputed_values
            
            # Store this completed imputation
            imp_list.append(current_data.copy())
        
        return imp_list
    
    async def _prepare_local_data(self, X: np.ndarray, y: np.ndarray, method: str) -> Dict[str, Any]:
        """
        Prepare local data for aggregation.
        """
        if method.lower() == "gaussian":
            XTX = X.T @ X
            XTy = X.T @ y
            yTy = y.T @ y
            n = len(y)
            
            return {
                'XTX': XTX,
                'XTy': XTy,
                'yTy': yTy,
                'n': n,
                'method': method
            }
        else:  # logistic
            # Simple logistic regression setup (can be improved)
            from sklearn.linear_model import LogisticRegression
            
            try:
                model = LogisticRegression(fit_intercept=False, max_iter=1000)
                model.fit(X, y)
                beta = model.coef_.flatten()
                
                # Approximate Hessian
                p = 1 / (1 + np.exp(-X @ beta))
                W = np.diag(p * (1 - p))
                H = X.T @ W @ X
                
            except Exception:
                # Fallback
                beta = np.zeros(X.shape[1])
                H = np.eye(X.shape[1])
            
            return {
                'beta': beta,
                'H': H,
                'method': method
            }
    
    async def _gaussian_impute_column(self, X: np.ndarray, missing_mask: pd.Series, 
                                    aggregated_data: Dict) -> np.ndarray:
        """
        Generate imputed values for a continuous variable following R implementation.
        R code: Sample beta from MVN, then generate Y = X*beta + epsilon
        """
        beta = np.array(aggregated_data['beta'])
        sigma = aggregated_data.get('sigma', 1.0)
        
        print(f"🎲 SIMICE Gaussian: Using beta shape={beta.shape}, sigma={sigma:.4f}")
        
        # Generate imputed values: Y = X*beta + N(0, sigma^2)
        X_missing = X[missing_mask]
        linear_pred = X_missing @ beta
        noise = np.random.normal(0, sigma, size=len(X_missing))
        imputed_values = linear_pred + noise
        
        print(f"✨ SIMICE Gaussian: Generated {len(imputed_values)} imputed values")
        return imputed_values
    
    async def _logistic_impute_column(self, X: np.ndarray, missing_mask: pd.Series,
                                    aggregated_data: Dict) -> np.ndarray:
        """
        Generate imputed values for a binary variable following R implementation.
        R code: Use sampled beta to generate probabilities, then sample from Bernoulli
        """
        beta = np.array(aggregated_data['beta'])
        
        print(f"🎲 SIMICE Logistic: Using beta shape={beta.shape}")
        
        # Generate imputed values using sigmoid and Bernoulli sampling
        X_missing = X[missing_mask]
        linear_pred = X_missing @ beta
        
        # Apply sigmoid with numerical stability
        linear_pred_clipped = np.clip(linear_pred, -500, 500)
        probs = 1 / (1 + np.exp(-linear_pred_clipped))
        probs = np.clip(probs, 1e-8, 1 - 1e-8)  # Avoid boundary values
        
        # Sample from Bernoulli distribution
        imputed_values = np.random.binomial(1, probs)
        
        print(f"✨ SIMICE Logistic: Generated {len(imputed_values)} binary imputed values")
        return imputed_values
        
        return imputed_values
    
    async def process_message(self, site_id: str, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from a remote site.
        
        Args:
            site_id: ID of the site that sent the message
            message_type: Type of the message
            payload: Message payload
            
        Returns:
            Response payload (if any)
        """
        if message_type == "data_prepared":
            # Remote site has prepared its local data
            return {"status": "received", "next_step": "aggregate"}
        
        elif message_type == "iteration_complete":
            # Remote site completed an iteration
            return {"status": "acknowledged"}
        
        else:
            return {"error": f"Unknown message type: {message_type}"}
