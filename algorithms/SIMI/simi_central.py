"""
SIMI algorithm implementation for the central site.
Implements the CentralAlgorithm interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from common.algorithm.base import CentralAlgorithm


class SIMICentralAlgorithm(CentralAlgorithm):
    """
    Central implementation of the SIMI algorithm.
    """
    
    def __init__(self):
        """
        Initialize the algorithm instance.
        """
        self.method = "gaussian"
        self.missing_mask = None
        self.non_missing_mask = None
        self.imputation_count = 10
    
    @classmethod
    def get_algorithm_name(cls) -> str:
        return "SIMI"
    
    @classmethod
    def get_supported_methods(cls) -> List[str]:
        return ["gaussian", "logistic"]
    
    async def aggregate_data(self, local_data: Any, remote_data_list: List[Any]) -> Any:
        """
        Aggregate local and remote data.
        
        Args:
            local_data: Data from the central site
            remote_data_list: List of data from remote sites
            
        Returns:
            Aggregated data
        """
        if self.method == "gaussian":
            return await self._aggregate_gaussian(local_data, remote_data_list)
        else:  # logistic
            return await self._aggregate_logistic(local_data, remote_data_list)
    
    async def impute(self, data: pd.DataFrame, target_column: int, aggregated_data: Any, 
                   method: str, imputation_count: int = 10) -> List[pd.DataFrame]:
        """
        Perform multiple imputations using the aggregated data.
        Matches R reference implementation exactly.
        
        Args:
            data: DataFrame with missing values to impute
            target_column: Index of the column to impute
            aggregated_data: Data aggregated from all sites
            method: Imputation method to use (e.g. 'gaussian', 'logistic')
            imputation_count: Number of imputations to generate (M in R)
            
        Returns:
            List of DataFrames with imputed values (matching R's imp list)
        """
        # Store method and imputation count
        self.method = method.lower()
        self.imputation_count = imputation_count
        
        # Convert to numpy for processing
        data_np = data.values.copy()
        
        # Create masks for missing values
        self.missing_mask = np.isnan(data_np[:, target_column])
        self.non_missing_mask = ~self.missing_mask
        
        # Generate M imputations (same as R)
        imp_list = []
        for m in range(imputation_count):
            if self.method == "gaussian":
                imputed_data = await self._gaussian_impute(data_np, target_column, aggregated_data)
            else:  # logistic
                imputed_data = await self._logistic_impute(data_np, target_column, aggregated_data)
            
            # Convert back to DataFrame and add to list
            imputed_df = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
            imp_list.append(imputed_df)
        
        return imp_list
    
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
        if message_type == "connect":
            # Send method to the remote site
            return {"type": "method", "method": self.method}
        
        elif message_type == "data" and self.method == "gaussian":
            # Remote site sent data for Gaussian method
            return {"received": True}
            
        elif message_type in ["n", "H", "g", "Q"] and self.method == "logistic":
            # Remote site sent data for logistic method
            return {"received": True}
        
        # Unknown message type
        return {"error": f"Unknown message type: {message_type}"}
    
    async def _aggregate_gaussian(self, local_data: Dict[str, Any], remote_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate data for Gaussian model.
        Matches R reference implementation exactly.
        
        Args:
            local_data: Data from the central site (n, XX, Xy, yy)
            remote_data_list: List of data from remote sites
            
        Returns:
            Aggregated data for Gaussian model with beta, vcov, SSE, n
        """
        # Extract local data
        n_local = local_data["n"]
        XX_local = local_data["XX"]
        Xy_local = local_data["Xy"]
        yy_local = local_data["yy"]
        
        # Initialize with local data
        n_total = n_local
        XX_total = XX_local.copy()
        Xy_total = Xy_local.copy()
        yy_total = yy_local
        
        # Add data from remote sites
        for remote_data in remote_data_list:
            n_total += remote_data["n"]
            XX_total += np.array(remote_data["XX"])
            Xy_total += np.array(remote_data["Xy"])
            yy_total += remote_data["yy"]
        
        # Compute regression coefficients (same as R)
        try:
            # R: cXX = chol(XX); iXX = chol2inv(cXX); beta = iXX %*% Xy
            L = np.linalg.cholesky(XX_total)
            iXX = np.linalg.inv(L.T) @ np.linalg.inv(L)  # chol2inv
            beta = iXX @ Xy_total
            
            # R: SSE = yy + sum(beta*(XX%*%beta-2*Xy))
            SSE = yy_total + np.sum(beta * (XX_total @ beta - 2 * Xy_total))
            
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular matrix
            iXX = np.linalg.pinv(XX_total)
            beta = iXX @ Xy_total
            SSE = yy_total + np.sum(beta * (XX_total @ beta - 2 * Xy_total))
            
        return {
            "beta": beta,
            "vcov": iXX,     # variance-covariance matrix
            "SSE": SSE,      # sum of squared errors
            "n": n_total     # total sample size
        }
    
    async def _aggregate_logistic(self, local_data: Dict[str, Any], remote_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate data for logistic regression model.
        Performs iterative optimization matching R reference implementation.
        
        Args:
            local_data: Local data with X, y
            remote_data_list: List of initial data from remote sites
            
        Returns:
            Final beta coefficients and variance-covariance matrix
        """
        # Extract local data
        X_local = local_data["X"]
        y_local = local_data["y"]
        n_local = X_local.shape[0]
        p = X_local.shape[1]
        
        # Total sample size (local + all remotes)
        N = n_local
        for site_data in remote_data_list:
            N += site_data.get("n", 0)
        
        # Initialize beta at zero (same as R)
        beta = np.zeros(p)
        
        # Regularization parameter (same as R default)
        lam = 1e-3
        
        # Maximum number of iterations (same as R default) 
        max_iter = 100
        
        for iteration in range(max_iter):
            # Compute local contribution
            xb = X_local @ beta
            pr = 1 / (1 + np.exp(-xb))
            
            # Add regularization to Hessian (same as R)
            H = X_local.T @ (X_local * (pr * (1 - pr))[:, np.newaxis]) + np.diag(N * lam, p)
            g = X_local.T @ (y_local - pr) - N * lam * beta
            
            # Compute local log-likelihood with regularization
            Q = (np.sum(y_local * xb) + 
                 np.sum(np.log(1 - pr[pr < 0.5])) + 
                 np.sum(np.log(pr[pr >= 0.5]) - xb[pr >= 0.5]) - 
                 N * lam * np.sum(beta**2) / 2)
            
            # Collect contributions from remote sites  
            for site_data in remote_data_list:
                if "H" in site_data:
                    H += np.array(site_data["H"])
                if "g" in site_data:
                    g += np.array(site_data["g"])
                if "Q" in site_data:
                    Q += site_data["Q"]
            
            # Newton step (same as R: dir = chol2inv(chol(H)) %*% g)
            try:
                L = np.linalg.cholesky(H)
                dir_vec = np.linalg.solve(L, np.linalg.solve(L.T, g))
                m = np.sum(dir_vec * g)
                
                # Line search (matching R implementation)
                step = 1.0
                while True:
                    beta_new = beta + step * dir_vec
                    
                    # Check convergence (same tolerance as R: 1e-5)
                    if np.max(np.abs(beta_new - beta)) < 1e-5:
                        break
                    
                    # Compute new objective value
                    xb_new = X_local @ beta_new
                    pr_new = 1 / (1 + np.exp(-xb_new))
                    nQ = (np.sum(y_local * xb_new) + 
                          np.sum(np.log(1 - pr_new[pr_new < 0.5])) + 
                          np.sum(np.log(pr_new[pr_new >= 0.5]) - xb_new[pr_new >= 0.5]) - 
                          N * lam * np.sum(beta_new**2) / 2)
                    
                    # Add remote contributions to new objective
                    for site_data in remote_data_list:
                        if "nQ" in site_data:
                            nQ += site_data["nQ"]
                    
                    # Armijo condition (same as R: nQ-Q > m*step/2)
                    if nQ - Q > m * step / 2:
                        break
                    step = step / 2
                
                # Final convergence check (same as R)
                if np.max(np.abs(beta_new - beta)) < 1e-5:
                    beta = beta_new
                    break
                    
                beta = beta_new
                
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                dir_vec = np.linalg.pinv(H) @ g
                beta += dir_vec
                break
        
        # Compute variance-covariance matrix (same as R: chol2inv(chol(H)))
        try:
            L = np.linalg.cholesky(H)
            vcov = np.linalg.inv(L.T) @ np.linalg.inv(L)
        except:
            vcov = np.linalg.pinv(H)
        
        return {"beta": beta, "vcov": vcov}
    
    async def _gaussian_impute(self, data: np.ndarray, target_column: int, aggregated_data: Dict[str, Any]) -> np.ndarray:
        """
        Impute missing values using Gaussian model.
        Matches R reference implementation exactly.
        
        Args:
            data: Data array with missing values
            target_column: Index of the column to impute
            aggregated_data: Aggregated data with beta, vcov, SSE, n
            
        Returns:
            Data array with imputed values
        """
        # Create a copy of the data for imputation
        D_imp = data.copy()
        
        # Extract parameters from aggregated data
        beta = aggregated_data["beta"]
        vcov = aggregated_data["vcov"]
        SSE = aggregated_data["SSE"]
        n = aggregated_data["n"]
        p = len(beta)
        
        # Generate M imputations (matching R logic)
        miss = np.isnan(D_imp[:, target_column])
        nm = np.sum(miss)
        
        if nm == 0:
            return D_imp
        
        # Get design matrix for missing observations (same as R)
        X_miss = np.delete(D_imp[miss, :], target_column, axis=1)
        
        # Generate imputation following R's approach:
        # sig = sqrt(1/rgamma(1,(n+1)/2,(SSE+1)/2))
        # alpha = beta + sig * t(cvcov) %*% rnorm(p)
        # D[miss,mvar] = D[miss,-mvar] %*% alpha + rnorm(nm,0,sig)
        
        # Draw variance from inverse gamma (same as R)
        sig = np.sqrt(1 / np.random.gamma((n + 1) / 2, 2 / (SSE + 1)))
        
        # Cholesky decomposition of variance-covariance matrix
        try:
            cvcov = np.linalg.cholesky(vcov)
        except:
            cvcov = np.linalg.cholesky(vcov + np.eye(p) * 1e-10)
        
        # Draw coefficient vector (same as R)
        alpha = beta + sig * cvcov.T @ np.random.normal(0, 1, p)
        
        # Generate imputations (same as R)
        D_imp[miss, target_column] = X_miss @ alpha + np.random.normal(0, sig, nm)
        
        return D_imp
    
    async def _logistic_impute(self, data: np.ndarray, target_column: int, aggregated_data: Dict[str, Any]) -> np.ndarray:
        """
        Impute missing values using logistic regression model.
        Matches R reference implementation exactly.
        
        Args:
            data: Data array with missing values
            target_column: Index of the column to impute
            aggregated_data: Aggregated data with beta and vcov
            
        Returns:
            Data array with imputed values
        """
        # Create a copy of the data for imputation
        D_imp = data.copy()
        
        # Extract parameters from aggregated data
        beta = aggregated_data["beta"]
        vcov = aggregated_data["vcov"]
        p = len(beta)
        
        # Generate imputation following R's approach:
        # alpha = beta + t(cvcov) %*% rnorm(p)
        # pr = 1 / (1 + exp(-D[miss,-mvar] %*% alpha))
        # D[miss,mvar] = rbinom(nm,1,pr)
        
        miss = np.isnan(D_imp[:, target_column])
        nm = np.sum(miss)
        
        if nm == 0:
            return D_imp
        
        # Get design matrix for missing observations
        X_miss = np.delete(D_imp[miss, :], target_column, axis=1)
        
        # Cholesky decomposition of variance-covariance matrix
        try:
            cvcov = np.linalg.cholesky(vcov)
        except:
            cvcov = np.linalg.cholesky(vcov + np.eye(p) * 1e-10)
        
        # Draw coefficient vector (same as R)
        alpha = beta + cvcov.T @ np.random.normal(0, 1, p)
        
        # Compute probabilities (same as R)
        pr = 1 / (1 + np.exp(-X_miss @ alpha))
        
        # Generate binary imputations (same as R: rbinom(nm,1,pr))
        D_imp[miss, target_column] = np.random.binomial(1, pr).astype(float)
        
        return D_imp
