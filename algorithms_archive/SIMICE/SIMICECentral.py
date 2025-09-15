"""
SIMICECentral.py - SIMICE Central site implementation
Main entry point for SIMICE algorithm on central site.
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

# Add the parent directory to the path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from simice_central_base import SIMICECentralLS, SIMICECentralLogit, SIMICEBase
except ImportError:
    # Fallback for when running from different directory
    from algorithms.SIMICE.simice_central_base import SIMICECentralLS, SIMICECentralLogit, SIMICEBase


class SIMICECentral(SIMICEBase):
    """
    Main SIMICE Central implementation matching R reference behavior.
    Implements Multiple Imputation using Chained Equations.
    """
    
    def __init__(self):
        super().__init__()
        self.connected_sites = {}
        self.data = None
        
    async def run_simice(self, D: np.ndarray, M: int, mvar: List[int], 
                        type_list: List[str], iter_val: int, iter0: int,
                        hosts: List[str], ports: List[int], cent_ports: List[int]) -> List[np.ndarray]:
        """
        Main SIMICE algorithm implementation matching R reference.
        
        Args:
            D: Data matrix
            M: Number of imputations
            mvar: List of indices of missing variables (1-based)
            type_list: List of "Gaussian" or "logistic" for each missing variable
            iter_val: Number of iterations between extracted imputed data
            iter0: Number of iterations before first extracted imputed data
            hosts: List of hostnames of remote sites
            ports: List of ports of remote sites  
            cent_ports: List of local listening ports for remote sites
            
        Returns:
            List of imputed data matrices
        """
        print(f"Starting SIMICE Central with {len(mvar)} target variables")
        
        # Add intercept column (matching R implementation)
        D = np.column_stack([D, np.ones(D.shape[0])])
        p = D.shape[1]
        n = D.shape[0]
        miss = np.isnan(D)
        l = len(mvar)
        
        # Convert to 0-based indexing for internal use
        mvar_0based = [idx - 1 for idx in mvar]
        
        # Initialize missing values with mean imputation
        for i, j in enumerate(mvar_0based):
            missing_idx = miss[:, j]
            non_missing_idx = ~missing_idx
            
            if non_missing_idx.any():
                if type_list[i].lower() == "gaussian":
                    # Mean imputation for continuous
                    D[missing_idx, j] = np.mean(D[non_missing_idx, j])
                else:
                    # Mode imputation for binary
                    values = D[non_missing_idx, j]
                    unique_vals, counts = np.unique(values, return_counts=True)
                    D[missing_idx, j] = unique_vals[np.argmax(counts)]
        
        self.data = D.copy()
        
        # Initialize connections (placeholder - actual implementation would set up WebSocket connections)
        K = len(hosts)
        print(f"Setting up connections to {K} remote sites")
        
        imp_list = []
        
        for m in range(M):
            print(f"Starting imputation {m + 1}/{M}")
            
            # Determine number of iterations for this imputation
            iterations = iter0 if m == 0 else iter_val
            
            # Run MICE iterations
            for it in range(iterations):
                print(f"  Iteration {it + 1}/{iterations}")
                
                # Cycle through all target variables
                for i in range(l):
                    j = mvar_0based[i]
                    missing_idx = miss[:, j]
                    non_missing_idx = ~missing_idx
                    nmidx = np.sum(missing_idx)
                    
                    if nmidx > 0:
                        print(f"    Imputing variable {mvar[i]} ({type_list[i]})")
                        
                        if type_list[i].lower() == "gaussian":
                            # Gaussian imputation
                            X = self.data[non_missing_idx, :]
                            X = np.delete(X, j, axis=1)  # Remove target column
                            y = self.data[non_missing_idx, j]
                            
                            # Get aggregated results from central LS function
                            # Using port offset for each variable to avoid conflicts
                            port = cent_ports[0] + j
                            try:
                                fit_imp = await SIMICECentralLS(
                                    X, y, 
                                    expected_sites=K,
                                    expected_site_names=[f"site_{k}" for k in range(K)],
                                    port=port
                                )
                                
                                # Generate imputed values (matching R implementation)
                                sig = np.sqrt(1 / np.random.gamma((fit_imp['n'] + 1) / 2, 2 / (fit_imp['SSE'] + 1)))
                                
                                # Sample coefficients
                                L = self._safe_cholesky(fit_imp['vcov'])
                                alpha = fit_imp['beta'] + sig * L @ np.random.normal(size=len(fit_imp['beta']))
                                
                                # Generate imputations
                                X_missing = self.data[missing_idx, :]
                                X_missing = np.delete(X_missing, j, axis=1)
                                self.data[missing_idx, j] = X_missing @ alpha + np.random.normal(0, sig, size=nmidx)
                                
                            except Exception as e:
                                print(f"    Error in Gaussian imputation for variable {j}: {e}")
                                # Keep existing values
                        
                        elif type_list[i].lower() == "logistic":
                            # Logistic imputation
                            X = self.data[non_missing_idx, :]
                            X = np.delete(X, j, axis=1)
                            y = self.data[non_missing_idx, j]
                            
                            # Get aggregated results from central logistic function  
                            port = cent_ports[0] + j + 100  # Offset to avoid conflicts
                            try:
                                fit_imp = await SIMICECentralLogit(
                                    X, y,
                                    expected_sites=K,
                                    expected_site_names=[f"site_{k}" for k in range(K)],
                                    port=port
                                )
                                
                                # Generate imputed values (matching R implementation)
                                L = self._safe_cholesky(fit_imp['vcov'])
                                alpha = fit_imp['beta'] + L @ np.random.normal(size=len(fit_imp['beta']))
                                
                                # Generate binary imputations
                                X_missing = self.data[missing_idx, :]
                                X_missing = np.delete(X_missing, j, axis=1)
                                logits = X_missing @ alpha
                                probs = 1 / (1 + np.exp(-logits))
                                probs = self._clip_probabilities(probs)
                                self.data[missing_idx, j] = np.random.binomial(1, probs)
                                
                            except Exception as e:
                                print(f"    Error in logistic imputation for variable {j}: {e}")
                                # Keep existing values
            
            # Store completed imputation (remove intercept column to match input)
            imp_data = self.data[:, :-1].copy()  # Remove last column (intercept)
            imp_list.append(imp_data)
            print(f"Completed imputation {m + 1}")
        
        print("SIMICE Central completed successfully")
        return imp_list


async def main():
    """
    Example usage of SIMICE Central algorithm.
    """
    # Example parameters
    np.random.seed(42)
    
    # Create sample data with missing values
    n, p = 100, 5
    data = np.random.randn(n, p)
    
    # Introduce missing values in columns 1 and 3 (1-based: 2 and 4)
    missing_prob = 0.2
    data[np.random.random((n,)) < missing_prob, 1] = np.nan  # Column 2
    data[np.random.random((n,)) < missing_prob, 3] = np.nan  # Column 4
    
    # Make column 4 binary
    data[:, 3] = np.where(data[:, 3] > 0, 1, 0)
    
    # SIMICE parameters
    M = 5  # Number of imputations
    mvar = [2, 4]  # 1-based indices of missing variables
    type_list = ["Gaussian", "logistic"]  # Types for each variable
    iter_val = 3  # Iterations between imputations
    iter0 = 5  # Iterations before first imputation
    
    # Remote site parameters (for demonstration)
    hosts = ["localhost"]
    ports = [6001]
    cent_ports = [6000]
    
    # Run SIMICE
    central = SIMICECentral()
    try:
        imp_list = await central.run_simice(
            data, M, mvar, type_list, iter_val, iter0,
            hosts, ports, cent_ports
        )
        
        print(f"Generated {len(imp_list)} imputations")
        for i, imp_data in enumerate(imp_list):
            print(f"Imputation {i + 1}: shape {imp_data.shape}")
            
    except Exception as e:
        print(f"Error running SIMICE: {e}")


if __name__ == "__main__":
    asyncio.run(main())
