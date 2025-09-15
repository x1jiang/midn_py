"""
SIMI algorithm implementation for the remote site.
Implements the RemoteAlgorithm interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from common.algorithm.base import RemoteAlgorithm


class SIMIRemoteAlgorithm(RemoteAlgorithm):
    """
    Remote implementation of the SIMI algorithm.
    """
    
    def __init__(self):
        """
        Initialize the algorithm instance.
        """
        self.X = None
        self.y = None
        self.method = "logistic"  # Default to logistic method
        print(f"SIMIRemoteAlgorithm initialized with method: {self.method}")
    
    @classmethod
    def get_algorithm_name(cls) -> str:
        return "SIMI"
    
    @classmethod
    def get_supported_methods(cls) -> List[str]:
        return ["gaussian", "logistic"]
    
    async def prepare_data(self, data: np.ndarray, target_column: int) -> Dict[str, Any]:
        """
        Prepare data for initial transmission to central.
        
        Args:
            data: Data array
            target_column: Index of the target column
            
        Returns:
            Data to send to the central site
        """
        # Extract non-missing rows and separate features from target
        miss = np.isnan(data[:, target_column])
        X = data[~miss, :]
        X = np.delete(X, target_column, axis=1)
        y = data[~miss, target_column]
        
        # Store for later use in methods
        self.X = X
        self.y = y
        
        # Return initial data with sample size and number of features
        # This ensures the central site has the feature count for either method
        return {
            "n": float(len(X)),
            "p": int(X.shape[1])  # Number of features - critical for logistic regression
        }
    
    async def process_message(self, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from the central site.
        
        Args:
            message_type: Type of the message
            payload: Message payload
            
        Returns:
            Response payload (if any)
        """
        print(f"SIMIRemoteAlgorithm.process_message: type={message_type}, method={self.method}")
        
        if message_type == "method":
            # Update method based on the payload
            self.method = payload.get("method", self.method).lower()
            print(f"Method updated to: {self.method}")
            
            # Return stats based on method
            if self.method == "gaussian":
                return await self._compute_gaussian_stats()
            else:
                # For logistic, return both the sample size and feature count
                return {
                    "n": float(self.X.shape[0]),
                    "p": int(self.X.shape[1])  # Number of features - critical for logistic regression
                }
        
        elif message_type == "mode" and self.method == "logistic":
            # Process logistic regression iterations
            return await self._compute_logistic_iteration(payload)
            
        # Default empty response
        return {}
    
    async def _compute_gaussian_stats(self) -> Dict[str, Any]:
        """
        Compute sufficient statistics for Gaussian model.
        
        Returns:
            Dictionary with n, XX, Xy, yy
        """
        n = self.X.shape[0]
        XX = np.matmul(self.X.T, self.X)
        Xy = np.matmul(self.X.T, self.y)
        yy = float(np.sum(self.y ** 2))
        
        return {
            "n": float(n),
            "XX": XX.tolist(),
            "Xy": Xy.tolist(),
            "yy": float(yy)
        }
    
    async def _compute_logistic_iteration(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute one iteration of logistic regression.
        Matches R reference implementation exactly.
        
        Args:
            payload: Message payload with mode and beta
            
        Returns:
            Dictionary with H, g, Q values (mode 1) or just Q (mode 2)
        """
        mode = payload.get("mode", 0)
        
        # Mode 0 means termination
        if mode == 0:
            return {"terminated": True}
        
        # Get beta from payload
        beta = np.array(payload.get("beta", []))
        
        # Compute probabilities (same as R)
        xb = self.X @ beta
        pr = 1 / (1 + np.exp(-xb))
        
        # Compute log-likelihood (same as R)
        # Q = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5])
        Q = (np.sum(self.y * xb) + 
             np.sum(np.log(1 - pr[pr < 0.5])) + 
             np.sum(np.log(pr[pr >= 0.5]) - xb[pr >= 0.5]))
        
        if mode == 1:
            # Mode 1: return H, g, Q (same as R)
            H = self.X.T @ (self.X * (pr * (1 - pr))[:, np.newaxis])
            g = self.X.T @ (self.y - pr)
            
            return {
                "H": H.tolist(),
                "g": g.tolist(),
                "Q": float(Q)
            }
        elif mode == 2:
            # Mode 2: return only Q for line search (same as R)
            return {
                "nQ": float(Q)  # New Q value for line search
            }
        
        # Unknown mode
        return {"error": f"Unknown mode: {mode}"}
