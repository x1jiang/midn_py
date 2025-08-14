"""
SIMICERemote.py - SIMICE Remote site implementation
Main entry point for SIMICE algorithm on remote sites.
"""

import sys
import os
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional

# Add the parent directory to the path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from simice_remote_base import SIMICERemoteLS, SIMICERemoteLogit, SIMICERemoteHelper
except ImportError:
    # Fallback for when running from different directory
    from algorithms.SIMICE.simice_remote_base import SIMICERemoteLS, SIMICERemoteLogit, SIMICERemoteHelper


class SIMICERemote(SIMICERemoteHelper):
    """
    Main SIMICE Remote implementation matching R reference behavior.
    Handles communication with central site for multiple target variables.
    """
    
    def __init__(self, site_id: str = "remote_1"):
        super().__init__(site_id)
        self.current_iteration = 0
        
    async def run_simice_remote(self, D: np.ndarray, port: int, 
                               cent_host: str = "localhost", cent_port: int = 6000) -> None:
        """
        Main SIMICE remote algorithm implementation matching R reference.
        
        Args:
            D: Local data matrix
            port: Local listening port (not used in this simplified version)
            cent_host: Hostname of central site
            cent_port: Base port of central site
        """
        print(f"Starting SIMICE Remote (site: {self.site_id})")
        
        # Add intercept column (matching R implementation)  
        D = np.column_stack([D, np.ones(D.shape[0])])
        p = D.shape[1]
        self.current_data = D.copy()
        
        # Store original missing patterns
        miss = np.isnan(D)
        
        try:
            # Wait for initialization from central site
            # In practice, this would be done through WebSocket communication
            print("Waiting for initialization from central site...")
            
            # Main communication loop - this would be event-driven in practice
            while True:
                # This is a simplified version - actual implementation would
                # respond to messages from central site
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
                # Break after some time for demo purposes
                self.current_iteration += 1
                if self.current_iteration > 100:  # Demo timeout
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error in SIMICE remote: {e}")
        
        print(f"SIMICE Remote (site: {self.site_id}) completed")
    
    async def handle_central_message(self, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle messages from the central site.
        
        Args:
            message_type: Type of message from central
            data: Message data
            
        Returns:
            Response to send back to central
        """
        try:
            if message_type == "INITIALIZE":
                # Central site is initializing the algorithm
                mvar = data.get('mvar', [])
                self.target_columns = [idx - 1 for idx in mvar]  # Convert to 0-based
                
                # Initialize missing data
                if self.current_data is not None:
                    is_binary = data.get('is_binary', [False] * len(mvar))
                    self.current_data = self.initialize_missing_data(
                        self.current_data, self.target_columns, is_binary
                    )
                
                return {
                    "type": "INITIALIZE_RESPONSE",
                    "status": "ready",
                    "site_id": self.site_id
                }
            
            elif message_type == "REQUEST_DATA":
                # Central site is requesting data for a specific variable
                target_col = data.get('target_column', 0) - 1  # Convert to 0-based
                method = data.get('method', 'gaussian').lower()
                
                if self.current_data is not None and target_col < self.current_data.shape[1]:
                    # Get non-missing observations for this target
                    missing_mask = np.isnan(self.current_data[:, target_col])
                    non_missing_mask = ~missing_mask
                    
                    if non_missing_mask.any():
                        # Prepare predictors (exclude target column)
                        X = self.current_data[non_missing_mask, :]
                        X = np.delete(X, target_col, axis=1)
                        y = self.current_data[non_missing_mask, target_col]
                        
                        if method == "gaussian":
                            # Compute sufficient statistics for Gaussian
                            XTX = X.T @ X
                            XTy = X.T @ y
                            yTy = y.T @ y
                            n = len(y)
                            
                            return {
                                "type": "DATA_RESPONSE",
                                "statistics": {
                                    "XTX": XTX.tolist(),
                                    "XTy": XTy.tolist(), 
                                    "yTy": float(yTy),
                                    "n": int(n)
                                },
                                "method": method
                            }
                        
                        elif method == "logistic":
                            # For logistic, we need iterative communication
                            return {
                                "type": "LOGISTIC_READY",
                                "n_obs": int(len(y)),
                                "method": method
                            }
                
                # Fallback response
                return {
                    "type": "NO_DATA",
                    "message": "No valid data for requested variable"
                }
            
            elif message_type == "REQUEST_GRADIENT":
                # For logistic regression iteration
                target_col = data.get('target_column', 0) - 1
                beta = np.array(data.get('beta', []))
                
                if (self.current_data is not None and 
                    target_col < self.current_data.shape[1] and
                    len(beta) > 0):
                    
                    missing_mask = np.isnan(self.current_data[:, target_col])
                    non_missing_mask = ~missing_mask
                    
                    if non_missing_mask.any():
                        X = self.current_data[non_missing_mask, :]
                        X = np.delete(X, target_col, axis=1)
                        y = self.current_data[non_missing_mask, target_col]
                        
                        # Compute gradient and Hessian
                        linear_pred = X @ beta
                        linear_pred = np.clip(linear_pred, -500, 500)
                        probs = 1 / (1 + np.exp(-linear_pred))
                        probs = self._clip_probabilities(probs)
                        
                        W = probs * (1 - probs)
                        H = X.T @ np.diag(W) @ X
                        gradient = X.T @ (y - probs)
                        
                        return {
                            "type": "GRADIENT_RESPONSE",
                            "H": H.tolist(),
                            "gradient": gradient.tolist()
                        }
                
                # Fallback
                return {
                    "type": "GRADIENT_ERROR",
                    "message": "Could not compute gradient"
                }
            
            elif message_type == "UPDATE_IMPUTATIONS":
                # Central site is sending new imputed values
                target_col = data.get('target_column', 0) - 1
                imputed_values = np.array(data.get('imputed_values', []))
                
                if (self.current_data is not None and 
                    target_col < self.current_data.shape[1] and
                    len(imputed_values) > 0):
                    
                    missing_mask = np.isnan(self.current_data[:, target_col])
                    if np.sum(missing_mask) == len(imputed_values):
                        self.current_data[missing_mask, target_col] = imputed_values
                        
                        return {
                            "type": "UPDATE_SUCCESS",
                            "updated_count": len(imputed_values)
                        }
                
                return {
                    "type": "UPDATE_ERROR", 
                    "message": "Could not update imputations"
                }
            
            else:
                return {
                    "type": "UNKNOWN_MESSAGE",
                    "message": f"Unknown message type: {message_type}"
                }
                
        except Exception as e:
            return {
                "type": "ERROR",
                "message": f"Error processing message: {str(e)}"
            }


async def main():
    """
    Example usage of SIMICE Remote algorithm.
    """
    # Example parameters
    np.random.seed(123)
    
    # Create sample data with missing values (should match central site data structure)
    n, p = 50, 5
    data = np.random.randn(n, p)
    
    # Introduce missing values in columns 1 and 3 (1-based: 2 and 4)
    missing_prob = 0.2
    data[np.random.random((n,)) < missing_prob, 1] = np.nan
    data[np.random.random((n,)) < missing_prob, 3] = np.nan
    
    # Make column 4 binary
    data[:, 3] = np.where(data[:, 3] > 0, 1, 0)
    
    # Remote site parameters
    site_id = "remote_1"
    local_port = 6001
    central_host = "localhost"  
    central_port = 6000
    
    # Run SIMICE Remote
    remote = SIMICERemote(site_id)
    try:
        await remote.run_simice_remote(data, local_port, central_host, central_port)
        print("SIMICE Remote completed successfully")
        
    except Exception as e:
        print(f"Error running SIMICE Remote: {e}")


if __name__ == "__main__":
    asyncio.run(main())
