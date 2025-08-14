"""
SIMICE remote base functionality.
Provides WebSocket-based communication for remote sites in SIMICE algorithm.
"""

import numpy as np
import asyncio
import websockets
import json
from typing import Dict, Any, List, Optional
from simice_central_base import SIMICEBase


async def SIMICERemoteLS(X: np.ndarray, y: np.ndarray, central_host: str = "localhost",
                        central_port: int = 6000, site_id: str = "remote_1") -> None:
    """
    Remote component of SIMICE algorithm for least squares (Gaussian) imputation.
    
    Args:
        X: Local predictor matrix
        y: Local target variable  
        central_host: Central site hostname
        central_port: Central site port
        site_id: Identifier for this remote site
    """
    print(f"Starting SIMICERemoteLS for site {site_id}")
    
    try:
        # Connect to central site
        uri = f"ws://{central_host}:{central_port}"
        async with websockets.connect(uri) as websocket:
            print(f"Connected to central site at {uri}")
            
            # Send identification
            await websocket.send(json.dumps({
                'type': 'REMOTE_SITE',
                'site_id': site_id
            }))
            
            # Wait for data request
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data['type'] == 'REQUEST_DATA':
                        method = data.get('method', 'gaussian')
                        
                        if method == 'gaussian':
                            # Compute local sufficient statistics
                            XTX = X.T @ X
                            XTy = X.T @ y  
                            yTy = y.T @ y
                            n = X.shape[0]
                            
                            # Send statistics to central
                            await websocket.send(json.dumps({
                                'type': 'DATA_RESPONSE',
                                'statistics': {
                                    'XTX': XTX.tolist(),
                                    'XTy': XTy.tolist(),
                                    'yTy': float(yTy),
                                    'n': int(n)
                                }
                            }))
                            
                            print(f"Sent Gaussian statistics to central site")
                            break
                            
                    else:
                        print(f"Unknown message type: {data['type']}")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("Connection to central site closed")
                    break
                except Exception as e:
                    print(f"Error in remote LS communication: {e}")
                    break
                    
    except Exception as e:
        print(f"Failed to connect to central site: {e}")


async def SIMICERemoteLogit(X: np.ndarray, y: np.ndarray, central_host: str = "localhost", 
                          central_port: int = 6000, site_id: str = "remote_1") -> None:
    """
    Remote component of SIMICE algorithm for logistic imputation.
    
    Args:
        X: Local predictor matrix
        y: Local binary target variable
        central_host: Central site hostname  
        central_port: Central site port
        site_id: Identifier for this remote site
    """
    print(f"Starting SIMICERemoteLogit for site {site_id}")
    
    base = SIMICEBase()
    
    try:
        # Connect to central site
        uri = f"ws://{central_host}:{central_port}"
        async with websockets.connect(uri) as websocket:
            print(f"Connected to central site at {uri}")
            
            # Send identification
            await websocket.send(json.dumps({
                'type': 'REMOTE_SITE', 
                'site_id': site_id
            }))
            
            # Process messages from central
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data['type'] == 'REQUEST_GRADIENT':
                        beta = np.array(data['beta'])
                        iteration = data['iteration']
                        
                        # Compute local gradient and Hessian
                        linear_pred = X @ beta
                        linear_pred = np.clip(linear_pred, -500, 500)
                        probs = 1 / (1 + np.exp(-linear_pred))
                        probs = base._clip_probabilities(probs)
                        
                        W = probs * (1 - probs)
                        H = X.T @ np.diag(W) @ X
                        gradient = X.T @ (y - probs)
                        
                        # Send to central
                        await websocket.send(json.dumps({
                            'type': 'GRADIENT_RESPONSE',
                            'H': H.tolist(),
                            'gradient': gradient.tolist(),
                            'iteration': iteration
                        }))
                        
                        print(f"Sent gradient for iteration {iteration}")
                        
                    elif data['type'] == 'FINAL_HESSIAN':
                        beta = np.array(data['beta'])
                        
                        # Compute final local Hessian
                        linear_pred = X @ beta
                        linear_pred = np.clip(linear_pred, -500, 500)
                        probs = 1 / (1 + np.exp(-linear_pred))
                        probs = base._clip_probabilities(probs)
                        
                        W = probs * (1 - probs)
                        H = X.T @ np.diag(W) @ X
                        
                        # Send to central
                        await websocket.send(json.dumps({
                            'type': 'HESSIAN_RESPONSE',
                            'H': H.tolist()
                        }))
                        
                        print("Sent final Hessian")
                        break
                        
                    else:
                        print(f"Unknown message type: {data['type']}")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("Connection to central site closed")
                    break
                except Exception as e:
                    print(f"Error in remote logit communication: {e}")
                    break
                    
    except Exception as e:
        print(f"Failed to connect to central site: {e}")


class SIMICERemoteHelper(SIMICEBase):
    """
    Helper class for remote SIMICE operations.
    """
    
    def __init__(self, site_id: str = "remote_1"):
        super().__init__()
        self.site_id = site_id
        self.current_data = None
        self.original_missing_masks = {}
    
    async def process_imputation_iteration(self, data: np.ndarray, target_col_idx: int,
                                         method: str, central_host: str = "localhost", 
                                         central_port: int = 6000) -> np.ndarray:
        """
        Process one iteration of imputation for a specific target column.
        
        Args:
            data: Current data matrix
            target_col_idx: 0-based index of target column
            method: "gaussian" or "logistic"
            central_host: Central site hostname
            central_port: Central site port
            
        Returns:
            Updated data matrix with new imputations
        """
        # Get non-missing observations for this target
        missing_mask = np.isnan(data[:, target_col_idx])
        non_missing_mask = ~missing_mask
        
        if not non_missing_mask.any():
            return data  # No observations to learn from
        
        # Prepare predictors
        X = self._prepare_predictor_matrix(data[non_missing_mask], target_col_idx, True)
        y = data[non_missing_mask, target_col_idx]
        
        # Communicate with central site based on method
        if method.lower() == "gaussian":
            await SIMICERemoteLS(X, y, central_host, central_port + target_col_idx, 
                               f"{self.site_id}_col_{target_col_idx}")
        else:
            await SIMICERemoteLogit(X, y, central_host, central_port + target_col_idx,
                                  f"{self.site_id}_col_{target_col_idx}")
        
        return data
    
    def initialize_missing_data(self, data: np.ndarray, target_columns: List[int], 
                              is_binary: List[bool]) -> np.ndarray:
        """
        Initialize missing values with simple imputation.
        
        Args:
            data: Original data matrix with missing values
            target_columns: List of 0-based target column indices
            is_binary: List of boolean flags for each target column
            
        Returns:
            Data matrix with initialized values
        """
        data_init = data.copy()
        
        for i, col_idx in enumerate(target_columns):
            missing_mask = np.isnan(data_init[:, col_idx])
            
            if missing_mask.any():
                non_missing_values = data_init[~missing_mask, col_idx]
                
                if len(non_missing_values) > 0:
                    if is_binary[i]:
                        # Mode for binary
                        unique_vals, counts = np.unique(non_missing_values, return_counts=True)
                        init_value = unique_vals[np.argmax(counts)]
                    else:
                        # Mean for continuous
                        init_value = np.mean(non_missing_values)
                    
                    data_init[missing_mask, col_idx] = init_value
                else:
                    # Default if no non-missing values
                    data_init[missing_mask, col_idx] = 0 if is_binary[i] else 0.0
        
        return data_init
