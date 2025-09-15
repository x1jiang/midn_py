#!/usr/bin/env python3

"""
SIMIRemote.py - SIMI Remote site implementation
"""

import sys
import os
import asyncio
import numpy as np
import websockets
import json
import time
import pandas as pd

async def SIRemoteLS(X, y, websocket):
    """
    Remote component of the SIMI algorithm for least squares.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    websocket : websockets.WebSocketClientProtocol
        WebSocket connection to central site
    """
    p = X.shape[1]
    n = X.shape[0]
    
    XX = np.matmul(X.T, X)
    Xy = np.matmul(X.T, y)
    yy = np.sum(y**2)
    
    # Send data to central site
    print("Sending data to central site")
    await websocket.send(json.dumps({
        'n': float(n),
        'XX': XX.tolist(),
        'Xy': Xy.tolist(),
        'yy': float(yy)
    }))
    
    print("Data sent successfully")

async def SIRemoteLogit(X, y, websocket):
    """
    Remote component of the SIMI algorithm for logistic regression.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    websocket : websockets.WebSocketClientProtocol
        WebSocket connection to central site
    """
    print("Processing with logistic method")
    p = X.shape[1]
    n = X.shape[0]
    
    print(f"Sending n = {n}")
    await websocket.send(json.dumps({
        'n': float(n)
    }))
    
    while True:
        print("Waiting for message from central site...")
        msg = await websocket.recv()
        data = json.loads(msg)
        
        if data.get('type') == 'mode':
            mode = data['mode']
            print(f"Received mode: {mode}")
            
            if mode == 0:
                print("Received termination signal (mode 0)")
                break
            else:
                print(f"Processing mode {mode}")
                beta = np.array(data['beta'])
                
                # Compute logistic values safely to avoid overflow
                xb = np.matmul(X, beta)
                
                # Use numerically stable calculation for probabilities
                pos_mask = xb > 0
                neg_mask = ~pos_mask
                pr = np.zeros_like(xb)
                pr[pos_mask] = 1 / (1 + np.exp(-xb[pos_mask]))
                pr[neg_mask] = np.exp(xb[neg_mask]) / (1 + np.exp(xb[neg_mask]))
                
                # Clip probabilities to prevent extreme values
                pr = np.clip(pr, 1e-15, 1-1e-15)
                
                # Compute log-likelihood (Q) in a numerically stable way
                # For logistic regression: sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
                Q = np.sum(y * np.log(pr) + (1-y) * np.log(1-pr))
                
                # Compute H matrix and g vector
                w = pr * (1 - pr)  # Second derivative of negative log-likelihood
                H = np.matmul(X.T * w, X)  # Hessian
                g = np.matmul(X.T, y - pr)  # Gradient
                
                # Send H, g, and Q back to the central site
                await websocket.send(json.dumps({'type': 'H', 'H': H.tolist()}))
                await websocket.send(json.dumps({'type': 'g', 'g': g.tolist()}))
                await websocket.send(json.dumps({'type': 'Q', 'Q': float(Q)}))
                print("Processing complete for this iteration")
        else:
            print(f"Unexpected message type: {data.get('type', 'unknown')}")
            break
    
    print("Logistic processing complete")

async def SIMIRemote(D, mvar, site_id, cent_host="127.0.0.1", cent_port=6000):
    """
    Remote component of the SIMI algorithm.
    
    Parameters:
    -----------
    D : numpy.ndarray
        Data matrix
    mvar : int
        Index of missing variable (0-based)
    site_id : str
        Unique identifier for this remote site
    cent_host : str, optional
        Hostname of central site (default: "127.0.0.1")
    cent_port : int, optional
        Port of central site (default: 6000)
    """
    miss = np.isnan(D[:, mvar])
    X = D[~miss, :]
    X = np.delete(X, mvar, axis=1)
    y = D[~miss, mvar]
    
    # Check if we have any valid data
    if len(X) == 0:
        raise ValueError(f"No valid data for site after filtering missing values at index {mvar}")
    
    # Keep track of n for later use
    n = X.shape[0]
    
    while True:
        try:
            # Print diagnostic info
            print(f"Connecting to central site at ws://{cent_host}:{cent_port}")
            
            # Try to establish WebSocket connection with appropriate timeouts
            async with websockets.connect(
                f"ws://{cent_host}:{cent_port}",
                ping_interval=None,
                close_timeout=5,
                open_timeout=5
            ) as websocket:
                print(f"Connection established successfully to {cent_host}:{cent_port}")
                
                # Identify this site to the central server with a unique ID
                print(f"Sending site identifier: {site_id}")
                await websocket.send(json.dumps({
                    'type': 'REMOTE_SITE',
                    'site_id': site_id
                }))
                
                # Receive method instruction from central
                print("Waiting to receive method from central site...")
                msg = await websocket.recv()
                data = json.loads(msg)
                method = data.get('method', '').lower()
                print(f"Received method: {method}")
                
                if method:
                    if method == "gaussian":
                        print("Processing with Gaussian method")
                        await SIRemoteLS(X, y, websocket)
                        print("Gaussian processing complete.")
                    elif method == "logistic":
                        print("Processing with logistic method")
                        await SIRemoteLogit(X, y, websocket)
                        print("Logistic processing complete.")
                    else:
                        print(f"WARNING: Unknown method received: {method}")
                else:
                    print("ERROR: Received empty method from central site")
                
                print("Job finished. Connection will be closed.")
                
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by the server.")
        except ConnectionRefusedError:
            print("Connection refused. Is the central server running?")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
        
        # Sleep for a short time before reconnecting
        print("Waiting 5 seconds before attempting to reconnect...")
        await asyncio.sleep(5)

async def main():
    # Parse command line arguments
    if len(sys.argv) < 7:
        print("Usage: python SIMIRemote.py <site_id> <cent_host> <cent_port> <missing_var_index> <method> <data_file>")
        print("  site_id: Identifier for this remote site (e.g., 'remote_site_1')")
        print("  cent_host: Hostname or IP address of the central site")
        print("  cent_port: Port number of the central site")
        print("  missing_var_index: 0-based index of the variable with missing values")
        print("  method: Gaussian or logistic")
        print("  data_file: Path to CSV file with the data")
        return
    
    site_id = sys.argv[1]
    cent_host = sys.argv[2]
    cent_port = int(sys.argv[3])
    mvar = int(sys.argv[4])
    method = sys.argv[5]
    data_file = sys.argv[6]
    
    print(f"=== SIMI Remote Site '{site_id}' ===")
    print(f"Connecting to central site at {cent_host}:{cent_port}")
    print(f"Missing variable index: {mvar}")
    print(f"Method: {method}")
    print(f"Data file: {data_file}")
    
    # Load data from the specified file
    print(f"Loading data from: {data_file}")
    X = pd.read_csv(data_file) 
    print(X.head(3))
    X=X.values
    print(f"Data loaded successfully. Shape: {X.shape}")
    # Run the remote site
    await SIMIRemote(X, mvar, site_id, cent_host, cent_port)
    
    print("\n=== SIMI Remote Site Completed ===")

if __name__ == "__main__":
    asyncio.run(main())
