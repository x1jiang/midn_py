"""
Python implementation of SIMIRemote.R using the new JSON-only WebSocket protocol

Original R function description:
SIMIRemote = function(D,mvar,port,cent_host,cent_port)

Arguments:
D: Data matrix
mvar: Index of missing variable
port: local listening port
cent_host: hostname of central site
cent_port: port of central site

Original R code for reference:
---------------------------------
SIMIRemote = function(D,mvar,port,cent_host,cent_port)
{
  miss = is.na(D[,mvar])
  X = D[!miss,-mvar]
  y = D[!miss,mvar]
  
  while (TRUE)
  {
    rcon <- socketConnection(host="localhost",port=port,blocking=TRUE,server=TRUE,open="w+b",timeout=60*10)
    Sys.sleep(0.1)
    method = readBin(rcon,character())
    
    wcon <- socketConnection(cent_host,cent_port,open="w+b")
    
    if ( method == "Gaussian" )
      SIRemoteLS(X,y,wcon)
    else if ( method == "logistic" )
      SIRemoteLogit(X,y,rcon,wcon)
    
    close(rcon)
    close(wcon)
  }
}
---------------------------------
"""
import numpy as np
import pandas as pd
import asyncio
import argparse
import uvicorn
import websockets
from scipy.special import expit
from fastapi import FastAPI, WebSocket
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector, 
    read_string, write_string, read_integer, write_integer, 
    WebSocketWrapper, get_wrapped_websocket
)

import numpy as np
import asyncio
from scipy.special import expit
from fastapi import FastAPI, WebSocket
import uvicorn
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector, 
    read_string, write_string, read_integer, write_integer, 
    WebSocketWrapper, get_wrapped_websocket
)

# NOTE: Key difference between R and Python implementations:
# -------------------------------------------------------------
# 1. R uses two separate socket connections:
#    - rcon: A server socket that listens for methods from central
#    - wcon: A client socket that connects to the central site
#
# 2. Python uses a single WebSocket connection that is bidirectional:
#    - We connect to the central WebSocket server
#    - We use the same connection for both receiving methods and sending data
#
# This approach is more modern but requires adapting the WebSocket interface
# to match what our transfer functions expect
# -------------------------------------------------------------

# Using the unified WebSocketWrapper from the transfer.py module

def create_remote_app(D: np.ndarray, mvar: int, central_url: str, site_id: str):
    """
    Create a FastAPI app for the remote site
    
    Arguments:
    D: Data matrix
    mvar: Index of missing variable
    central_url: URL of the central server
    site_id: ID of this remote site
    
    Note: This function has no direct equivalent in the R code. The R implementation
    uses simple blocking sockets in a loop, while the Python version:
    1. Creates a FastAPI web application
    2. Runs a background task to connect to the central server
    3. Uses async/await for non-blocking communication
    4. Provides a /status endpoint to check connection status
    """
    # Filter out missing values and prepare data
    miss = np.isnan(D[:, mvar])
    X = np.delete(D[~miss], mvar, axis=1)
    y = D[~miss, mvar]
    
    # Create FastAPI app
    app = FastAPI()
    
    # Store connection status
    connection_status = {"connected": False}
    
    @app.on_event("startup")
    async def startup_event():
        # Start background task to connect to central
        asyncio.create_task(connect_to_central(central_url, site_id, X, y, connection_status))
    
    async def connect_to_central(central_url: str, site_id: str, X: np.ndarray, y: np.ndarray, status: dict):
        """Background task to connect to central server using the new JSON-only protocol"""
        url = f"{central_url}/{site_id}"
        print(f"Connecting to central server at {url}")
        
        # Try to connect until successful
        while True:
            try:
                # Connect using websockets library
                async with websockets.connect(url) as ws_conn:
                    print(f"Connected to central server as site {site_id}")
                    status["connected"] = True
                    
                    # Wrap the websockets connection with our JSON-only wrapper
                    websocket = get_wrapped_websocket(ws_conn)
                    
                    # Process messages
                    while True:
                        try:
                            # Wait for method from central with timeout
                            method = await asyncio.wait_for(read_string(websocket), timeout=60.0)
                            print(f"Received method: {method}")
                            
                            if method == "Gaussian":
                                print("Processing Gaussian method request")
                                await si_remote_ls(X, y, websocket)
                                print("Gaussian method completed")
                            elif method == "logistic":
                                print("Processing logistic method request")
                                await si_remote_logit(X, y, websocket)
                                print("Logistic method completed")
                            elif method == "ping":
                                # Simple ping-pong for connection keep-alive
                                try:
                                    await write_string("pong", websocket)
                                    print("Ping-pong exchange completed")
                                except Exception as e:
                                    # If we can't respond to a ping, the connection is likely broken
                                    print(f"Failed to respond to ping: {e}")
                                    raise ConnectionClosed(1000, "Failed to respond to ping")
                            elif method == "heartbeat":
                                # Acknowledge the heartbeat with a response
                                await write_string("ack", websocket)
                                print("Heartbeat acknowledged")
                            else:
                                print(f"Unknown method: {method}")
                                # Send error response
                                await write_string(f"ERROR: Unknown method {method}", websocket)
                        except asyncio.TimeoutError:
                            # No message received in a while, just loop again
                            continue
                        except websockets.exceptions.ConnectionClosedError:
                            print("Connection closed while processing messages. Will attempt to reconnect.")
                            break
                        except Exception as e:
                            print(f"Error processing message: {type(e).__name__}: {e}")
                            # For non-connection errors, we can try to continue
                            # But if it's likely a connection issue, break the loop
                            if "connection" in str(e).lower() or "closed" in str(e).lower() or "send" in str(e).lower():
                                print("Breaking connection loop due to likely connection issue")
                                break
                            
            except websockets.exceptions.InvalidStatusCode as e:
                status["connected"] = False
                print(f"Invalid status code: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            
            except ConnectionClosed:
                status["connected"] = False
                print("Connection closed. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
                
            except websockets.exceptions.ConnectionClosedError as e:
                status["connected"] = False
                if e.code == 1000:  # Normal closure
                    print("Central server closed connection normally. Reconnecting in 10 seconds...")
                    await asyncio.sleep(10)
                else:
                    print(f"Connection closed with error: {e}. Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                    
            except websockets.exceptions.ConnectionClosedOK:
                status["connected"] = False
                print("Connection closed cleanly. Reconnecting in 10 seconds...")
                await asyncio.sleep(10)
                
            except Exception as e:
                status["connected"] = False
                print(f"Connection error: {type(e).__name__}: {e}")
                print("Attempting to reconnect in 5 seconds...")
                await asyncio.sleep(5)
    
    # Add status endpoint
    @app.get("/status")
    async def get_status():
        return {"connected": connection_status["connected"]}
    
    return app

async def si_remote_ls(X: np.ndarray, y: np.ndarray, websocket):
    """
    Python implementation of SIRemoteLS using JSON-only WebSocket communication
    
    R equivalent:
    SIRemoteLS = function(X,y,wcon)
    {
      p = ncol(X)
      n = nrow(X)
      XX = t(X)%*%X
      Xy = drop(t(X)%*%y)
      yy = sum(y^2)
      
      writeVec(n,wcon)
      writeMat(XX,wcon)
      writeVec(Xy,wcon)
      writeVec(yy,wcon)
    }
    """
    p = X.shape[1]
    n = X.shape[0]
    
    # Ensure the arrays don't have NaN or Inf values
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    yy = np.sum(y**2)
    
    # Validate data before sending
    if np.isnan(XX).any() or np.isinf(XX).any():
        print("Warning: XX contains NaN or Inf values, replacing with zeros")
        XX = np.nan_to_num(XX)
    if np.isnan(Xy).any() or np.isinf(Xy).any():
        print("Warning: Xy contains NaN or Inf values, replacing with zeros")
        Xy = np.nan_to_num(Xy)
    
    # Send data back to central using the JSON protocol
    await write_vector(np.array([float(n)]), websocket)  # Send as float for consistency
    await write_matrix(XX, websocket)  # Matrix is automatically sent as JSON with base64 payload
    await write_vector(Xy, websocket)  # Vector is sent as JSON with base64 payload
    await write_vector(np.array([float(yy)]), websocket)  # Scalar sent as 1-element vector
    
    print(f"Sent data to central: n={n}, XX shape={XX.shape}, Xy shape={Xy.shape}, yy={yy}")

async def si_remote_logit(X: np.ndarray, y: np.ndarray, websocket):
    """
    Python implementation of SIRemoteLogit using JSON-only WebSocket communication.

    Changes:
      - HIGHLIGHT: mode == 0 is now "CSL offset request" (returns offset only).
      - HIGHLIGHT: mode == -1 terminates the SI loop.
    R references: CSL expects (N, offset) from each site; SI uses modes 1/2 for H/g/Q and Q respectively.
    """

    p = X.shape[1]
    n = X.shape[0]

    # Send sample size using the JSON protocol
    await write_vector(np.array([float(n)]), websocket)  # Send as float for consistency
    print(f"Sent sample size: n={n}")

    while True:
        # Get mode from central using the JSON protocol
        mode = await read_integer(websocket)
        print(f"Received mode: {mode}")

        # HIGHLIGHT: new termination code for SI loop
        if mode == -1:
            print("Received termination signal")
            break

        # HIGHLIGHT: CSL branch — return offset only (no Q/H/g)
        if mode == 0:
            # Read beta used to compute site-specific offset
            beta = await read_vector(websocket)
            print(f"Received beta vector (CSL) of length {len(beta)}")

            # Calculate predictions
            xb = np.dot(X, beta)
            pr = expit(xb)  # 1/(1+exp(-xb))

            # Compute per-site offset: X^T (y - pr) / n
            offset = np.dot(X.T, (y - pr)) / n

            # Defensive cleanup if needed
            if np.isnan(offset).any() or np.isinf(offset).any():
                print("Warning: offset contains NaN/Inf; replacing with zeros")
                offset = np.nan_to_num(offset)

            await write_vector(offset.astype(float), websocket)
            print(f"Sent CSL offset vector of length {len(offset)}")

            return  # HIGHLIGHT: end CSL request here (no Q/H/g)

        # --- SI branch (modes 1 / 2) ---
        # Read beta using the JSON protocol
        beta = await read_vector(websocket)
        print(f"Received beta vector of length {len(beta)}")

        # Calculate predictions
        xb = np.dot(X, beta)
        pr = expit(xb)  # 1/(1+exp(-xb))

        # Calculate Q with safety checks
        low_pr_mask = pr < 0.5
        high_pr_mask = ~low_pr_mask

        Q = np.sum(y * xb)

        # Safely compute log(1-pr) for pr<0.5
        if np.any(low_pr_mask):
            log_vals = np.log(np.maximum(1e-10, 1 - pr[low_pr_mask]))  # Avoid log(0)
            Q += np.sum(log_vals)

        # Safely compute log(pr) for pr>=0.5
        if np.any(high_pr_mask):
            log_vals = np.log(np.maximum(1e-10, pr[high_pr_mask]))  # Avoid log(0)
            Q += np.sum(log_vals - xb[high_pr_mask])

        if mode == 1:
            # Calculate Hessian
            weights = pr * (1 - pr)
            H = np.dot(X.T * weights, X)

            # Check for NaN/Inf in H
            if np.isnan(H).any() or np.isinf(H).any():
                print("Warning: H contains NaN or Inf values, replacing with zeros")
                H = np.nan_to_num(H)

            await write_matrix(H, websocket)
            print(f"Sent Hessian matrix of shape {H.shape}")

            # Calculate gradient
            g = np.dot(X.T, (y - pr))

            # Check for NaN/Inf in g
            if np.isnan(g).any() or np.isinf(g).any():
                print("Warning: g contains NaN or Inf values, replacing with zeros")
                g = np.nan_to_num(g)

            await write_vector(g, websocket)
            print(f"Sent gradient vector of length {len(g)}")

        # Send Q using the JSON protocol (SI only; CSL returned early above)
        await write_vector(np.array([float(Q)]), websocket)  # Send as float for consistency
        print(f"Sent Q value: {Q}")

# Run the remote client
def run_remote_client(data, central_host, central_port, site_id, remote_port=None, config=None):
    """
    Run the remote client
    
    Arguments:
    data: Data matrix as numpy array or path to data file
    central_host: Hostname of central server
    central_port: Port of central server
    site_id: ID of this remote site
    remote_port: Port for the remote FastAPI server (default: 8000 + site number)
    config: Configuration dictionary containing parameters (e.g., mvar)
    """
    import pandas as pd
    import uvicorn
    
    # Check if data is a string (file path) or already a numpy array
    if isinstance(data, str):
        D = pd.read_csv(data).values
    else:
        D = data
    
    # Extract mvar from config if provided, otherwise use default
    mvar = config.get("mvar", 1) if config else 1
    
    # Convert mvar from 1-based (R style) to 0-based (Python style)
    mvar_py = mvar - 1
    print(f"Converting mvar from 1-based to 0-based: {mvar} -> {mvar_py}", flush=True)
    
    # Determine port based on site_id if not provided
    if remote_port is None:
        if site_id.endswith('1'):
            remote_port = 8001
        elif site_id.endswith('2'):
            remote_port = 8002
        else:
            remote_port = 8010
    
    # Define central URL
    central_url = f"ws://{central_host}:{central_port}/ws"
    
    # Create FastAPI app
    app = create_remote_app(D, mvar_py, central_url, site_id)
    
    # Run the app with uvicorn
    print(f"Starting remote site {site_id} on port {remote_port}")
    uvicorn.run(app, host="0.0.0.0", port=remote_port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SIMI Remote")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--mvar", type=int, required=True, help="Index of missing variable")
    parser.add_argument("--central_host", required=True, help="Central server hostname")
    parser.add_argument("--central_port", type=int, required=True, help="Central server port")
    parser.add_argument("--site_id", required=True, help="Remote site ID")
    parser.add_argument("--port", type=int, help="Port for this remote site")
    
    args = parser.parse_args()
    
    run_remote_client(
        args.data, 
        args.mvar, 
        args.central_host, 
        args.central_port, 
        args.site_id,
        args.port
    )
