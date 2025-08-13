import websockets
import asyncio
import numpy as np
import json
import pandas as pd

from algorithms.SIMI.SIMIRemote import SIRemoteLS, SIRemoteLogit

async def run_simi_remote(D, mvar, job_id, site_id, central_url, token, is_binary: bool = False, extra_params=None, status_callback=None):
    miss = np.isnan(D[:, mvar])
    X = D[~miss, :]
    X = np.delete(X, mvar, axis=1)
    y = D[~miss, mvar]

    method = "logistic" if is_binary else "gaussian"
    uri = f"{central_url}/ws/{site_id}?token={token}"
    
    # Log the connection details to help with debugging
    print(f"SIMI Remote connecting with site_id={site_id}, central_url={central_url}")
    
    # Function to check if job is stopped by user
    def is_job_stopped():
        if status_callback:
            job_state = status_callback.get_job_state()
            return job_state and job_state.get('completed', False)
        return False
    
    # Retry connection until user stops the job
    retry_delay = 5  # seconds
    attempt_count = 0
    
    while not is_job_stopped():
        attempt_count += 1
        try:
            # Update status before connecting
            if status_callback:
                if attempt_count == 1:
                    await status_callback.on_message(f"Connecting to central server: {uri}")
                else:
                    await status_callback.on_message(f"Connection attempt #{attempt_count}: Reconnecting to central server...")
            
            # Try to establish connection with timeouts
            async with websockets.connect(
                uri,
                ping_interval=None,  # Disable automatic ping
                close_timeout=5,     # Timeout for closing the connection
                open_timeout=10      # Timeout for opening the connection
            ) as websocket:
                # Connection successful
                if status_callback:
                    await status_callback.on_message("Connection established, initializing job...")
                
                # announce job and wait for central to instruct method
                connect_message = {
                    "type": "connect",
                    "job_id": job_id
                }
                
                # Add any extra parameters if provided
                if extra_params:
                    connect_message.update(extra_params)
                
                try:
                    # Set a timeout for the send operation
                    await asyncio.wait_for(websocket.send(json.dumps(connect_message)), timeout=5)
                    
                    if status_callback:
                        await status_callback.on_message(f"Sent connection message for job {job_id}")
    
                    # Set a timeout for the receive operation
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(msg)
                    central_method = data.get('method', '').lower()
                except asyncio.TimeoutError:
                    # If we hit a timeout, raise an exception to trigger retry
                    raise Exception("Timeout waiting for response from central server")
                
                # If central server specifies a method, use it
                if central_method:
                    method = central_method
                # Otherwise, ensure we're using logistic if is_binary was set to True
                elif is_binary and method != "logistic":
                    method = "logistic"
                    
                if status_callback:
                    await status_callback.on_message(f"Received method from central: {method}")
                    # Log the binary status for clarity
                    if method == "logistic":
                        await status_callback.on_message(f"Using binary imputation method (logistic)")

                # Create a modified websocket wrapper to track progress
                if status_callback:
                    original_send = websocket.send
                    original_recv = websocket.recv
                    
                    async def tracked_send(message):
                        try:
                            msg_data = json.loads(message)
                            msg_type = msg_data.get('type', '')
                            await status_callback.on_message(f"Sending {msg_type} message to central")
                        except:
                            pass
                        return await original_send(message)
                    
                    async def tracked_recv():
                        msg = await original_recv()
                        try:
                            msg_data = json.loads(msg)
                            msg_type = msg_data.get('type', '')
                            await status_callback.on_message(f"Received {msg_type} message from central")
                        except:
                            pass
                        return msg
                    
                    websocket.send = tracked_send
                    websocket.recv = tracked_recv

                # Determine the correct method based on the method variable
                if method == "logistic":
                    if status_callback:
                        await status_callback.on_message("Starting Logistic method for binary variable")
                    await SIRemoteLogit(X, y, websocket)
                else:
                    if status_callback:
                        await status_callback.on_message("Starting Gaussian method for continuous variable")
                    await SIRemoteLS(X, y, websocket)
                
                # Successfully completed a round of data exchange but don't mark the job as complete
                if status_callback:
                    await status_callback.on_message("Completed data exchange with central server, waiting for next round...")
                
                # Wait before continuing to the next retry cycle
                await asyncio.sleep(retry_delay)
                
        except Exception as e:
            if is_job_stopped():
                # If job was stopped by user, don't retry
                if status_callback:
                    await status_callback.on_message("Job stopped by user, canceling connection attempts")
                    await status_callback.on_error("Job stopped by user")
                return
            
            # Get more specific error message based on exception type
            error_message = str(e)
            if isinstance(e, websockets.exceptions.InvalidStatusCode):
                error_message = f"Server returned invalid status code: {str(e)}"
            elif isinstance(e, websockets.exceptions.ConnectionClosed):
                error_message = f"Connection closed unexpectedly: {str(e)}"
            elif isinstance(e, asyncio.TimeoutError):
                error_message = "Connection timed out"
            elif "Timeout" in str(e):
                error_message = f"Operation timed out: {str(e)}"
            
            # Log the error and retry after delay
            print(f"Connection error (attempt #{attempt_count}): {error_message}")
            if status_callback:
                await status_callback.on_message(f"Connection error: {error_message}")
                await status_callback.on_message(f"Will retry in {retry_delay} seconds... (Stop the job to cancel)")
            
            # Wait before retrying
            await asyncio.sleep(retry_delay)