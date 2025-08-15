"""
SIMI client implementation for remote site.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Type

from common.algorithm.base import RemoteAlgorithm
from common.algorithm.protocol import MessageType
from ..websockets.connection_client import ConnectionClient
from .base_algorithm_client import BaseAlgorithmClient


class SIMIClient(BaseAlgorithmClient):
    """
    Client for the SIMI algorithm on the remote site.
    """
    
    def __init__(self, algorithm_class: Type[RemoteAlgorithm]):
        """
        Initialize SIMI client.
        
        Args:
            algorithm_class: Algorithm class to use
        """
        super().__init__(algorithm_class)
        self.algorithm_instance = algorithm_class()
        self.method = "gaussian"  # Default method
    
    async def run_algorithm(self, data: np.ndarray, target_column: int,
                           job_id: int, site_id: str, central_url: str, token: str,
                           extra_params: Optional[Dict[str, Any]] = None,
                           status_callback: Optional[Any] = None,
                           is_binary: bool = False) -> None:
        """
        Run the SIMI algorithm.
        
        Args:
            data: Data array
            target_column: Index of the target column
            job_id: ID of the job
            site_id: ID of this site
            central_url: URL of the central server
            token: Authentication token
            extra_params: Additional parameters for the algorithm
            status_callback: Callback for status updates
            is_binary: Whether the target variable is binary (for logistic regression)
        """
        # Create a connection client
        client = ConnectionClient(central_url, site_id, token, status_callback)
        
        # Prepare the data
        initial_data = await self.algorithm_instance.prepare_data(data, target_column)
        
        # Binary flag (from parameter or extra_params)
        # Parameter takes precedence over extra_params
        if extra_params and "is_binary" in extra_params and not is_binary:
            is_binary = extra_params.get("is_binary", False)
        
        # Set method based on is_binary flag
        self.method = "logistic" if is_binary else "gaussian"
        
        # Add any extra parameters to the connection message
        connect_message = {
            "type": "connect",
            "job_id": job_id
        }
        if extra_params:
            connect_message.update(extra_params)
        
        # Main connection loop - implementing the required communication pattern
        while not client.is_job_stopped():
            # Try to connect
            success, websocket = await client.connect()
            if not success:
                # Connection failed - could mean central is not ready or job already running
                await client.send_status(f"Connection failed, will retry in {client.retry_delay} seconds...")
                await asyncio.sleep(client.retry_delay)
                continue
            
            try:
                # Connection successful - now try to start the job
                await client.send_status("Connected to central server, attempting to start job...")
                
                # Send connect message
                if not await client.send_message(websocket, MessageType.CONNECT, **connect_message):
                    # Connection error during send, retry
                    await client.send_status("Failed to send connection message, reconnecting...")
                    continue
                
                # Wait for response from central
                message = await client.receive_message(websocket)
                if not message:
                    await client.send_status("No response from central, reconnecting...")
                    continue
                
                message_type = message.get("type")
                
                # Check if central rejected us due to various reasons
                if message_type == "error" or message_type == "job_conflict":
                    error_msg = message.get("message", "Job conflict or central busy")
                    error_code = message.get("code", "UNKNOWN")
                    await client.send_status(f"Central server error: {error_msg} (Code: {error_code})")
                    
                    # Different wait times based on error type
                    if error_code == "NO_JOBS_AVAILABLE":
                        # No jobs available - use normal retry delay (15 seconds)
                        await client.send_status(f"No jobs available. Waiting {client.retry_delay} seconds before retry...")
                        await asyncio.sleep(client.retry_delay)
                    elif error_code in ["JOB_NOT_FOUND", "MISSING_JOB_ID", "UNAUTHORIZED_SITE"]:
                        # These are configuration errors - wait longer before retry  
                        await client.send_status(f"Configuration issue. Waiting {client.completion_wait_time} seconds before retry...")
                        await asyncio.sleep(client.completion_wait_time)
                    else:
                        # Default case - assume job conflict, wait for completion
                        await client.send_status(f"Job conflict or other issue. Waiting {client.completion_wait_time} seconds...")
                        await asyncio.sleep(client.completion_wait_time)
                    continue
                
                if message_type != "method":
                    # Unexpected response, retry
                    await client.send_status(f"Unexpected response: {message_type}, reconnecting...")
                    continue
                
                # Successfully connected and received method instruction
                await client.send_status("Successfully connected! Job starting...")
                
                # Process the method message
                method = message.get("method", "gaussian").lower()
                await self.process_method_message(method)
                
                # Mark that we have an established connection for this job
                job_active = True
                
                # Handle the rest of the protocol based on the method
                if self.method == "gaussian":
                    # For Gaussian, just send the stats
                    ls_stats = await self.algorithm_instance.process_message("method", {"method": self.method})
                    if not await client.send_message(websocket, MessageType.DATA, **ls_stats):
                        await client.send_status("Failed to send data, job may have failed")
                        job_active = False
                        continue
                        
                    # Wait for completion signal or timeout
                    await client.send_status("Data sent, waiting for job completion...")
                    completion_timeout = 300  # 5 minutes for job completion
                    start_wait = asyncio.get_event_loop().time()
                    
                    while job_active and (asyncio.get_event_loop().time() - start_wait) < completion_timeout:
                        try:
                            # Check for completion message
                            completion_msg = await asyncio.wait_for(
                                client.receive_message(websocket), 
                                timeout=10  # Short timeout for checking
                            )
                            if completion_msg and completion_msg.get("type") == "job_complete":
                                await client.send_status("Job completed successfully!")
                                job_active = False
                                break
                        except asyncio.TimeoutError:
                            # No message yet, keep waiting
                            pass
                        except Exception:
                            # Connection lost or other error
                            job_active = False
                            break
                    
                    if job_active:
                        # Timeout waiting for completion
                        await client.send_status("Timeout waiting for job completion")
                    
                else:  # Logistic
                    # First send initial sample size
                    if not await client.send_message(websocket, "n", **initial_data):
                        await client.send_status("Failed to send initial data, reconnecting...")
                        job_active = False
                        continue
                    
                    # Then handle iterations from central
                    while job_active:
                        try:
                            message = await client.receive_message(websocket)
                            if not message:
                                await client.send_status("Failed to receive message, connection may be lost")
                                job_active = False
                                break
                                
                            message_type = message.get("type")
                            
                            if message_type == "mode":
                                mode = message.get("mode", 0)
                                
                                # Mode 0 means termination - job completed
                                if mode == 0:
                                    await client.send_status("Received job completion signal from central")
                                    job_active = False
                                    break
                                
                                # Process this iteration
                                await client.send_status(f"Processing iteration {mode}...")
                                
                                # Make sure we have the beta parameter in the payload
                                if "beta" not in message:
                                    await client.send_error("Missing 'beta' parameter in mode message")
                                    job_active = False
                                    break
                                
                                # Explicitly ensure the algorithm instance has the correct method set
                                if hasattr(self.algorithm_instance, 'method'):
                                    if self.algorithm_instance.method != self.method:
                                        await client.send_status(f"Updating algorithm method from {self.algorithm_instance.method} to {self.method}")
                                        self.algorithm_instance.method = self.method
                                    
                                # Process the message
                                results = await self.algorithm_instance.process_message("mode", message)
                                
                                # Different handling based on mode
                                if mode == 2 and "nQ" in results:
                                    # Mode 2 expects only nQ (line search)
                                    await client.send_status(f"Mode 2 (line search): Received nQ={results['nQ']}")
                                    
                                    # Send nQ as Q value to central
                                    payload = {"type": "Q", "Q": results["nQ"]}
                                    await client.send_status(f"Sending Q value: {results['nQ']} for line search")
                                    if not await client.send_message(websocket, "Q", **payload):
                                        await client.send_error(f"Failed to send Q data")
                                        job_active = False
                                        break
                                    else:
                                        await client.send_status(f"Sent Q data successfully for line search")
                                        
                                else:
                                    # Mode 1 and others expect H, g, Q
                                    missing_keys = [k for k in ["H", "g", "Q"] if k not in results]
                                    if missing_keys:
                                        await client.send_error(f"Missing expected keys in results: {missing_keys}")
                                        job_active = False
                                        break
                                    
                                    # Send H, g, Q separately
                                    await client.send_status(f"Sending results for iteration {mode}...")
                                    
                                    # First send H (largest payload, may take more time)
                                    if "H" in results:
                                        payload = {"type": "H", "H": results["H"]}
                                        await client.send_status(f"Sending H matrix of size {len(results['H'])}x{len(results['H'][0]) if len(results['H']) > 0 else 0}")
                                        if not await client.send_message(websocket, "H", **payload):
                                            await client.send_error(f"Failed to send H data")
                                            job_active = False
                                            break
                                        else:
                                            await client.send_status(f"Sent H data successfully")
                                    
                                    # Then send g
                                    if "g" in results:
                                        payload = {"type": "g", "g": results["g"]}
                                        await client.send_status(f"Sending g vector of size {len(results['g'])}")
                                        if not await client.send_message(websocket, "g", **payload):
                                            await client.send_error(f"Failed to send g data")
                                            job_active = False
                                            break
                                        else:
                                            await client.send_status(f"Sent g data successfully")
                                    
                                    # Finally send Q (scalar value)
                                    if "Q" in results:
                                        payload = {"type": "Q", "Q": results["Q"]}
                                        await client.send_status(f"Sending Q value: {results['Q']}")
                                        if not await client.send_message(websocket, "Q", **payload):
                                            await client.send_error(f"Failed to send Q data")
                                            job_active = False
                                            break
                                        else:
                                            await client.send_status(f"Sent Q data successfully")
                                            
                            elif message_type == "error":
                                error_msg = message.get("message", "Unknown error from central")
                                await client.send_status(f"Central server error: {error_msg}")
                                job_active = False
                                break
                                
                        except Exception as e:
                            await client.send_status(f"Error processing message: {str(e)}")
                            job_active = False
                            break
                
                # If we reach here, the job has completed (successfully or with error)
                if job_active:
                    await client.send_status("Job completed successfully!")
                
                # Mark job as completed and wait before next attempt
                client.mark_job_completed()
                await client.send_status(f"Job finished. Waiting {client.completion_wait_time} seconds before checking for new jobs...")
                await asyncio.sleep(client.completion_wait_time)
                
                # Reset for potential new job
                client.reset_for_new_job()
                
            except Exception as e:
                await client.send_status(f"Unexpected error: {str(e)}")
                # Connection lost or other error - wait and retry
                client.is_connection_established = False
                
                # Wait 2 minutes if we had an established connection (job may have been running)
                if client.is_connection_established:
                    await client.send_status("Connection lost during job execution. Waiting for central to complete...")
                    await asyncio.sleep(client.completion_wait_time)
                else:
                    # Quick retry if we never established connection
                    await asyncio.sleep(client.retry_delay)
    
    async def handle_message(self, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a message from the central site.
        
        Args:
            message_type: Type of the message
            payload: Message payload
            
        Returns:
            Response payload (if any)
        """
        return await self.algorithm_instance.process_message(message_type, payload)
    
    async def process_method_message(self, method: str) -> None:
        """
        Process a method message from the central site.
        
        Args:
            method: Method to use for the algorithm
        """
        self.method = method
