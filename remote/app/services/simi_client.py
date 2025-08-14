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
        
        # Main connection loop
        while not client.is_job_stopped():
            # Try to connect
            success, websocket = await client.connect()
            if not success:
                # Wait before retrying
                await asyncio.sleep(client.retry_delay)
                continue
            
            try:
                # Send connect message
                if not await client.send_message(websocket, MessageType.CONNECT, **connect_message):
                    # Connection error, retry
                    continue
                
                # Wait for method instruction from central
                message = await client.receive_message(websocket)
                if not message or message.get("type") != "method":
                    # Invalid response, retry
                    continue
                
                # Process the method message
                method = message.get("method", "gaussian").lower()
                await self.process_method_message(method)
                
                # Handle the rest of the protocol based on the method
                if self.method == "gaussian":
                    # For Gaussian, just send the stats
                    ls_stats = await self.algorithm_instance.process_message("method", {"method": self.method})
                    if not await client.send_message(websocket, MessageType.DATA, **ls_stats):
                        continue
                        
                    # Wait for completion or timeout
                    await asyncio.sleep(30)
                    
                else:  # Logistic
                    # First send initial sample size
                    if not await client.send_message(websocket, "n", **initial_data):
                        continue
                    
                    # Then handle iterations from central
                    while True:
                        try:
                            message = await client.receive_message(websocket)
                            if not message:
                                await client.send_status("Failed to receive message, reconnecting...")
                                break
                                
                            message_type = message.get("type")
                            
                            if message_type == "mode":
                                mode = message.get("mode", 0)
                                
                                # Mode 0 means termination
                                if mode == 0:
                                    await client.send_status("Received termination signal from central")
                                    break
                                
                                # Process this iteration
                                await client.send_status(f"Processing iteration {mode}...")
                                
                                # Add debug information
                                await client.send_status(f"Message payload: {message}")
                                
                                # Make sure we have the beta parameter in the payload
                                if "beta" not in message:
                                    await client.send_error("Missing 'beta' parameter in mode message")
                                    break
                                
                                # Explicitly ensure the algorithm instance has the correct method set
                                if hasattr(self.algorithm_instance, 'method'):
                                    if self.algorithm_instance.method != self.method:
                                        await client.send_status(f"Updating algorithm method from {self.algorithm_instance.method} to {self.method}")
                                        self.algorithm_instance.method = self.method
                                    
                                # Process the message
                                results = await self.algorithm_instance.process_message("mode", message)
                                
                                # Add debug output for the results
                                await client.send_status(f"Algorithm returned: {results}")
                                
                                # Different handling based on mode
                                if mode == 2 and "nQ" in results:
                                    # Mode 2 expects only nQ (line search)
                                    await client.send_status(f"Mode 2 (line search): Received nQ={results['nQ']}")
                                    
                                    # Send nQ as Q value to central
                                    payload = {"type": "Q", "Q": results["nQ"]}
                                    await client.send_status(f"Sending Q value: {results['nQ']} for line search")
                                    if not await client.send_message(websocket, "Q", **payload):
                                        await client.send_error(f"Failed to send Q data")
                                        break
                                    else:
                                        await client.send_status(f"Sent Q data successfully for line search")
                                        
                                else:
                                    # Mode 1 and others expect H, g, Q
                                    missing_keys = [k for k in ["H", "g", "Q"] if k not in results]
                                    if missing_keys:
                                        await client.send_error(f"Missing expected keys in results: {missing_keys}")
                                    
                                    # Send H, g, Q separately
                                    await client.send_status(f"Sending results for iteration {mode}...")
                                    
                                    # First send H (largest payload, may take more time)
                                    if "H" in results:
                                        payload = {"type": "H", "H": results["H"]}
                                        await client.send_status(f"Sending H matrix of size {len(results['H'])}x{len(results['H'][0]) if len(results['H']) > 0 else 0}")
                                        if not await client.send_message(websocket, "H", **payload):
                                            await client.send_error(f"Failed to send H data")
                                            break
                                        else:
                                            await client.send_status(f"Sent H data successfully")
                                    
                                    # Then send g
                                    if "g" in results:
                                        payload = {"type": "g", "g": results["g"]}
                                        await client.send_status(f"Sending g vector of size {len(results['g'])}")
                                        if not await client.send_message(websocket, "g", **payload):
                                            await client.send_error(f"Failed to send g data")
                                            break
                                        else:
                                            await client.send_status(f"Sent g data successfully")
                                    
                                    # Finally send Q (scalar value)
                                    if "Q" in results:
                                        payload = {"type": "Q", "Q": results["Q"]}
                                        await client.send_status(f"Sending Q value: {results['Q']}")
                                        if not await client.send_message(websocket, "Q", **payload):
                                            await client.send_error(f"Failed to send Q data")
                                            break
                                        else:
                                            await client.send_status(f"Sent Q data successfully")
                        except Exception as e:
                            await client.send_status(f"Error processing message: {str(e)}")
                            break
                
            except Exception as e:
                await client.send_status(f"Error: {str(e)}")
            
            # Wait before next connection attempt
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
