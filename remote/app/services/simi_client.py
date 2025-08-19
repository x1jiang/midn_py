"""
SIMI client implementation for remote site using standardized job protocol.
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, Type

from common.algorithm.base import RemoteAlgorithm
from common.algorithm.job_protocol import Protocol, JobStatus, RemoteStatus, ProtocolMessageType, ErrorCode
from .federated_job_protocol_client import FederatedJobProtocolClient
from ..websockets.connection_client import ConnectionClient


class SIMIClient(FederatedJobProtocolClient):
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
        # Binary flag (from parameter or extra_params)
        # Parameter takes precedence over extra_params
        if extra_params is None:
            extra_params = {}
            
        if is_binary:
            extra_params["is_binary"] = True
        elif "is_binary" in extra_params:
            is_binary = extra_params.get("is_binary", False)
        
        # Set method based on is_binary flag
        self.method = "logistic" if is_binary else "gaussian"
        
        # Call the base implementation with the updated extra_params
        await super().run_algorithm(
            data=data, 
            target_column=target_column,
            job_id=job_id, 
            site_id=site_id, 
            central_url=central_url, 
            token=token, 
            extra_params=extra_params,
            status_callback=status_callback
        )
    
    async def _process_method_instruction(self, client: ConnectionClient, method: str) -> None:
        """
        Process a method instruction from the central server.
        
        Args:
            client: Connection client
            method: Method to use
        """
        # Update the method based on the instruction from central
        self.method = method.lower()
        await client.send_status(f"Using SIMI method: {self.method}")
        
        # Make sure the algorithm instance has the correct method
        if hasattr(self.algorithm_instance, 'method'):
            if self.algorithm_instance.method != self.method:
                self.algorithm_instance.method = self.method
                await client.send_status(f"Updated algorithm method to {self.method}")
    
    async def _handle_algorithm_computation(self, client: ConnectionClient, websocket: Any,
                                           data: np.ndarray, target_column: int, 
                                           job_id: int, initial_data: Dict[str, Any]) -> None:
        """
        Handle SIMI-specific computation.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            data: Data array
            target_column: Target column index
            job_id: Job ID
            initial_data: Initial prepared data
        """
        await client.send_status(f"Starting SIMI computation with method: {self.method}")
        
        # Different handling based on the method
        if self.method == "gaussian":
            # For Gaussian, send the statistics
            ls_stats = await self.algorithm_instance.process_message("method", {"method": self.method})
            if not await client.send_message(websocket, ProtocolMessageType.DATA, job_id=job_id, **ls_stats):
                await client.send_status("Failed to send data, job may have failed")
                return
            
            await client.send_status("Gaussian statistics sent to central server")
            
        else:  # Logistic method
            # First send initial sample size
            if not await client.send_message(websocket, ProtocolMessageType.DATA, job_id=job_id, **initial_data):
                await client.send_status("Failed to send initial data")
                return
            
            await client.send_status("Initial sample size sent to central server")
    
    async def _process_algorithm_message(self, client: ConnectionClient, websocket: Any, message: Dict[str, Any]) -> bool:
        """
        Process an algorithm-specific message.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message to process
            
        Returns:
            True to continue processing, False to exit processing loop
        """
        message_type = message.get("type")
        
        if message_type == "mode":
            mode = message.get("mode", 0)
            
            # Mode 0 means termination (backward compatibility)
            if mode == 0:
                await client.send_status("Received legacy termination signal (mode 0)")
                
                # Convert to standardized format
                standardized_message = {
                    "type": ProtocolMessageType.JOB_COMPLETED.value,
                    "job_id": self.job_state.get("job_id"),
                    "status": JobStatus.COMPLETED.value,
                    "message": "Job completed successfully (mode 0)"
                }
                await self._handle_job_completed(client, websocket, standardized_message)
                return False  # Exit processing loop
            
            # Process this iteration
            await client.send_status(f"Processing logistic iteration {mode}...")
            
            # Make sure we have the beta parameter
            if "beta" not in message:
                await client.send_status("Missing 'beta' parameter in mode message")
                return False  # Exit processing loop
            
            # Process the message with the algorithm
            await client.send_status(f"Processing mode {mode} with beta values...")
            results = await self.algorithm_instance.process_message("mode", message)
            
            # Send results based on the mode
            if mode >= 2 and "nQ" in results:
                # Mode 2+ expects only Q for line search
                payload = {"type": "Q", "Q": results["nQ"]}
                await client.send_status(f"Sending Q value: {results['nQ']} for line search")
                if not await client.send_message(websocket, ProtocolMessageType.DATA, job_id=self.job_state.get("job_id"), **payload):
                    await client.send_status("Failed to send Q data")
                    return False  # Exit processing loop
            else:
                # Mode 1 expects H, g, Q
                # Send H first (largest payload)
                if "H" in results:
                    payload = {"type": "H", "H": results["H"]}
                    await client.send_status(f"Sending H matrix of size {len(results['H'])}x{len(results['H'][0]) if len(results['H']) > 0 else 0}")
                    if not await client.send_message(websocket, ProtocolMessageType.DATA, job_id=self.job_state.get("job_id"), **payload):
                        await client.send_status("Failed to send H data")
                        return False  # Exit processing loop
                
                # Then send g
                if "g" in results:
                    payload = {"type": "g", "g": results["g"]}
                    await client.send_status(f"Sending g vector of size {len(results['g'])}")
                    if not await client.send_message(websocket, ProtocolMessageType.DATA, job_id=self.job_state.get("job_id"), **payload):
                        await client.send_status("Failed to send g data")
                        return False  # Exit processing loop
                
                # Finally send Q
                if "Q" in results:
                    payload = {"type": "Q", "Q": results["Q"]}
                    await client.send_status(f"Sending Q value: {results['Q']}")
                    if not await client.send_message(websocket, ProtocolMessageType.DATA, job_id=self.job_state.get("job_id"), **payload):
                        await client.send_status("Failed to send Q data")
                        return False  # Exit processing loop
        
        elif message_type in ["job_completed", "job_complete"]:
            # Handle both standardized and legacy completion messages
            if message_type == "job_complete":
                # Convert legacy format to standardized format
                standardized_message = {
                    "type": ProtocolMessageType.JOB_COMPLETED.value,
                    "job_id": self.job_state.get("job_id"),
                    "status": JobStatus.COMPLETED.value,
                    "message": "Job completed successfully (legacy format)"
                }
                await self._handle_job_completed(client, websocket, standardized_message)
            else:
                # Handle standardized format
                await self._handle_job_completed(client, websocket, message)
                
            return False  # Exit processing loop
            
        return True  # Continue processing
    
    async def _handle_job_completed(self, client: ConnectionClient, websocket: Any, message: Dict[str, Any]) -> None:
        """
        Handle a job completion notification with SIMI-specific logic.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Job completion message
        """
        # Call the base implementation
        await super()._handle_job_completed(client, websocket, message)
        
        # Add SIMI-specific completion handling
        await client.send_status("SIMI job completed successfully")
        
        # Check if we should reset for the next iteration
        if message.get("next_iteration", False):
            # Reset job state for the next iteration
            self.job_state["job_completed"] = False
            self.job_state["completion_acknowledged"] = False
            
            # Wait before reconnection for next iteration
            await client.send_status("Waiting for next iteration...")
            await asyncio.sleep(30)
        else:
            # Final completion, no need to reset
            await client.send_status("SIMI job fully completed, no more iterations required")
            # Leave job_completed and completion_acknowledged as True
    
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
