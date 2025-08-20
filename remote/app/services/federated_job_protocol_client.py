"""
Federated Job Protocol Client for remote site.
Provides standardized job management communication protocol for remote federated learning clients.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Type, Tuple

from common.algorithm.base import RemoteAlgorithm
from common.algorithm.job_protocol import (
    Protocol, JobStatus, RemoteStatus, ProtocolMessageType, ErrorCode,
    create_message, parse_message
)
from .base_algorithm_client import BaseAlgorithmClient
from ..websockets.connection_client import ConnectionClient


class FederatedJobProtocolClient(BaseAlgorithmClient):
    """
    Standardized client for job management protocol in federated learning algorithms.
    Handles job connections, status transitions, and completion notifications.
    Algorithm-specific data exchange is delegated to subclasses.
    """
    
    def __init__(self, algorithm_class: Type[RemoteAlgorithm]):
        """
        Initialize federated job protocol client.
        
        Args:
            algorithm_class: Algorithm class to use
        """
        super().__init__(algorithm_class)
        self.algorithm_instance = algorithm_class()
        self.method = None
        self.job_state = {
            "status": RemoteStatus.DISCONNECTED.value,
            "computation_started": False,
            "job_completed": False,
            "completion_acknowledged": False,
            "reconnection_attempts": 0
        }
    
    async def run_algorithm(self, data: np.ndarray, target_column: int,
                           job_id: int, site_id: str, central_url: str, token: str,
                           extra_params: Optional[Dict[str, Any]] = None,
                           status_callback: Optional[Any] = None) -> None:
        """
        Run the algorithm with standardized protocol handling.
        
        Args:
            data: Data array
            target_column: Index of the target column
            job_id: ID of the job
            site_id: ID of this site
            central_url: URL of the central server
            token: Authentication token
            extra_params: Additional parameters for the algorithm
            status_callback: Callback for status updates
        """
        # Initialize job parameters
        self.job_state["job_id"] = job_id
        self.job_state["site_id"] = site_id
        
        # Create connection client
        client = ConnectionClient(central_url, site_id, token, status_callback)
        
        # Prepare initial data
        initial_data = await self.algorithm_instance.prepare_data(data, target_column)
        
        # Additional params to include in connection message
        connect_message = Protocol.create_connect_message(job_id, site_id)
        if extra_params:
            connect_message.update(extra_params)
        
        # Main connection and job execution loop
        while not self._is_job_fully_completed():
            try:
                # Attempt to connect
                success, websocket = await self._establish_connection(client)
                if not success:
                    await self._handle_connection_failure(client)
                    continue
                
                # Connection established, send connect message
                self.job_state["status"] = RemoteStatus.CONNECTED.value
                await client.send_status(f"Connected to central server for job {job_id}")
                
                # Send connection request
                if not await client.send_message_dict(websocket, connect_message):
                    await client.send_status("Failed to send connection request")
                    await asyncio.sleep(client.retry_delay)
                    continue
                
                # Wait for connection confirmation
                confirmation = await client.receive_message(websocket)
                if not confirmation or confirmation.get("type") != ProtocolMessageType.CONNECTION_CONFIRMED.value:
                    if confirmation and confirmation.get("type") == ProtocolMessageType.ERROR.value:
                        error_code = confirmation.get("code", "UNKNOWN")
                        error_message = confirmation.get("message", "Unknown error")
                        await client.send_status(f"Connection error: {error_message} (Code: {error_code})")
                        await self._handle_connection_error(client, error_code)
                    else:
                        await client.send_status("Did not receive connection confirmation")
                        await asyncio.sleep(client.retry_delay)
                    continue
                
                # Connection confirmed, process any algorithm-specific setup
                await client.send_status("Connection confirmed by central server")
                
                # Process algorithm-specific connection data
                await self._handle_algorithm_setup(client, websocket, confirmation)
                
                # Send site_ready message
                self.job_state["status"] = RemoteStatus.READY.value
                site_ready_message = Protocol.create_site_ready_message(job_id, site_id)
                if not await client.send_message_dict(websocket, site_ready_message):
                    await client.send_status("Failed to send ready notification")
                    await asyncio.sleep(client.retry_delay)
                    continue
                
                await client.send_status("Sent ready notification to central server")
                
                # Wait for start_computation message
                start_received = False
                while not start_received and not self._is_job_fully_completed():
                    message = await client.receive_message(websocket)
                    if not message:
                        await client.send_status("Connection lost while waiting for start signal")
                        break
                        
                    message_type = message.get("type")
                    
                    if message_type == ProtocolMessageType.START_COMPUTATION.value:
                        # Start computation message received
                        start_received = True
                        self.job_state["status"] = RemoteStatus.COMPUTING.value
                        self.job_state["computation_started"] = True
                        await client.send_status("Received computation start signal")
                        
                        # Handle algorithm-specific computation
                        await self._handle_algorithm_computation(
                            client, websocket, data, target_column, job_id, initial_data
                        )
                        
                    elif message_type == ProtocolMessageType.JOB_COMPLETED.value:
                        # Job already completed
                        await self._handle_job_completed(client, websocket, message)
                        break
                        
                    elif message_type == ProtocolMessageType.ERROR.value:
                        # Error from central
                        error_code = message.get("code", "UNKNOWN")
                        error_message = message.get("message", "Unknown error")
                        await client.send_status(f"Error from central: {error_message} (Code: {error_code})")
                        await asyncio.sleep(client.retry_delay)
                        break
                    
                    else:
                        # Handle any other messages
                        await self._handle_other_message(client, websocket, message)
                
                # If job is fully completed, break the main loop
                if self._is_job_fully_completed():
                    break
                
            except Exception as e:
                # Handle any unexpected errors
                await client.send_status(f"Unexpected error: {str(e)}")
                await asyncio.sleep(client.retry_delay)
                self.job_state["reconnection_attempts"] += 1
    
    async def _establish_connection(self, client: ConnectionClient) -> Tuple[bool, Any]:
        """
        Establish a connection to the central server.
        
        Args:
            client: Connection client
            
        Returns:
            Tuple of (success, websocket)
        """
        # Reset connection state
        self.job_state["status"] = RemoteStatus.DISCONNECTED.value
        
        # Attempt to connect
        await client.send_status(f"Connecting to central server (attempt {self.job_state['reconnection_attempts'] + 1})...")
        success, websocket = await client.connect()
        
        return success, websocket
    
    async def _handle_connection_failure(self, client: ConnectionClient) -> None:
        """
        Handle a failed connection attempt.
        
        Args:
            client: Connection client
        """
        # Use exponential backoff for reconnection
        delay = min(30, client.retry_delay * (2 ** min(self.job_state["reconnection_attempts"], 4)))
        
        await client.send_status(f"Connection failed, will retry in {delay} seconds")
        await asyncio.sleep(delay)
        
        self.job_state["reconnection_attempts"] += 1
    
    async def _handle_connection_error(self, client: ConnectionClient, error_code: str) -> None:
        """
        Handle a connection error.
        
        Args:
            client: Connection client
            error_code: Error code from server
        """
        if error_code == ErrorCode.NO_JOBS_AVAILABLE.value:
            # No jobs available - use normal retry delay
            await client.send_status(f"No jobs available. Waiting {client.retry_delay} seconds before retry")
            await asyncio.sleep(client.retry_delay)
            
        elif error_code in [ErrorCode.JOB_NOT_FOUND.value, ErrorCode.MISSING_JOB_ID.value, ErrorCode.UNAUTHORIZED_SITE.value]:
            # Configuration errors - wait longer
            await client.send_status(f"Configuration issue. Waiting {client.completion_wait_time} seconds before retry")
            await asyncio.sleep(client.completion_wait_time)
            
        else:
            # Default case - assume job conflict
            await client.send_status(f"Job conflict or other issue. Waiting {client.completion_wait_time} seconds")
            await asyncio.sleep(client.completion_wait_time)
            
        self.job_state["reconnection_attempts"] += 1
    
    async def _handle_algorithm_setup(self, client: ConnectionClient, websocket: Any, confirmation: Dict[str, Any]) -> None:
        """
        Handle algorithm-specific setup based on connection confirmation.
        To be overridden by subclasses.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            confirmation: Connection confirmation message
        """
        # Default implementation just logs algorithm info
        algorithm = confirmation.get("algorithm", "Unknown")
        status = confirmation.get("status", "Unknown")
        await client.send_status(f"Connected to {algorithm} job with status: {status}")
        
        # Wait for method instruction if needed
        await self._wait_for_method_instruction(client, websocket)
    
    async def _wait_for_method_instruction(self, client: ConnectionClient, websocket: Any) -> None:
        """
        Wait for method instruction from central server.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
        """
        # Default implementation waits for a method message
        # Subclasses may override this if the method is determined differently
        try:
            # Wait for up to 10 seconds for a method message
            message = await asyncio.wait_for(client.receive_message(websocket), timeout=10)
            if message and message.get("type") == ProtocolMessageType.METHOD.value:
                method = message.get("method", "default")
                await self._process_method_instruction(client, method)
        except asyncio.TimeoutError:
            # No method instruction received, use default
            await client.send_status("No method instruction received, using default method")
        except Exception as e:
            await client.send_status(f"Error waiting for method instruction: {str(e)}")
    
    async def _process_method_instruction(self, client: ConnectionClient, method: str) -> None:
        """
        Process a method instruction from the central server.
        To be overridden by subclasses.
        
        Args:
            client: Connection client
            method: Method to use
        """
        # Default implementation just sets the method
        self.method = method
        await client.send_status(f"Using method: {method}")
        
        # Update algorithm instance method if it has that attribute
        if hasattr(self.algorithm_instance, 'method'):
            self.algorithm_instance.method = method
    
    async def _handle_algorithm_computation(self, client: ConnectionClient, websocket: Any,
                                          data: np.ndarray, target_column: int, 
                                          job_id: int, initial_data: Dict[str, Any]) -> None:
        """
        Handle algorithm-specific computation.
        To be overridden by subclasses.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            data: Data array
            target_column: Target column index
            job_id: Job ID
            initial_data: Initial prepared data
        """
        # Default implementation processes messages until job completion
        await client.send_status("Starting computation")
        
        try:
            # Process messages until job completion
            while True:
                message = await client.receive_message(websocket)
                if not message:
                    await client.send_status("Connection lost during computation")
                    break
                    
                message_type = message.get("type")
                
                if message_type == ProtocolMessageType.JOB_COMPLETED.value:
                    # Job completed
                    await self._handle_job_completed(client, websocket, message)
                    break
                    
                # Process other algorithm-specific messages
                result = await self._process_algorithm_message(client, websocket, message)
                if result is False:
                    # Processing indicates we should exit the loop
                    break
        except Exception as e:
            await client.send_status(f"Error during computation: {str(e)}")
    
    async def _process_algorithm_message(self, client: ConnectionClient, websocket: Any, message: Dict[str, Any]) -> bool:
        """
        Process an algorithm-specific message.
        To be overridden by subclasses.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message to process
            
        Returns:
            True to continue processing, False to exit processing loop
        """
        # Default implementation just logs the message type
        message_type = message.get("type", "unknown")
        await client.send_status(f"Received message of type: {message_type}")
        return True
    
    async def _handle_job_completed(self, client: ConnectionClient, websocket: Any, message: Dict[str, Any]) -> None:
        """
        Handle a job completion notification.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Job completion message
        """
        # Extract information from the completion message
        status = message.get("status", "completed")
        completion_message = message.get("message", "Job completed successfully")
        result_path = message.get("result_path", None)
        
        # Update job state
        self.job_state["status"] = RemoteStatus.COMPLETED.value
        self.job_state["job_completed"] = True
        
        # Log completion
        await client.send_status(f"Job completed with status: {status}")
        await client.send_status(f"Completion message: {completion_message}")
        if result_path:
            await client.send_status(f"Results available at: {result_path}")
        
        # Send acknowledgment
        ack_message = Protocol.create_completion_ack_message(
            job_id=self.job_state["job_id"],
            site_id=self.job_state["site_id"]
        )
        
        if await client.send_message_dict(websocket, ack_message):
            await client.send_status("Sent completion acknowledgment to central server")
            self.job_state["completion_acknowledged"] = True
        else:
            await client.send_status("Failed to send completion acknowledgment")
    
    async def _handle_other_message(self, client: ConnectionClient, websocket: Any, message: Dict[str, Any]) -> None:
        """
        Handle any other messages.
        To be overridden by subclasses if needed.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message to process
        """
        # Default implementation just logs the message type
        message_type = message.get("type", "unknown")
        await client.send_status(f"Received unexpected message of type: {message_type}")
    
    def _is_job_fully_completed(self) -> bool:
        """
        Check if the job is fully completed.
        
        Returns:
            True if the job is completed and acknowledged, False otherwise
        """
        return self.job_state.get("job_completed", False) and self.job_state.get("completion_acknowledged", False)
