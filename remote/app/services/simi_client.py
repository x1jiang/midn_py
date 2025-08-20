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
    
    async def _handle_algorithm_setup(self, client: ConnectionClient, websocket: Any, confirmation: Dict[str, Any]) -> None:
        """
        Handle SIMI-specific setup based on connection confirmation.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            confirmation: Connection confirmation message
        """
        # Call base implementation
        await super()._handle_algorithm_setup(client, websocket, confirmation)
        
        # For SIMI, check if this is a reconnection and we already have the method
        job_status = confirmation.get("status", "unknown")
        if job_status == "active" and hasattr(self, 'method') and self.method != "gaussian":
            # This is a reconnection to an active job, skip method waiting
            await client.send_status(f"Reconnection detected - using existing method: {self.method}")
            if hasattr(self.algorithm_instance, 'method'):
                self.algorithm_instance.method = self.method
            # Don't call _wait_for_method_instruction for reconnections
            return
        
        # For new connections, wait for method instruction
        await self._wait_for_method_instruction(client, websocket)
    
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
        await client.send_status(f"Starting SIMI computation")
        
        # Check if we already have the method (reconnection case)
        method_received = hasattr(self, 'method') and self.method != "gaussian"
        skip_initial_data = False
        
        if not method_received:
            # Wait for method message from central server (new connection)
            await client.send_status("Waiting for method specification from central server...")
            
            while True:
                message = await client.receive_message(websocket, timeout=15)  # Use timeout for setup
                if not message:
                    await client.send_status("Connection lost while waiting for method message")
                    return
                    
                message_type = message.get("message_type") or message.get("type")
                
                if message_type == "method":
                    # Method message received
                    method = message.get("method", "gaussian")
                    self.method = method
                    await self.process_method_message(method)
                    await client.send_status(f"Received method: {method}")
                    break
                elif message_type == ProtocolMessageType.JOB_COMPLETED.value:
                    # Job completed before method was received
                    await client.send_status("Job completed before computation started")
                    return
                elif message_type == "mode_sync":
                    # Mode sync message for reconnection
                    current_mode = message.get("current_mode", 1)
                    await client.send_status(f"Received mode sync - job is in iteration mode {current_mode}")
                    skip_initial_data = True
                else:
                    await client.send_status(f"Received unexpected message type: {message_type}")
        else:
            # We already have the method from reconnection setup
            await client.send_status(f"Using existing method from reconnection: {self.method}")
            
            # Check for mode sync message to see if we should skip initial data
            await client.send_status("Checking for mode sync information...")
            try:
                print(f"🔍 SIMI Client: Waiting for mode sync message with 5s timeout...")
                message = await client.receive_message(websocket, timeout=5)
                print(f"📨 SIMI Client: Received message during mode sync check: {message}")
                
                if message and message.get("type") == "mode_sync":
                    current_mode = message.get("current_mode", 1)
                    skip_initial_flag = message.get("skip_initial", False)
                    await client.send_status(f"Received mode sync - mode {current_mode}, skip_initial={skip_initial_flag}")
                    print(f"✅ SIMI Client: Mode sync received - mode {current_mode}, skip_initial={skip_initial_flag}")
                    skip_initial_data = skip_initial_flag
                elif message and message.get("type") == "method":
                    # Method message during reconnection - this is expected but should be ignored for iterations
                    await client.send_status(f"Received method message during reconnection: {message.get('method')}")
                    print(f"🔧 SIMI Client: Method message received during reconnection - method already known")
                    # Don't set as pending - method messages are not part of iteration flow
                elif message:
                    # Put the message back for later processing (not mode sync)
                    print(f"📝 SIMI Client: Received other message: {message}")
                    await client.send_status(f"Received other message: {message.get('type', 'unknown')}")
                    self._pending_message = message
                else:
                    print(f"⚠️ SIMI Client: No message received during mode sync check")
                    await client.send_status("No mode sync message received")
            except Exception as e:
                # No mode sync message, proceed normally
                print(f"⚠️ SIMI Client: Exception during mode sync check: {e}")
                await client.send_status(f"Mode sync check failed: {e}")
                pass
        
        # Now proceed with computation based on the method
        if self.method == "gaussian":
            # For Gaussian, send the statistics
            ls_stats = await self.algorithm_instance.process_message("method", {"method": self.method})
            
            # Create algorithm message with "data" type
            data_message = {
                "type": "data",
                "job_id": job_id,
                **ls_stats
            }
            
            if not await client.send_message_dict(websocket, data_message):
                await client.send_status("Failed to send data, job may have failed")
                return
            
            await client.send_status("Gaussian statistics sent to central server")
            
        else:  # Logistic method
            # For logistic regression, check if we should skip initial data
            print(f"🔄 SIMI Client: Logistic method - skip_initial_data = {skip_initial_data}")
            
            if not skip_initial_data:
                # Send initial sample size only if not reconnecting to active job
                print(f"📤 SIMI Client: Sending initial data (n message): {initial_data}")
                initial_message = {
                    "type": "n",
                    "job_id": job_id,
                    **initial_data
                }
                
                if not await client.send_message_dict(websocket, initial_message):
                    await client.send_status("Failed to send initial data")
                    return
                
                await client.send_status("Initial sample size sent to central server")
                print(f"✅ SIMI Client: Successfully sent initial data")
            else:
                await client.send_status("Skipping initial data - joining job in progress")
                print(f"⏭️ SIMI Client: Skipped initial data sending due to reconnection to active job")
            
            # Continue processing algorithm messages for logistic iterations
            await client.send_status("Starting logistic regression iteration loop...")
            print(f"🔄 SIMI Client: Entering iteration loop...")
            
            while True:
                # Check if we have a pending message from mode sync check
                message = None
                if hasattr(self, '_pending_message'):
                    message = self._pending_message
                    print(f"📝 SIMI Client: Using pending message: {message}")
                    delattr(self, '_pending_message')
                else:
                    print(f"⏳ SIMI Client: Waiting for next iteration message (60s timeout)...")
                    # Use 60 second timeout for logistic regression iterations
                    message = await client.receive_message(websocket, timeout=60)
                
                if not message:
                    # Timeout or connection error during computation - treat as job error
                    await client.send_status("TIMEOUT: No response from central server within 60s - marking job as error")
                    print(f"❌ SIMI Client: Timeout during logistic regression - setting job error")
                    
                    # Send error message to central server
                    error_message = {
                        "type": ProtocolMessageType.ERROR.value,
                        "job_id": job_id,
                        "site_id": client.site_id,
                        "error": "Timeout waiting for mode message during logistic regression",
                        "timeout_duration": 60
                    }
                    await client.send_message_dict(websocket, error_message)
                    await client.send_status("Sent error notification to central server")
                    print(f"📤 SIMI Client: Sent timeout error to central server")
                    break
                
                print(f"📨 SIMI Client: Received iteration message: {message}")
                    
                # Process the message and check if we should continue
                should_continue = await self._process_algorithm_message(client, websocket, message)
                print(f"🔄 SIMI Client: Message processed, should_continue = {should_continue}")
                if not should_continue:
                    print(f"🏁 SIMI Client: Iteration loop completed")
                    break
    
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
        print(f"🎯 SIMI Client: Processing algorithm message type: {message_type}")
        print(f"📝 SIMI Client: Full message: {message}")
        
        if message_type == "mode":
            mode = message.get("mode", 0)
            print(f"🔢 SIMI Client: Processing mode {mode}")
            
            # Mode 0 means termination (backward compatibility)
            if mode == 0:
                print(f"🛑 SIMI Client: Received termination signal (mode 0)")
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
            print(f"⚙️ SIMI Client: Processing logistic iteration {mode}...")
            await client.send_status(f"Processing logistic iteration {mode}...")
            
            # Make sure we have the beta parameter
            if "beta" not in message:
                print(f"❌ SIMI Client: Missing 'beta' parameter in mode message")
                await client.send_status("Missing 'beta' parameter in mode message")
                return False  # Exit processing loop
            
            # Process the message with the algorithm
            await client.send_status(f"Processing mode {mode} with beta values...")
            print(f"🔧 SIMI Client: Calling algorithm.process_message for mode {mode}")
            results = await self.algorithm_instance.process_message("mode", message)
            print(f"📊 SIMI Client: Algorithm returned results: {list(results.keys()) if isinstance(results, dict) else type(results)}")
            
            # Send results based on the mode
            if mode >= 2:
                print(f"🔍 SIMI Client: Mode {mode} >= 2, looking for Q data")
                # Mode 2+ expects only Q for line search - check both 'Q' and 'nQ' keys
                q_value = None
                if "nQ" in results:
                    q_value = results["nQ"]
                elif "Q" in results:
                    q_value = results["Q"]
                
                if q_value is not None:
                    payload = {"type": "Q", "job_id": self.job_state.get("job_id"), "Q": q_value}
                    print(f"📤 SIMI Client: Sending Q payload: {payload}")
                    await client.send_status(f"Sending Q value: {q_value} for line search")
                    
                    send_success = await client.send_message_dict(websocket, payload)
                    print(f"📨 SIMI Client: Q message send result: {send_success}")
                    
                    if not send_success:
                        await client.send_status("Failed to send Q data")
                        return False  # Exit processing loop
                    else:
                        await client.send_status("Sent Q message to central server")
                        print(f"✅ SIMI Client: Successfully sent Q data for mode {mode}")
                else:
                    print(f"❌ SIMI Client: Missing Q data (neither 'Q' nor 'nQ') in results for mode {mode}")
                    print(f"🔍 SIMI Client: Available keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                    await client.send_status(f"Error: Missing Q data for mode {mode}")
                    return False  # Exit processing loop
            else:
                # Mode 1 expects H, g, Q
                # Send H first (largest payload)
                if "H" in results:
                    payload = {"type": "H", "job_id": self.job_state.get("job_id"), "H": results["H"]}
                    await client.send_status(f"Sending H matrix of size {len(results['H'])}x{len(results['H'][0]) if len(results['H']) > 0 else 0}")
                    if not await client.send_message_dict(websocket, payload):
                        await client.send_status("Failed to send H data")
                        return False  # Exit processing loop
                
                # Then send g
                if "g" in results:
                    payload = {"type": "g", "job_id": self.job_state.get("job_id"), "g": results["g"]}
                    await client.send_status(f"Sending g vector of size {len(results['g'])}")
                    if not await client.send_message_dict(websocket, payload):
                        await client.send_status("Failed to send g data")
                        return False  # Exit processing loop
                
                # Finally send Q
                if "Q" in results:
                    payload = {"type": "Q", "job_id": self.job_state.get("job_id"), "Q": results["Q"]}
                    await client.send_status(f"Sending Q value: {results['Q']}")
                    if not await client.send_message_dict(websocket, payload):
                        await client.send_status("Failed to send Q data")
                        return False  # Exit processing loop
        
        elif message_type in ["job_completed", "job_complete"]:
            # Handle both standardized and legacy completion messages
            print(f"🏁 SIMI Client: Received completion message: {message_type}")
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
        
        elif message_type == "method":
            # Method messages are setup messages, not iteration messages
            print(f"🔧 SIMI Client: Received method message during iteration - ignoring")
            await client.send_status(f"Ignoring method message during iteration: {message.get('method')}")
            # Continue processing (don't exit)
            
        else:
            # Handle unknown message types
            print(f"❓ SIMI Client: Unknown message type: {message_type}")
            await client.send_status(f"Received unknown message type: {message_type}")
            print(f"📝 SIMI Client: Full unknown message: {message}")
            
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
    
    async def process_method_message(self, method: str) -> None:
        """
        Process a method message from the central site.
        
        Args:
            method: Method to use for the algorithm (gaussian or logistic)
        """
        print(f"🎯 SIMI: Setting method to {method}")
        self.method = method
        if hasattr(self.algorithm_instance, 'set_method'):
            self.algorithm_instance.set_method(method)
