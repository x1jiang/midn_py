"""
SIMICE client implementation for remote site using standardized job protocol.
Multiple Imputation using Chained Equations.
"""

import asyncio
import json
import numpy as np
import pandas as pd
import traceback
from typing import Dict, Any, List, Optional, Type, Tuple, Union

from common.algorithm.base import RemoteAlgorithm
from common.algorithm.job_protocol import (
    Protocol, JobStatus, RemoteStatus, ProtocolMessageType, ErrorCode,
    create_message, parse_message
)
from ..websockets.connection_client import ConnectionClient
from .federated_job_protocol_client import FederatedJobProtocolClient


class SIMICEClient(FederatedJobProtocolClient):
    """
    Client for the SIMICE algorithm on the remote site.
    Multiple Imputation using Chained Equations.
    """
    
    def __init__(self, algorithm_class: Type[RemoteAlgorithm]):
        """
        Initialize SIMICE client.
        
        Args:
            algorithm_class: Algorithm class to use
        """
        super().__init__(algorithm_class)
        self.target_column_indexes = []
        self.is_binary = []
        self.iteration_before_first_imputation = 5
        self.iteration_between_imputations = 3
    
    async def run_algorithm(self, data: np.ndarray, target_column: int,
                           job_id: int, site_id: str, central_url: str, token: str,
                           extra_params: Optional[Dict[str, Any]] = None,
                           status_callback: Optional[Any] = None,
                           **kwargs) -> None:
        """
        Run the SIMICE algorithm using the standardized federated job protocol.
        
        Args:
            data: Data array
            target_column: Index of the target column (used for compatibility, SIMICE uses multiple columns)
            job_id: ID of the job
            site_id: ID of this site
            central_url: URL of the central server
            token: Authentication token
            extra_params: Additional parameters for the algorithm
            status_callback: Callback for status updates
            **kwargs: Additional keyword arguments (target_column_indexes, is_binary, etc.)
        """
        # Extract SIMICE-specific parameters from kwargs and extra_params
        if extra_params is None:
            extra_params = {}
            
        target_column_indexes = kwargs.get('target_column_indexes')
        is_binary = kwargs.get('is_binary')
        
        if not target_column_indexes:
            target_column_indexes = extra_params.get('target_column_indexes')
            is_binary = extra_params.get('is_binary')
        
        # Fallback to single column if no specific columns provided
        if not target_column_indexes:
            target_column_indexes = [target_column + 1]  # Convert to 1-based
        
        if not is_binary:
            is_binary = [False] * len(target_column_indexes)
        
        # Store parameters and update extra_params to ensure they get passed to base class
        self.target_column_indexes = target_column_indexes
        self.is_binary = is_binary
        extra_params['target_column_indexes'] = target_column_indexes
        extra_params['is_binary'] = is_binary
        
        # Store additional parameters
        self.iteration_before_first_imputation = extra_params.get("iteration_before_first_imputation", 5)
        self.iteration_between_imputations = extra_params.get("iteration_between_imputations", 3)
        
        # Override the algorithm instance's prepare_data method to use SIMICE-specific parameters
        original_prepare_data = self.algorithm_instance.prepare_data
        
        async def simice_prepare_data(data_param, target_column_param):
            """Override prepare_data to use SIMICE-specific parameters."""
            return await original_prepare_data(
                data_param, 
                self.target_column_indexes, 
                self.is_binary
            )
        
        # Replace the algorithm instance's prepare_data method
        self.algorithm_instance.prepare_data = simice_prepare_data
        
        # Call the parent class's run_algorithm method which follows the standardized protocol
        await super().run_algorithm(
            data=data,
            target_column=target_column,  # This will be passed to our overridden prepare_data
            job_id=job_id,
            site_id=site_id,
            central_url=central_url,
            token=token,
            extra_params=extra_params,
            status_callback=status_callback
        )
    
    async def _handle_algorithm_computation(self, client: ConnectionClient, websocket: Any,
                                            data: np.ndarray, target_column: int, 
                                            job_id: int, initial_data: Dict[str, Any]) -> None:
        """
        Handle SIMICE-specific computation following R implementation.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            data: Data array
            target_column: Target column index (not used in SIMICE)
            job_id: Job ID
            initial_data: Initial prepared data
        """
        await client.send_status("Starting SIMICE computation...")
        
        # Wait for Initialize message from central server (R-style)
        await client.send_status("Waiting for Initialize message from central server...")
        
        while True:
            message = await client.receive_message(websocket, timeout=None)  # Wait indefinitely
            if not message:
                await client.send_status("Connection lost while waiting for Initialize message")
                return
                
            message_type = message.get("message_type") or message.get("type")
            instruction = message.get("instruction", "")
            
            if instruction == "Initialize" or instruction == "initialize":
                # Initialize message received (R-style)
                await client.send_status("Received Initialize message")
                break
            elif message_type == "method":
                # Method specification message received
                await client.send_status("Received method specification message")
                break  
            elif message_type == "instruction":
                # Instruction message received
                await client.send_status(f"Received instruction message: {instruction}")
                break
            elif message_type == "data":
                # Data message received
                await client.send_status(f"Received data message: {instruction}")
                break
            elif message_type == "request_data_summary":
                # Legacy data summary request received
                await client.send_status("Received legacy data summary request")
                break
            elif message_type == ProtocolMessageType.JOB_COMPLETED.value:
                # Job completed before computation started
                await client.send_status("Job completed before computation started")
                return
            else:
                await client.send_status(f"Received unexpected message type: {message_type}, instruction: {instruction}")

        try:
            # Data has already been prepared by the parent class, use the initial_data
            # that was passed to this method
            
            # Handle the first message appropriately
            if instruction == "Initialize":
                # Process R-style Initialize message
                await self._handle_initialize_message(client, websocket, message, job_id)
            else:
                # Handle legacy data summary request
                await client.send_status("Sending data summary to central...")
                
                # Create data summary message
                # Use Protocol helper for creating DATA messages
                data_summary_message = Protocol.create_data_message(
                    job_id=job_id,
                    instruction="data_summary",
                    data=initial_data
                )
                
                if not await client.send_message_dict(websocket, data_summary_message):
                    await client.send_status("Failed to send data summary")
                    return

                await client.send_status("Data summary sent, waiting for instructions...")
            
            # Main SIMICE protocol loop
            while True:
                # Wait for instructions from central
                message = await client.receive_message(websocket, timeout=None)  # Wait indefinitely
                if not message:
                    await client.send_status("Connection closed")
                    break
                
                # Process the message
                continue_processing = await self._process_algorithm_message(client, websocket, message)
                if not continue_processing:
                    break
        
        except Exception as e:
            await client.send_status(f"Error in SIMICE computation: {str(e)}")
            traceback.print_exc()
    
    async def _process_algorithm_message(self, client: ConnectionClient, websocket: Any, 
                                        message: Dict[str, Any]) -> bool:
        """
        Process an algorithm-specific message using SIMI-like structure.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message to process
            
        Returns:
            True to continue processing, False to exit processing loop
        """
        print(f"🔧 SIMICE CLIENT DEBUG: _process_algorithm_message called")
        print(f"📋 SIMICE CLIENT DEBUG: Processing message: {message}")
        job_id = self.job_state.get("job_id")
        message_type = message.get("type")
        instruction = message.get("instruction", "")
        
        print(f"🔍 SIMICE CLIENT DEBUG: Processing algorithm message")
        print(f"📋 SIMICE CLIENT DEBUG: Message type: {message_type}")
        print(f"📋 SIMICE CLIENT DEBUG: Instruction: {instruction}")
        print(f"📋 SIMICE CLIENT DEBUG: Full message: {message}")
        
        try:
            # Handle DATA type messages with instructions (similar to SIMI)
            if message_type == ProtocolMessageType.DATA.value:
                # Extract the instruction from the data message
                instruction = message.get("instruction", "")
                
                print(f"📤 SIMICE CLIENT DEBUG: Processing DATA message with instruction: {instruction}")
                print(f"📋 SIMICE CLIENT DEBUG: Full DATA message content: {message}")
                await client.send_status(f"Processing SIMICE DATA message with instruction: {instruction}")
                
                # Handle instruction-based messages (computation-related only)
                # Preserve R-style instruction names
                
                # R-style: "Initialize"
                if instruction == "Initialize":
                    print(f"🎯 SIMICE CLIENT DEBUG: Handling Initialize instruction (R-style)")
                    await self._handle_initialize_message(client, websocket, message, job_id)
                    return True
                
                # Legacy/custom instruction
                elif instruction == "initialize":
                    print(f"🎯 SIMICE CLIENT DEBUG: Handling initialize instruction")
                    await self._handle_initialize_message(client, websocket, message, job_id)
                    return True
                
                # R-style: "Information"  
                elif instruction == "Information":
                    print(f"📊 SIMICE CLIENT DEBUG: Handling Information instruction (R-style)")
                    await self._handle_statistics_request(client, websocket, message, job_id)
                    return True
                
                # Legacy/custom instruction
                elif instruction == "get_statistics":
                    print(f"📊 SIMICE CLIENT DEBUG: Handling get_statistics instruction")
                    await self._handle_statistics_request(client, websocket, message, job_id)
                    return True
                
                # R-style: "Impute"
                elif instruction == "Impute":
                    print(f"� SIMICE CLIENT DEBUG: Handling Impute instruction (R-style)")
                    await self._handle_update_imputations(client, websocket, message, job_id)
                    return True
                
                # Legacy/custom instruction
                elif instruction == "update_imputations":
                    print(f"🔄 SIMICE CLIENT DEBUG: Handling update_imputations instruction")
                    await self._handle_update_imputations(client, websocket, message, job_id)
                    return True
                
                # R-style: "End"
                elif instruction == "End":
                    print(f"🏁 SIMICE CLIENT DEBUG: Handling End instruction (R-style)")
                    await self._handle_end_message(client, websocket, message, job_id)
                    return False  # End the processing loop
                
                # Legacy/custom instruction
                elif instruction == "end":
                    print(f"🏁 SIMICE CLIENT DEBUG: Handling end instruction")
                    await self._handle_end_message(client, websocket, message, job_id)
                    return False  # End the processing loop
                    
                else:
                    # If we don't recognize the instruction, log it but continue processing
                    print(f"❓ SIMICE CLIENT DEBUG: Unknown instruction in DATA message: {instruction}")
                    await client.send_status(f"Unknown instruction in DATA message: {instruction}, continuing")
                    return True
            
            # Handle METHOD messages (algorithm specification)
            elif message_type == ProtocolMessageType.METHOD.value:
                print(f"🎯 SIMICE CLIENT DEBUG: Handling METHOD message")
                print(f"📋 SIMICE CLIENT DEBUG: METHOD message content: {message}")
                # Handle algorithm parameters in METHOD messages
                algorithm = message.get("algorithm")
                if algorithm == "SIMICE":
                    # Store algorithm parameters
                    old_target_columns = getattr(self, 'target_column_indexes', [])
                    self.target_column_indexes = message.get("target_column_indexes", self.target_column_indexes)
                    self.is_binary = message.get("is_binary", self.is_binary)
                    self.iteration_before_first_imputation = message.get("iteration_before_first", self.iteration_before_first_imputation)
                    self.iteration_between_imputations = message.get("iteration_between", self.iteration_between_imputations)
                    
                    print(f"🎯 SIMICE CLIENT DEBUG: Updated target_column_indexes from {old_target_columns} to {self.target_column_indexes}")
                    print(f"🎯 SIMICE CLIENT DEBUG: Updated is_binary to {self.is_binary}")
                    
                    await client.send_status(f"Received SIMICE method parameters: {len(self.target_column_indexes)} columns")
                    print(f"✅ SIMICE CLIENT DEBUG: Updated algorithm parameters from METHOD message")
                    return True
                
                # For backwards compatibility with legacy instructions
                instruction = message.get("instruction", "")
                if instruction == "Initialize":
                    await self._handle_initialize_message(client, websocket, message, job_id)
                    return True
                    
                return True
                
            # Handle error messages
            elif message_type == ProtocolMessageType.ERROR.value:
                error_msg = message.get("message", "Unknown error")
                await client.send_status(f"Error from server: {error_msg}")
                return False  # End the processing loop
                
            # Handle legacy message types for backward compatibility
            elif message_type == "compute_statistics":
                await self._handle_compute_statistics(client, websocket, message, job_id)
                return True
                
            elif message_type == "update_imputations":
                await self._handle_update_imputations(client, websocket, message, job_id)
                return True
                
            elif message_type == "get_final_data":
                await self._handle_get_final_data(client, websocket, message, job_id)
                return True
                
            # Handle job completion messages
            elif message_type in [ProtocolMessageType.JOB_COMPLETED.value, "job_complete"]:
                # Standardize the message format for the base handler
                if message_type == "job_complete":
                    # Convert legacy format to standardized format
                    standardized_message = {
                        "type": ProtocolMessageType.JOB_COMPLETED.value,
                        "job_id": job_id,
                        "status": JobStatus.COMPLETED.value,
                        "message": "Job completed successfully (legacy format)"
                    }
                    await self._handle_job_completed(client, websocket, standardized_message)
                else:
                    await self._handle_job_completed(client, websocket, message)
                
                return False  # Exit processing loop
                
            # Handle error messages
            elif message_type == ProtocolMessageType.ERROR.value:
                error_msg = message.get("message", "Unknown error")
                error_code = message.get("code", "UNKNOWN_ERROR")
                await client.send_status(f"Error from central: {error_msg} (Code: {error_code})")
                return False  # Exit processing loop
                
            # Handle unknown message types
            else:
                await client.send_status(f"Unknown message type: {message_type}, instruction: {instruction}")
                return True  # Continue processing
        
        except Exception as e:
            await client.send_status(f"Error processing message: {str(e)}")
            traceback.print_exc()
            return True  # Continue processing despite error

    # Message handlers for SIMI-style structure
    
    async def _handle_initialize_message(self, client: ConnectionClient, websocket: Any, 
                                        message: Dict[str, Any], job_id: int) -> None:
        """
        Handle initialize instruction from central (SIMI-style).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Initialize message
            job_id: Job ID
        """
        missing_variables = message.get("missing_variables", [])
        print(f"🎯 SIMICE CLIENT DEBUG: Initializing with missing variables: {missing_variables}")
        
        # Log using the appropriate terminology based on instruction
        instruction = message.get("instruction")
        if instruction == "Initialize":
            await client.send_status(f"Received Initialize instruction (R-style) with {len(missing_variables)} missing variables")
        else:
            await client.send_status(f"Received initialize instruction with {len(missing_variables)} missing variables")
        
        try:
            # Store missing variables for processing
            self.missing_variables = missing_variables
            print(f"🎯 SIMICE CLIENT DEBUG: Stored missing variables: {self.missing_variables}")
            
            # Check algorithm instance
            if not hasattr(self, 'algorithm_instance') or self.algorithm_instance is None:
                print(f"❌ SIMICE CLIENT DEBUG: Algorithm instance not found!")
                await client.send_status("Error: Algorithm instance not initialized")
                return
            
            print(f"✅ SIMICE CLIENT DEBUG: Algorithm instance exists: {type(self.algorithm_instance)}")
            
            # Prepare the algorithm instance with missing variables
            print(f"🔧 SIMICE CLIENT DEBUG: About to call initialize_imputation")
            await self.algorithm_instance.initialize_imputation(missing_variables)
            print(f"✅ SIMICE CLIENT DEBUG: Successfully called initialize_imputation")
            
            # Send acknowledgment back to central using R-style instruction if needed
            # Use R-style "Initialize_response" if original was "Initialize", otherwise use legacy name
            original_instruction = message.get("instruction", "")
            response_instruction = "Initialize_response" if original_instruction == "Initialize" else "initialize"
            
            response_message = Protocol.create_data_message(
                job_id=job_id,
                instruction=response_instruction,
                status="ready"
            )
            
            await client.send_message_dict(websocket, response_message)
            await client.send_status("Sent initialize acknowledgment")
            
        except Exception as e:
            await client.send_status(f"Error in initialize: {str(e)}")
            
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "message": f"Error during initialization: {str(e)}",
                "code": ErrorCode.COMPUTATION_ERROR.value
            }
            
            await client.send_message_dict(websocket, error_message)

    async def _handle_statistics_request(self, client: ConnectionClient, websocket: Any, 
                                         message: Dict[str, Any], job_id: int) -> None:
        """
        Handle get_statistics instruction from central (SIMI-style).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Statistics request message
            job_id: Job ID
        """
        print(f"📊 SIMICE CLIENT DEBUG: Handling statistics request")
        print(f"📋 SIMICE CLIENT DEBUG: Message content: {message}")
        
        # Handle R-style naming (yidx) and other naming conventions
        target_column_index = message.get("yidx") or message.get("target_column_index") or message.get("target_col_idx")
        method = message.get("method", "Gaussian")
        
        print(f"📊 SIMICE CLIENT DEBUG: Target column: {target_column_index}, Method: {method}")
        
        # Log using the appropriate terminology based on instruction
        instruction = message.get("instruction")
        if instruction == "Information":
            await client.send_status(f"Computing Information statistics for column {target_column_index} using {method} (R-style)")
        else:
            await client.send_status(f"Computing statistics for column {target_column_index} using {method}")
        
        try:
            # Compute local statistics using the algorithm
            print(f"🧮 SIMICE CLIENT DEBUG: About to compute local statistics")
            stats = await self.algorithm_instance.compute_local_statistics(target_column_index, method)
            print(f"📊 SIMICE CLIENT DEBUG: Computed statistics: {stats}")
            
            # Send statistics back to central using DATA message type with R-style instruction name
            # Use R-style "Information_response" if original was "Information", otherwise use legacy name
            original_instruction = message.get("instruction", "")
            response_instruction = "Information_response" if original_instruction == "Information" else "statistics"
            
            response_message = Protocol.create_data_message(
                job_id=job_id,
                instruction=response_instruction,
                yidx=target_column_index,  # R-style parameter name
                target_col_idx=target_column_index,  # Legacy parameter name for compatibility
                method=method,
                statistics=stats
            )
            
            print(f"📤 SIMICE CLIENT DEBUG: Sending response message with instruction '{response_instruction}': {response_message}")
            await client.send_message_dict(websocket, response_message)
            await client.send_status(f"Sent {response_instruction} for column {target_column_index}")
            print(f"✅ SIMICE CLIENT DEBUG: Successfully sent statistics response")
            
        except Exception as e:
            print(f"❌ SIMICE CLIENT DEBUG: Error in statistics computation: {str(e)}")
            await client.send_status(f"Error computing statistics: {str(e)}")
            
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "message": f"Error computing statistics: {str(e)}",
                "code": ErrorCode.COMPUTATION_ERROR.value
            }
            
            await client.send_message_dict(websocket, error_message)

    async def _handle_update_imputations(self, client: ConnectionClient, websocket: Any, 
                                    message: Dict[str, Any], job_id: int) -> None:
        """
        Handle update_imputations instruction from central (SIMI-style).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Imputation update message
            job_id: Job ID
        """
        # Handle R-style naming (yidx) and other naming conventions
        target_column_index = message.get("yidx") or message.get("target_column_index") or message.get("target_col_idx")
        method = message.get("method", "Gaussian")
        parameters = message.get("parameters", {})
        
        # Log using the appropriate terminology based on instruction
        instruction = message.get("instruction")
        if instruction == "Impute":
            await client.send_status(f"Processing Impute instruction for column {target_column_index} using {method} (R-style)")
        else:
            await client.send_status(f"Updating imputations for column {target_column_index} using {method}")
        
        try:
            # Update local imputations using the computed parameters
            await self.algorithm_instance.update_imputations(target_column_index, parameters, method)
            
            # Send confirmation back to central with DATA message type
            # Use R-style "Impute_response" if original was "Impute", otherwise use legacy name
            original_instruction = message.get("instruction", "")
            response_instruction = "Impute_response" if original_instruction == "Impute" else "imputation_updated"
            
            response_message = Protocol.create_data_message(
                job_id=job_id,
                instruction=response_instruction,
                yidx=target_column_index,  # R-style parameter name
                target_col_idx=target_column_index,  # Legacy parameter name for compatibility
                status="completed"
            )
            
            await client.send_message_dict(websocket, response_message)
            await client.send_status(f"Imputations updated for column {target_column_index}")
            
        except Exception as e:
            await client.send_status(f"Error updating imputations: {str(e)}")
            
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "message": f"Error updating imputations: {str(e)}",
                "code": ErrorCode.COMPUTATION_ERROR.value
            }
            
            await client.send_message_dict(websocket, error_message)

    async def _handle_end_message(self, client: ConnectionClient, websocket: Any, 
                                 message: Dict[str, Any], job_id: int) -> None:
        """
        Handle end instruction from central (SIMI-style).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: End message
            job_id: Job ID
        """
        # Log using the appropriate terminology based on instruction
        instruction = message.get("instruction")
        if instruction == "End":
            await client.send_status("Received End instruction - finalizing SIMICE (R-style)")
        else:
            await client.send_status("Received end instruction - finalizing SIMICE")
        
        try:
            # Finalize the imputation process
            await self.algorithm_instance.finalize_imputation()
            
            # Send acknowledgment back to central using R-style instruction if needed
            # Use R-style "End_response" if original was "End", otherwise use legacy name
            original_instruction = message.get("instruction", "")
            response_instruction = "End_response" if original_instruction == "End" else "end"
            
            response_message = Protocol.create_data_message(
                job_id=job_id,
                instruction=response_instruction,
                status="completed"
            )
            
            await client.send_message_dict(websocket, response_message)
            await client.send_status("SIMICE algorithm completed")
            
        except Exception as e:
            await client.send_status(f"Error in end: {str(e)}")
            
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "message": f"Error finalizing algorithm: {str(e)}",
                "code": ErrorCode.COMPUTATION_ERROR.value
            }
            
            await client.send_message_dict(websocket, error_message)
    
    # Legacy message handlers (for backward compatibility)
    
    async def _handle_compute_statistics(self, client: ConnectionClient, websocket: Any, 
                                        message: Dict[str, Any], job_id: int) -> None:
        """
        Handle compute statistics request (legacy format).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message with compute statistics request
            job_id: Job ID
        """
        target_col_idx = message.get("target_col_idx", 0)  # 0-based index
        method = message.get("method", "gaussian")
        
        await client.send_status(f"Computing statistics for column {target_col_idx} using {method} (legacy format)")
        
        try:
            # Compute local statistics using the algorithm
            stats = await self.algorithm_instance.compute_local_statistics(target_col_idx, method)
            
            # Send statistics back to central using the SIMI-like format
            await client.send_status(f"Sending statistics for column {target_col_idx}")
            
            statistics_message = {
                "type": ProtocolMessageType.DATA.value,
                "job_id": job_id,
                "instruction": "statistics_result",
                "target_column_index": target_col_idx,
                "method": method,
                "statistics": stats
            }
            
            await client.send_message_dict(websocket, statistics_message)
            
        except Exception as e:
            await client.send_status(f"Error computing statistics: {str(e)}")
            
            # Send error back to central
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "message": f"Error computing statistics: {str(e)}",
                "code": ErrorCode.COMPUTATION_ERROR.value
            }
            
            await client.send_message_dict(websocket, error_message)
    
    async def _handle_update_imputations(self, client: ConnectionClient, websocket: Any, 
                                        message: Dict[str, Any], job_id: int) -> None:
        """
        Handle update imputations request.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message with update imputations request
            job_id: Job ID
        """
        target_col_idx = message.get("target_col_idx", 0)  # 0-based index
        global_params = message.get("global_parameters", {})
        
        await client.send_status(f"Updating imputations for column {target_col_idx}")
        
        try:
            # Update local imputations using the algorithm
            updated_data = await self.algorithm_instance.update_imputations(target_col_idx, global_params)
            
            # Send confirmation back to central
            update_message = {
                "type": "imputation_updated",
                "job_id": job_id,
                "target_col_idx": target_col_idx,
                "status": "completed"
            }
            
            await client.send_message_dict(websocket, update_message)
            
            await client.send_status(f"Imputations updated for column {target_col_idx}")
            
        except Exception as e:
            await client.send_status(f"Error updating imputations: {str(e)}")
            
            # Send error back to central
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "target_col_idx": target_col_idx,
                "error": str(e)
            }
            
            await client.send_message_dict(websocket, error_message)
    
    async def _handle_get_final_data(self, client: ConnectionClient, websocket: Any, 
                                    message: Dict[str, Any], job_id: int) -> None:
        """
        Handle request for final imputed data (legacy format).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message with get final data request
            job_id: Job ID
        """
        await client.send_status(f"Collecting final imputed data (legacy format)")
        
        try:
            # Get the final imputed datasets from the algorithm
            imputed_data = {}
            
            # Get the current data with all imputations from the algorithm
            final_data = await self.algorithm_instance.get_final_imputed_data()
            
            if final_data:
                # Convert DataFrames to dictionaries for JSON serialization
                for key, df in final_data.items():
                    if isinstance(df, pd.DataFrame):
                        imputed_data[key] = df.to_dict(orient='split')
                    else:
                        imputed_data[key] = df
            
            # Send the final data back to central using SIMI-like format
            final_data_message = {
                "type": ProtocolMessageType.DATA.value,
                "job_id": job_id,
                "instruction": "final_data",
                "imputed_data": imputed_data
            }
            
            await client.send_message_dict(websocket, final_data_message)
            
            await client.send_status(f"Sent final imputed data")
            
        except Exception as e:
            await client.send_status(f"Error getting final data: {str(e)}")
            
            # Send error response
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "message": f"Error getting final data: {str(e)}",
                "code": ErrorCode.COMPUTATION_ERROR.value
            }
            
            await client.send_message_dict(websocket, error_message)
    
    async def _handle_job_completed(self, client: ConnectionClient, websocket: Any, 
                                   message: Dict[str, Any]) -> None:
        """
        Handle a job completion notification with SIMICE-specific logic.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Job completion message
        """
        # Call the base implementation
        await super()._handle_job_completed(client, websocket, message)
        
        # Add SIMICE-specific completion handling
        await client.send_status("SIMICE job completed successfully")
        
        # Check if imputation results are available
        result_path = message.get("result_path")
        if result_path:
            await client.send_status(f"Imputation results available at: {result_path}")
    
    async def handle_message(self, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a message from the central site.
        
        Args:
            message_type: Type of the message
            payload: Message payload
            
        Returns:
            Response payload (if any)
        """
        try:
            # Process the message using the algorithm instance
            if message_type == "compute_statistics":
                target_col_idx = payload.get("target_col_idx", 0)
                method = payload.get("method", "gaussian")
                return await self.algorithm_instance.compute_local_statistics(target_col_idx, method)
                
            elif message_type == "update_imputations":
                target_col_idx = payload.get("target_col_idx", 0)
                global_params = payload.get("global_parameters", {})
                return await self.algorithm_instance.update_imputations(target_col_idx, global_params)
                
            elif message_type == "get_final_data":
                return await self.algorithm_instance.get_final_imputed_data()
                
            else:
                return await self.algorithm_instance.process_message(message_type, payload)
                
        except Exception as e:
            print(f"Error in handle_message: {str(e)}")
            return {"error": str(e)}
    
    async def process_method_message(self, method: str) -> None:
        """
        Process a method message from the central site.
        
        Args:
            method: Method to use for the algorithm (not used in SIMICE as it handles multiple methods)
        """
        print(f"🎯 SIMICE: Received method message: {method} (SIMICE handles multiple methods automatically)")
        # SIMICE doesn't need to set a single method as it handles multiple target columns with different methods


# Entry point for running as a module
if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    
    from common.algorithm.base import RemoteAlgorithm
    
    class SimpleSIMICEAlgorithm(RemoteAlgorithm):
        """Simple SIMICE algorithm implementation for testing."""
        
        def __init__(self):
            self.initial_data = None
            self.missing_variables = []  # Initialize as empty list
            self.data = None
            self.target_columns = []
            self.initialized = False  # Track initialization state
            print(f"🔧 SimpleSIMICE: Algorithm instance created")
        
        @classmethod
        def get_algorithm_name(cls) -> str:
            return "SIMICE"
        
        @classmethod 
        def get_supported_methods(cls) -> List[str]:
            return ["gaussian", "logistic"]
            
        async def prepare_data(self, data: np.ndarray, target_column_indexes: List[int], 
                             is_binary: List[bool]) -> Dict[str, Any]:
            """Prepare data for SIMICE."""
            self.data = data
            self.target_columns = target_column_indexes.copy() if target_column_indexes else []
            print(f"🔧 SimpleSIMICE: Prepared data with shape {data.shape}, target columns: {target_column_indexes}")
            return {
                "data_shape": data.shape,
                "target_columns": target_column_indexes,
                "is_binary": is_binary
            }
            
        async def initialize_imputation(self, missing_variables: List[int]) -> None:
            """Initialize imputation for missing variables."""
            print(f"🔧 SimpleSIMICE: initialize_imputation called with missing_variables: {missing_variables}")
            self.missing_variables = missing_variables.copy() if missing_variables else []  # Make a copy
            self.initialized = True
            print(f"✅ SimpleSIMICE: Successfully set self.missing_variables = {self.missing_variables}")
            print(f"✅ SimpleSIMICE: Initialization complete, initialized = {self.initialized}")
            
        async def compute_local_statistics(self, target_column_index: int, method: str) -> Dict[str, Any]:
            """Compute local statistics for a target column."""
            print(f"📊 SimpleSIMICE: Computing statistics for column {target_column_index} using {method}")
            print(f"🔍 SimpleSIMICE DEBUG: self.missing_variables = {getattr(self, 'missing_variables', 'NOT SET')}")
            print(f"🔍 SimpleSIMICE DEBUG: self.initialized = {getattr(self, 'initialized', 'NOT SET')}")
            print(f"🔍 SimpleSIMICE DEBUG: target_column_index = {target_column_index}")
            print(f"🔍 SimpleSIMICE DEBUG: data is None = {self.data is None}")
            print(f"🔍 SimpleSIMICE DEBUG: target_columns = {getattr(self, 'target_columns', 'NOT SET')}")
            
            if self.data is None:
                print(f"❌ SimpleSIMICE: Data not initialized")
                return {"error": f"Data not initialized"}
                
            if not hasattr(self, 'missing_variables') or self.missing_variables is None:
                print(f"❌ SimpleSIMICE: missing_variables not set")
                return {"error": f"Missing variables not initialized - call initialize_imputation first"}
                
            if not hasattr(self, 'initialized') or not self.initialized:
                print(f"❌ SimpleSIMICE: Algorithm not initialized")
                return {"error": f"Algorithm not initialized - call initialize_imputation first"}
                
            if target_column_index not in self.missing_variables:
                print(f"❌ SimpleSIMICE: Target column {target_column_index} not in missing_variables {self.missing_variables}")
                # Let's also check if it's an off-by-one error
                if (target_column_index - 1) in self.missing_variables:
                    print(f"🔍 SimpleSIMICE: Found {target_column_index - 1} in missing_variables - possible indexing issue")
                if (target_column_index + 1) in self.missing_variables:
                    print(f"🔍 SimpleSIMICE: Found {target_column_index + 1} in missing_variables - possible indexing issue")
                return {"error": f"Target column {target_column_index} not initialized"}
                
            # Simple mock statistics
            stats = {
                "n": self.data.shape[0],
                "mean": 0.5,
                "variance": 0.25,
                "target_column": target_column_index,
                "method": method
            }
            
            print(f"✅ SimpleSIMICE: Successfully computed statistics: {stats}")
            return stats
            
        async def update_imputations(self, target_column_index: int, global_params: Dict[str, Any]) -> None:
            """Update imputations based on global parameters.""" 
            print(f"🔄 SimpleSIMICE: Updating imputations for column {target_column_index}")
            
        async def process_message(self, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            """Process message from central."""
            print(f"📨 SimpleSIMICE: Processing message type {message_type}")
            return {"status": "processed"}
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SIMICE client')
    parser.add_argument('--data-file', required=True, help='Path to the data file')
    parser.add_argument('--job-id', required=True, type=int, help='Job ID')
    parser.add_argument('--site-id', required=True, help='Site ID')
    parser.add_argument('--target-column', required=True, type=int, help='Target column index (0-based)')
    parser.add_argument('--central-url', required=True, help='Central server URL')
    parser.add_argument('--token', required=True, help='Authentication token')
    
    args = parser.parse_args()
    
    async def main():
        """Main function to run the SIMICE client."""
        try:
            # Load data
            import pandas as pd
            data_df = pd.read_csv(args.data_file)
            data = data_df.values
            
            print(f"📊 SIMICE Client: Loaded data with shape {data.shape}")
            print(f"🎯 SIMICE Client: Target column: {args.target_column}")
            print(f"🔗 SIMICE Client: Connecting to {args.central_url}")
            print(f"🆔 SIMICE Client: Site ID: {args.site_id}")
            
            # Create client instance
            client = SIMICEClient(SimpleSIMICEAlgorithm)
            
            # Run the algorithm
            await client.run_algorithm(
                data=data,
                target_column=args.target_column,
                job_id=args.job_id,
                site_id=args.site_id,
                central_url=args.central_url,
                token=args.token
            )
            
        except Exception as e:
            print(f"❌ SIMICE Client Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Run the main function
    asyncio.run(main())