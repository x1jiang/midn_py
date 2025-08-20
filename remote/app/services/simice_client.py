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
        Run the SIMICE algorithm.
        
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
            message = await client.receive_message(websocket)
            if not message:
                await client.send_status("Connection lost while waiting for Initialize message")
                return
                
            message_type = message.get("message_type") or message.get("type")
            instruction = message.get("instruction", "")
            
            if instruction == "Initialize":
                # Initialize message received (R-style)
                await client.send_status("Received Initialize message")
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
            # Prepare the data for SIMICE
            simice_data = await self.algorithm_instance.prepare_data(
                data, 
                self.target_column_indexes, 
                self.is_binary
            )
            
            # Handle the first message appropriately
            if instruction == "Initialize":
                # Process R-style Initialize message
                await self._handle_initialize_message(client, websocket, message, job_id)
            else:
                # Handle legacy data summary request
                await client.send_status("Sending data summary to central...")
                
                # Create data summary message
                data_summary_message = {
                    "type": ProtocolMessageType.DATA_SUMMARY.value,
                    "job_id": job_id,
                    "data_summary": simice_data
                }
                
                if not await client.send_message_dict(websocket, data_summary_message):
                    await client.send_status("Failed to send data summary")
                    return

                await client.send_status("Data summary sent, waiting for instructions...")
            
            # Main SIMICE protocol loop
            while True:
                # Wait for instructions from central
                message = await client.receive_message(websocket)
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
        Process an algorithm-specific message following R SIMICE implementation.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message to process
            
        Returns:
            True to continue processing, False to exit processing loop
        """
        job_id = self.job_state.get("job_id")
        message_type = message.get("type")
        instruction = message.get("instruction", "")
        
        await client.send_status(f"Processing SIMICE message: {instruction or message_type}")
        
        try:
            # Handle R-style instructions
            if instruction == "Initialize":
                await self._handle_initialize_message(client, websocket, message, job_id)
                return True
                
            elif instruction == "Information":
                await self._handle_information_message(client, websocket, message, job_id)
                return True
                
            elif instruction == "Impute":
                await self._handle_impute_message(client, websocket, message, job_id)
                return True
                
            elif instruction == "End":
                await self._handle_end_message(client, websocket, message, job_id)
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
                
            elif message_type == ProtocolMessageType.ERROR.value:
                error_msg = message.get("message", "Unknown error")
                error_code = message.get("code", "UNKNOWN_ERROR")
                await client.send_status(f"Error from central: {error_msg} (Code: {error_code})")
                return False  # Exit processing loop
            
            else:
                await client.send_status(f"Unknown message type: {message_type}, instruction: {instruction}")
                return True  # Continue processing
        
        except Exception as e:
            await client.send_status(f"Error processing message: {str(e)}")
            traceback.print_exc()
            return True  # Continue processing despite error

    # R-style message handlers
    
    async def _handle_initialize_message(self, client: ConnectionClient, websocket: Any, 
                                        message: Dict[str, Any], job_id: int) -> None:
        """
        Handle Initialize message from central (R-style).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Initialize message
            job_id: Job ID
        """
        missing_variables = message.get("missing_variables", [])
        await client.send_status(f"Received Initialize with {len(missing_variables)} missing variables")
        
        try:
            # Store missing variables for processing
            self.missing_variables = missing_variables
            
            # Prepare the algorithm instance with missing variables
            await self.algorithm_instance.initialize_imputation(missing_variables)
            
            # Send acknowledgment back to central
            response_message = {
                "type": ProtocolMessageType.METHOD.value,
                "job_id": job_id,
                "instruction": "Initialize",
                "status": "ready"
            }
            
            await client.send_message_dict(websocket, response_message)
            await client.send_status("Sent Initialize acknowledgment")
            
        except Exception as e:
            await client.send_status(f"Error in Initialize: {str(e)}")
            
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "instruction": "Initialize",
                "error": str(e)
            }
            
            await client.send_message_dict(websocket, error_message)

    async def _handle_information_message(self, client: ConnectionClient, websocket: Any, 
                                         message: Dict[str, Any], job_id: int) -> None:
        """
        Handle Information message from central (R-style).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Information message
            job_id: Job ID
        """
        target_column_index = message.get("target_column_index")
        method = message.get("method", "Gaussian")
        
        await client.send_status(f"Computing statistics for column {target_column_index} using {method}")
        
        try:
            # Compute local statistics using the algorithm
            stats = await self.algorithm_instance.compute_local_statistics(target_column_index, method)
            
            # Send statistics back to central
            response_message = {
                "type": ProtocolMessageType.DATA.value,
                "job_id": job_id,
                "instruction": "Information",
                "target_column_index": target_column_index,
                "method": method,
                "statistics": stats
            }
            
            await client.send_message_dict(websocket, response_message)
            await client.send_status(f"Sent statistics for column {target_column_index}")
            
        except Exception as e:
            await client.send_status(f"Error computing statistics: {str(e)}")
            
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "instruction": "Information",
                "target_column_index": target_column_index,
                "error": str(e)
            }
            
            await client.send_message_dict(websocket, error_message)

    async def _handle_impute_message(self, client: ConnectionClient, websocket: Any, 
                                    message: Dict[str, Any], job_id: int) -> None:
        """
        Handle Impute message from central (R-style).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Impute message
            job_id: Job ID
        """
        target_column_index = message.get("target_column_index")
        method = message.get("method", "Gaussian")
        parameters = message.get("parameters", {})
        
        await client.send_status(f"Updating imputations for column {target_column_index} using {method}")
        
        try:
            # Update local imputations using the computed parameters
            await self.algorithm_instance.update_imputations(target_column_index, parameters, method)
            
            # Send confirmation back to central
            response_message = {
                "type": ProtocolMessageType.DATA.value,
                "job_id": job_id,
                "instruction": "Impute",
                "target_column_index": target_column_index,
                "status": "completed"
            }
            
            await client.send_message_dict(websocket, response_message)
            await client.send_status(f"Imputations updated for column {target_column_index}")
            
        except Exception as e:
            await client.send_status(f"Error updating imputations: {str(e)}")
            
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "instruction": "Impute",
                "target_column_index": target_column_index,
                "error": str(e)
            }
            
            await client.send_message_dict(websocket, error_message)

    async def _handle_end_message(self, client: ConnectionClient, websocket: Any, 
                                 message: Dict[str, Any], job_id: int) -> None:
        """
        Handle End message from central (R-style).
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: End message
            job_id: Job ID
        """
        await client.send_status("Received End message - finalizing SIMICE")
        
        try:
            # Finalize the imputation process
            await self.algorithm_instance.finalize_imputation()
            
            # Send acknowledgment back to central
            response_message = {
                "type": ProtocolMessageType.METHOD.value,
                "job_id": job_id,
                "instruction": "End",
                "status": "completed"
            }
            
            await client.send_message_dict(websocket, response_message)
            await client.send_status("SIMICE algorithm completed")
            
        except Exception as e:
            await client.send_status(f"Error in End: {str(e)}")
            
            error_message = {
                "type": ProtocolMessageType.ERROR.value,
                "job_id": job_id,
                "instruction": "End",
                "error": str(e)
            }
            
            await client.send_message_dict(websocket, error_message)
    
    # Legacy message handlers (for backward compatibility)
    
    async def _handle_compute_statistics(self, client: ConnectionClient, websocket: Any, 
                                        message: Dict[str, Any], job_id: int) -> None:
        """
        Handle compute statistics request.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message with compute statistics request
            job_id: Job ID
        """
        target_col_idx = message.get("target_col_idx", 0)  # 0-based index
        method = message.get("method", "gaussian")
        
        await client.send_status(f"Computing statistics for column {target_col_idx} using {method}")
        
        try:
            # Compute local statistics using the algorithm
            stats = await self.algorithm_instance.compute_local_statistics(target_col_idx, method)
            
            # Send statistics back to central
            await client.send_status(f"Sending statistics for column {target_col_idx}")
            
            statistics_message = {
                "type": ProtocolMessageType.STATISTICS.value,
                "job_id": job_id,
                "target_col_idx": target_col_idx,
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
                "target_col_idx": target_col_idx,
                "error": str(e)
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
        Handle request for final imputed data.
        
        Args:
            client: Connection client
            websocket: WebSocket connection
            message: Message with get final data request
            job_id: Job ID
        """
        await client.send_status(f"Collecting final imputed data")
        
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
            
            # Send the final data back to central
            final_data_message = {
                "type": "final_data",
                "job_id": job_id,
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
                "error": str(e)
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
