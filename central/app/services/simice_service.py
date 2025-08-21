import asyncio
import json
import os
import time
import traceback
from typing import Dict, List, Any, Set, Tuple, Optional
import uuid
import numpy as np

from common.algorithm.job_protocol import Protocol, ProtocolMessageType, ErrorCode
from central.app.services.federated_job_protocol_service import FederatedJobProtocolService
from central.app.services.job_status import JobStatusTracker


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that can handle NumPy data types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class SIMICEService(FederatedJobProtocolService):
    """
    Service for managing SIMICE algorithm jobs.
    SIMICE (Sequential Imputation with Multiple Imputation using Chained Equations)
    is a distributed federated learning algorithm for handling missing data.
    
    This implementation aligns with the standardized federated job protocol.
    """
    
    def __init__(self, manager):
        """
        Initialize the SIMICE service.
        
        Args:
            manager: WebSocket connection manager
        """
        super().__init__(manager)  # Initialize the parent class
        self.jobs = {}  # Jobs managed by this service
        
        print(f"🚀 SIMICE Service initialized")
    
    def add_status_message(self, job_id: int, message: str) -> None:
        """
        Add a status message to the job status tracker.
        
        Args:
            job_id: Job ID
            message: Status message
        """
        # Use the job status tracker from the parent class
        self.job_status_tracker.add_message(job_id, message)
        print(f"� SIMICE JOB {job_id}: {message}")
    
    async def initialize_job(self, job_id: int, job_config: Dict[str, Any]) -> None:
        """
        Initialize a new SIMICE job. This registers the job with the service
        and initializes tracking.
        
        Args:
            job_id: ID of the new job
            job_config: Configuration for the job
        """
        try:
            print(f"🔧 SIMICE: Initializing job {job_id}")
            print(f"📋 SIMICE: Job config: {job_config}")
            
            # Store job information
            self.jobs[job_id] = {
                'id': job_id,
                'config': job_config,
                'participants': job_config.get('participants', []),
                'algorithm_data': {
                    'current_iteration': 0,
                    'max_iterations': job_config.get('max_iterations', 10),
                    'iteration_before_first_imputation': job_config.get('iteration_before_first_imputation', 5),
                    'iteration_between_imputations': job_config.get('iteration_between_imputations', 1),
                    'target_column_indexes': job_config.get('target_column_indexes', []),
                    'is_binary': job_config.get('is_binary', []),
                    'current_target_column_index': None
                }
            }
            
            self.add_status_message(job_id, f"Job {job_id} initialized with {len(job_config.get('participants', []))} participants")
            
            # Note: We're NOT setting job['status'] directly anymore - the protocol handles this
            # The job status is tracked by job_status_tracker, not in the job dict
            
            print(f"🚀 SIMICE: Job {job_id} initialized and tracking started")
            
            # With standardized protocol, we don't need to manually wait for participants
            # The protocol will handle connection management and trigger _handle_algorithm_start when ready
            self.add_status_message(job_id, f"Job {job_id}: Waiting for all participants to connect via standardized protocol")
        except Exception as e:
            print(f"💥 SIMICE: Error initializing job {job_id}: {str(e)}")
            # If anything goes wrong during initialization, mark the job as failed
            print(f"❌ SIMICE JOB {job_id}: Job initialization error: {str(e)}")
            # Re-raise to propagate error to the API layer
            raise
    
    async def _handle_algorithm_message(self, site_id: str, job_id: int, message_type: str, data: Dict[str, Any]) -> None:
        """
        Handle algorithm-specific messages for SIMICE.
        Similar structure to SIMI for consistency, with clear separation of message types.
        
        Args:
            site_id: ID of the remote site
            job_id: ID of the job
            message_type: Type of the message
            data: Message data
        """
        print(f"🎯 SIMICE DEBUG: Processing {message_type} for job {job_id} from site {site_id}")
        print(f"📋 SIMICE DEBUG: Message data keys: {list(data.keys())}")
        print(f"📋 SIMICE DEBUG: Full message data: {data}")
        
        try:
            job = self.jobs[job_id]
            
            # === COMPUTATION-RELATED MESSAGES ===
            # All computation messages must be of type DATA with an instruction field
            
            # Check if this is a DATA message
            if message_type == ProtocolMessageType.DATA.value:
                # Extract the instruction from the DATA message
                instruction = data.get("instruction", "")
                
                print(f"📥 SIMICE: Received DATA message with instruction: {instruction} from site {site_id}")
                
                # Handle different instruction types - preserve R instruction names
                if instruction == "data_summary":
                    # Handle data summary from remote site
                    summary_data = data.get("data", {})
                    print(f"📊 SIMICE: Received data summary from site {site_id} for job {job_id}")
                    print(f"📊 SIMICE: Summary data keys: {list(summary_data.keys()) if summary_data else 'No data'}")
                    
                    # Store the summary data for later use
                    if 'site_data' not in job['algorithm_data']:
                        job['algorithm_data']['site_data'] = {}
                    job['algorithm_data']['site_data'][site_id] = summary_data
                
                # R-style instruction: "Information" response
                elif instruction == "Information_response":
                    # Handle statistics data using R naming
                    target_col_idx = data.get("yidx") or data.get("target_col_idx")
                    method = data.get("method")
                    statistics = data.get("statistics", {})
                    
                    print(f"🔥 SIMICE STATS (R-style): Site {site_id} sent Information response for column {target_col_idx}")
                    print(f"🔥 SIMICE STATS: Method={method}, n={statistics.get('n', 'unknown')}")
                    
                    # Process statistics data
                    key = f"{target_col_idx}_{method}"
                    if 'statistics' not in job['algorithm_data']:
                        job['algorithm_data']['statistics'] = {}
                    if key not in job['algorithm_data']['statistics']:
                        job['algorithm_data']['statistics'][key] = {}
                    
                    # Store statistics for this site
                    job['algorithm_data']['statistics'][key][site_id] = statistics
                    
                    # Check if we've received statistics from all sites for this variable
                    waiting_key = f'waiting_for_stats_{target_col_idx}'
                    print(f"� SIMICE WAITING: Looking for key: {waiting_key}")
                    
                    if waiting_key in job:
                        current_waiting = job[waiting_key].copy()  # Make a copy to show before/after
                        print(f"� SIMICE WAITING: Before removal: {current_waiting}")
                        
                        if site_id in job[waiting_key]:
                            job[waiting_key].remove(site_id)
                            print(f"� SIMICE WAITING: After removing {site_id}: {job[waiting_key]}")
                            print(f"🔥 SIMICE WAITING: Still waiting for {len(job[waiting_key])} sites")
                        else:
                            print(f"🔥 SIMICE ERROR: Site {site_id} NOT in waiting list {job[waiting_key]}")
                        
                        if not job[waiting_key]:
                            print(f"🔥 🎉 SIMICE COMPLETE: All sites responded for column {target_col_idx}!")
                            await self._process_aggregated_statistics(job_id, target_col_idx, method)
                        else:
                            print(f"🔥 SIMICE WAITING: Still need statistics from: {job[waiting_key]}")
                    else:
                        print(f"🔥 SIMICE ERROR: Waiting key {waiting_key} NOT FOUND in job")
                        print(f"🔥 SIMICE ERROR: Available keys: {[k for k in job.keys() if 'waiting' in k]}")
                
                # Custom instruction: "statistics" response (non-R naming)
                elif instruction == "statistics":
                    # Handle statistics data
                    target_col_idx = data.get("target_col_idx")
                    method = data.get("method")
                    statistics = data.get("statistics", {})
                    
                    print(f"🔥 SIMICE STATS: Site {site_id} sent statistics for column {target_col_idx}")
                    print(f"🔥 SIMICE STATS: Method={method}, n={statistics.get('n', 'unknown')}")
                    
                    # Process statistics data
                    key = f"{target_col_idx}_{method}"
                    if 'statistics' not in job['algorithm_data']:
                        job['algorithm_data']['statistics'] = {}
                    if key not in job['algorithm_data']['statistics']:
                        job['algorithm_data']['statistics'][key] = {}
                    
                    # Store statistics for this site
                    job['algorithm_data']['statistics'][key][site_id] = statistics
                    
                    # Check if we've received statistics from all sites for this variable
                    waiting_key = f'waiting_for_stats_{target_col_idx}'
                    print(f"� SIMICE WAITING: Looking for key: {waiting_key}")
                    
                    if waiting_key in job:
                        current_waiting = job[waiting_key].copy()  # Make a copy to show before/after
                        print(f"� SIMICE WAITING: Before removal: {current_waiting}")
                        
                        if site_id in job[waiting_key]:
                            job[waiting_key].remove(site_id)
                            print(f"� SIMICE WAITING: After removing {site_id}: {job[waiting_key]}")
                            print(f"🔥 SIMICE WAITING: Still waiting for {len(job[waiting_key])} sites")
                        else:
                            print(f"🔥 SIMICE ERROR: Site {site_id} NOT in waiting list {job[waiting_key]}")
                        
                        if not job[waiting_key]:
                            print(f"🔥 🎉 SIMICE COMPLETE: All sites responded for column {target_col_idx}!")
                            await self._process_aggregated_statistics(job_id, target_col_idx, method)
                        else:
                            print(f"🔥 SIMICE WAITING: Still need statistics from: {job[waiting_key]}")
                    else:
                        print(f"🔥 SIMICE ERROR: Waiting key {waiting_key} NOT FOUND in job")
                        print(f"🔥 SIMICE ERROR: Available keys: {[k for k in job.keys() if 'waiting' in k]}")
                
                # R-style instruction: "Impute_response"
                elif instruction == "Impute_response":
                    # Handle imputation update confirmation using R naming
                    target_col_idx = data.get("yidx") or data.get("target_col_idx")
                    status = data.get("status", "completed")
                    
                    print(f"✅ SIMICE: Site {site_id} completed Impute response for column {target_col_idx} ({status})")
                    
                    # Track sites that have completed imputation update
                    if 'waiting_for_updates' not in job:
                        job['waiting_for_updates'] = set()
                        
                    if site_id in job.get('waiting_for_updates', set()):
                        job['waiting_for_updates'].discard(site_id)
                        
                        # If all sites have updated, continue to next step
                        if len(job['waiting_for_updates']) == 0:
                            print(f"🔄 SIMICE: All sites updated, continuing iteration...")
                            await self._continue_simice_iteration(job_id)
                
                # Custom instruction: "imputation_updated" (non-R naming)
                elif instruction == "imputation_updated":
                    # Handle imputation update confirmation
                    target_col_idx = data.get("target_col_idx")
                    status = data.get("status")
                    
                    print(f"✅ SIMICE: Site {site_id} completed imputation update for column {target_col_idx} ({status})")
                    
                    # Track sites that have completed imputation update
                    if 'waiting_for_updates' not in job:
                        job['waiting_for_updates'] = set()
                        
                    if site_id in job.get('waiting_for_updates', set()):
                        job['waiting_for_updates'].discard(site_id)
                        
                        # If all sites have updated, continue to next step
                        if len(job['waiting_for_updates']) == 0:
                            print(f"🔄 SIMICE: All sites updated, continuing iteration...")
                            await self._continue_simice_iteration(job_id)
                
                # === INSTRUCTION-RELATED MESSAGES ===
                
                # Handle instruction responses
                elif instruction == "initialize":
                    # Response to initialization
                    print(f"🔄 SIMICE: Site {site_id} initialized")
                    self.add_status_message(job_id, f"Site {site_id} initialized successfully")
                    
                    # Mark site as ready for iterations
                    if 'ready_sites' not in job['algorithm_data']:
                        job['algorithm_data']['ready_sites'] = set()
                    
                    job['algorithm_data']['ready_sites'].add(site_id)
                    
                    # Check if all sites are initialized
                    if len(job['algorithm_data']['ready_sites']) >= len(job['participants']):
                        print(f"✅ SIMICE: All sites initialized")
                        await self._start_simice_algorithm(job_id)
                
                # R-style: "Initialize_response"
                elif instruction == "Initialize_response":
                    # Response to initialization using R-style naming
                    print(f"🔄 SIMICE: Site {site_id} initialized (R-style response)")
                    self.add_status_message(job_id, f"Site {site_id} initialized successfully (R-style)")
                    
                    # Mark site as ready for iterations
                    if 'ready_sites' not in job['algorithm_data']:
                        job['algorithm_data']['ready_sites'] = set()
                    
                    job['algorithm_data']['ready_sites'].add(site_id)
                    
                    # Check if all sites are initialized
                    if len(job['algorithm_data']['ready_sites']) >= len(job['participants']):
                        print(f"✅ SIMICE: All sites initialized")
                        await self._start_simice_algorithm(job_id)
                
                elif instruction == "end":
                    # Response to end instruction
                    print(f"🏁 SIMICE: Site {site_id} acknowledged end of job")
                    self.add_status_message(job_id, f"Site {site_id} completed algorithm execution")
                    
                    # Mark site as done
                    if 'completed_sites' not in job['algorithm_data']:
                        job['algorithm_data']['completed_sites'] = set()
                    
                    job['algorithm_data']['completed_sites'].add(site_id)
                    
                    # Check if all sites are done
                    if len(job['algorithm_data']['completed_sites']) >= len(job['participants']):
                        print(f"🎉 SIMICE: All sites completed")
                        # Make sure the job status is properly updated in the tracker
                        print(f"✅ SIMICE JOB {job_id}: All sites completed successfully")
                        await self._save_final_results(job_id)
                
                # R-style: "End_response"
                elif instruction == "End_response":
                    # Response to end instruction using R-style naming
                    print(f"🏁 SIMICE: Site {site_id} acknowledged end of job (R-style response)")
                    self.add_status_message(job_id, f"Site {site_id} completed algorithm execution (R-style)")
                    
                    # Mark site as done
                    if 'completed_sites' not in job['algorithm_data']:
                        job['algorithm_data']['completed_sites'] = set()
                    
                    job['algorithm_data']['completed_sites'].add(site_id)
                    
                    # Check if all sites are done
                    if len(job['algorithm_data']['completed_sites']) >= len(job['participants']):
                        print(f"🎉 SIMICE: All sites completed")
                        # Make sure the job status is properly updated in the tracker
                        print(f"✅ SIMICE JOB {job_id}: All sites completed successfully")
                        await self._save_final_results(job_id)
                
                else:
                    print(f"⚠️ SIMICE: Unknown instruction response from site {site_id}: {instruction}")
            
            # Legacy message types for backward compatibility
            # Handle imputation update confirmations
            elif message_type == "imputation_updated":
                print(f"✅ SIMICE: Site {site_id} completed imputation update")
                
                target_col_idx = data.get("target_col_idx")
                status = data.get("status")
                
                self.add_status_message(job_id, f"Site {site_id} completed imputation for column {target_col_idx}")
                
                # Check if all sites have completed this imputation step
                if hasattr(job, 'waiting_for_updates'):
                    job['waiting_for_updates'].discard(site_id)
                    if not job['waiting_for_updates']:
                        # All sites have completed updates, continue to next step
                        await self._continue_simice_iteration(job_id)
            
            # Handle sample size messages (similar to SIMI's "n" message)
            elif message_type == "n":
                print(f"📊 SIMICE: Received sample size from site {site_id}")
                
                n_value = float(data.get("n", 0))
                target_col_idx = data.get("target_col_idx")
                
                # Store the sample size
                if 'sample_sizes' not in job['algorithm_data']:
                    job['algorithm_data']['sample_sizes'] = {}
                
                key = f"{target_col_idx}"
                if key not in job['algorithm_data']['sample_sizes']:
                    job['algorithm_data']['sample_sizes'][key] = {}
                
                job['algorithm_data']['sample_sizes'][key][site_id] = n_value
                self.add_status_message(job_id, f"Received sample size from site {site_id} (n={n_value})")
                
                # Check if we have all sample sizes for this variable
                if all(site in job['algorithm_data']['sample_sizes'].get(key, {}) for site in job['participants']):
                    print(f"✅ SIMICE: Received all sample sizes for column {target_col_idx}")
                    # Could trigger next step if needed
            
            # Other message types
            else:
                print(f"⚠️ SIMICE: Unhandled message type from site {site_id}: {message_type}")
                
        except Exception as e:
            print(f"💥 SIMICE: Error handling algorithm message from site {site_id}: {str(e)}")
            traceback.print_exc()
