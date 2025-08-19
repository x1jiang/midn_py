"""
Federated Job Protocol Service for central site.
Provides standardized job management communication protocol for federated learning algorithms.
"""

import asyncio
import json
import pandas as pd
from typing import Dict, Any, List, Set, Optional, Type
from abc import abstractmethod

from common.algorithm.job_protocol import (
    Protocol, JobStatus, RemoteStatus, ProtocolMessageType, ErrorCode,
    create_message, parse_message
)
from .base_algorithm_service import BaseAlgorithmService
from .. import models
from ..websockets.connection_manager import ConnectionManager


class FederatedJobProtocolService(BaseAlgorithmService):
    """
    Standardized service for job management protocol in federated learning algorithms.
    Handles job connections, status transitions, and completion notifications.
    Algorithm-specific data exchange is delegated to subclasses.
    """
    
    def __init__(self, manager: ConnectionManager):
        """
        Initialize the federated job protocol service.
        
        Args:
            manager: WebSocket connection manager
        """
        super().__init__(manager)
        self.site_to_job = {}  # Map site IDs to job IDs for tracking connections
    
    async def _handle_site_message_impl(self, site_id: str, message_str: str) -> None:
        """
        Implementation of site message handling for job management protocol.
        
        Args:
            site_id: ID of the remote site
            message_str: Message content as string
        """
        # Parse the message
        data = json.loads(message_str)
        job_id = data.get("job_id")
        message_type = data.get("type")
        
        print(f"🔍 Protocol: Received message from site {site_id}: type={message_type}, job_id={job_id}")
        
        # Handle job management protocol messages
        if message_type == ProtocolMessageType.CONNECT.value:
            await self._handle_connect(site_id, data)
            return
        elif message_type == ProtocolMessageType.SITE_READY.value:
            await self._handle_site_ready(site_id, data)
            return
        elif message_type == ProtocolMessageType.COMPLETION_ACK.value:
            await self._handle_completion_ack(site_id, data)
            return
        elif message_type == ProtocolMessageType.HEARTBEAT.value:
            await self._handle_heartbeat(site_id, data)
            return
        
        # Resolve job_id via mapping if absent
        if job_id is None:
            job_id = self.site_to_job.get(site_id)
            if job_id is None:
                print(f"❌ Protocol: No job_id provided and no mapping for site {site_id}")
                error_message = create_message(
                    ProtocolMessageType.ERROR,
                    message="No job_id specified and no active connection",
                    code="UNKNOWN_CONNECTION"
                )
                await self.manager.send_to_site(error_message, site_id)
                return
        
        if job_id not in self.jobs:
            print(f"❌ Protocol: Job {job_id} not found for site {site_id}")
            error_message = create_message(
                ProtocolMessageType.ERROR,
                message=f"Job {job_id} is not currently running or does not exist",
                code="JOB_NOT_FOUND"
            )
            await self.manager.send_to_site(error_message, site_id)
            return
        
        # Delegate algorithm-specific messages to subclasses
        await self._handle_algorithm_message(site_id, job_id, message_type, data)
    
    async def _handle_connect(self, site_id: str, data: Dict[str, Any]) -> None:
        """
        Handle a connection request from a remote site.
        
        Args:
            site_id: ID of the remote site
            data: Message data
        """
        job_id = data.get("job_id")
        
        # Check if there are no jobs available at all
        if not self.jobs:
            print(f"❌ Protocol: No jobs available for site {site_id}")
            error_message = create_message(
                ProtocolMessageType.ERROR,
                message=f"No {self.get_algorithm_name()} jobs are currently running. Please try again later.",
                code=ErrorCode.NO_JOBS_AVAILABLE.value
            )
            await self.manager.send_to_site(error_message, site_id)
            print(f"📤 Protocol: Sent 'no jobs available' error to site {site_id}")
            return
        
        # Check if the specific job exists
        if not job_id:
            print(f"❌ Protocol: No job_id provided by site {site_id}")
            error_message = create_message(
                ProtocolMessageType.ERROR,
                message="No job_id specified in connection request",
                code=ErrorCode.MISSING_JOB_ID.value
            )
            await self.manager.send_to_site(error_message, site_id)
            print(f"📤 Protocol: Sent 'missing job_id' error to site {site_id}")
            return
        
        if job_id not in self.jobs:
            print(f"❌ Protocol: Job {job_id} not found for site {site_id}")
            print(f"📋 Protocol: Available jobs: {list(self.jobs.keys())}")
            error_message = create_message(
                ProtocolMessageType.ERROR,
                message=f"Job {job_id} is not currently running or does not exist",
                code=ErrorCode.JOB_NOT_FOUND.value,
                available_jobs=list(self.jobs.keys())
            )
            await self.manager.send_to_site(error_message, site_id)
            print(f"📤 Protocol: Sent 'job not found' error to site {site_id}")
            return
        
        job = self.jobs[job_id]
        
        # Check if site is authorized for this job
        if site_id not in job["participants"]:
            print(f"❌ Protocol: Site {site_id} not in participants list for job {job_id}")
            print(f"📋 Protocol: Expected participants: {job['participants']}")
            error_message = create_message(
                ProtocolMessageType.ERROR,
                message=f"Site {site_id} is not authorized for job {job_id}",
                code=ErrorCode.UNAUTHORIZED_SITE.value
            )
            await self.manager.send_to_site(error_message, site_id)
            print(f"📤 Protocol: Sent 'unauthorized site' error to site {site_id}")
            return
        
        # Check for duplicate connections (site already connected to this job)
        if site_id in job["connected_sites"]:
            print(f"⚠️ Protocol: Site {site_id} already connected to job {job_id}")
            # Instead of error, send current job status update (might be reconnection)
            status_message = create_message(
                ProtocolMessageType.CONNECTION_CONFIRMED.value,
                job_id=job_id,
                status=job["status"],
                algorithm=self.get_algorithm_name()
            )
            await self.manager.send_to_site(status_message, site_id)
            return
        
        # Accept the connection
        job["connected_sites"].append(site_id)
        self.site_to_job[site_id] = job_id
        self.add_status_message(job_id, f"Site {site_id} connected ({len(job['connected_sites'])}/{len(job['participants'])} connected)")
        
        # Send connection confirmation with job parameters
        confirm_message = create_message(
            ProtocolMessageType.CONNECTION_CONFIRMED.value,
            job_id=job_id,
            status=job["status"],
            algorithm=self.get_algorithm_name()
        )
        await self.manager.send_to_site(confirm_message, site_id)
        
        # Let subclasses handle algorithm-specific connection logic
        await self._handle_algorithm_connection(site_id, job_id)
    
    async def _handle_site_ready(self, site_id: str, data: Dict[str, Any]) -> None:
        """
        Handle a site_ready message from a remote site.
        
        Args:
            site_id: ID of the remote site
            data: Message data
        """
        job_id = data.get("job_id") or self.site_to_job.get(site_id)
        if not job_id or job_id not in self.jobs:
            print(f"❌ Protocol: Cannot process site_ready - Job not found for site {site_id}")
            return
        
        job = self.jobs[job_id]
        
        # Log the site as ready
        if "ready_sites" not in job:
            job["ready_sites"] = []
        
        if site_id not in job["ready_sites"]:
            job["ready_sites"].append(site_id)
            self.add_status_message(job_id, f"Site {site_id} ready for computation ({len(job['ready_sites'])}/{len(job['participants'])} ready)")
        
        # If job is still in waiting status and all sites are ready, 
        # send start computation message to all sites
        if job["status"] == JobStatus.WAITING.value and set(job["ready_sites"]) == set(job["participants"]):
            self.add_status_message(job_id, f"All sites ready, starting computation")
            job["status"] = JobStatus.ACTIVE.value
            
            # Send start computation message to all sites
            start_message = create_message(
                ProtocolMessageType.START_COMPUTATION.value,
                job_id=job_id
            )
            
            for site in job["connected_sites"]:
                await self.manager.send_to_site(start_message, site)
            
            # Let subclasses handle algorithm-specific start logic
            await self._handle_algorithm_start(job_id)
    
    async def _handle_completion_ack(self, site_id: str, data: Dict[str, Any]) -> None:
        """
        Handle a completion acknowledgment from a remote site.
        
        Args:
            site_id: ID of the remote site
            data: Message data
        """
        job_id = data.get("job_id") or self.site_to_job.get(site_id)
        if not job_id:
            print(f"❌ Protocol: Cannot process completion_ack - Job not found for site {site_id}")
            return
        
        # Job might be already removed from the jobs dictionary if cleanup happened
        if job_id in self.jobs:
            job = self.jobs[job_id]
            
            # Track site acknowledgments
            if "completion_acks" not in job:
                job["completion_acks"] = []
            
            if site_id not in job["completion_acks"]:
                job["completion_acks"].append(site_id)
                print(f"✅ Protocol: Site {site_id} acknowledged completion of job {job_id}")
                self.add_status_message(job_id, f"Site {site_id} acknowledged job completion")
                
            # Check if all sites have acknowledged
            if set(job["completion_acks"]) == set(job["connected_sites"]):
                print(f"🎉 Protocol: All sites acknowledged completion for job {job_id}")
                self.add_status_message(job_id, "All sites acknowledged job completion")
                
                # Clean up job data after all sites have acknowledged
                if job_id in self.jobs:
                    self.jobs.pop(job_id, None)
                    print(f"🧹 Protocol: Cleaned up job {job_id} after all acknowledgments received")
    
    async def _handle_heartbeat(self, site_id: str, data: Dict[str, Any]) -> None:
        """
        Handle heartbeat message from a remote site.
        
        Args:
            site_id: ID of the remote site
            data: Message data
        """
        job_id = data.get("job_id") or self.site_to_job.get(site_id)
        if not job_id or job_id not in self.jobs:
            print(f"❌ Protocol: Cannot process heartbeat - Job not found for site {site_id}")
            return
        
        # Update last heartbeat time
        job = self.jobs[job_id]
        if "heartbeats" not in job:
            job["heartbeats"] = {}
        
        job["heartbeats"][site_id] = asyncio.get_event_loop().time()
        
        # Respond with heartbeat ack if requested
        if data.get("require_ack", False):
            ack_message = create_message(
                ProtocolMessageType.HEARTBEAT_ACK.value,
                job_id=job_id,
                timestamp=job["heartbeats"][site_id]
            )
            await self.manager.send_to_site(ack_message, site_id)
    
    async def notify_job_completed(self, job_id: int, result_path: Optional[str] = None, message: Optional[str] = None) -> None:
        """
        Send job completion notification to all sites.
        
        Args:
            job_id: ID of the job
            result_path: Path to the result file (optional)
            message: Custom completion message (optional)
        """
        if job_id not in self.jobs:
            print(f"❌ Protocol: Job {job_id} not found - cannot send completion notification")
            return
        
        job = self.jobs[job_id]
        job["status"] = JobStatus.COMPLETED.value
        
        # Create completion message
        completion_message = create_message(
            ProtocolMessageType.JOB_COMPLETED.value,
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            message=message or f"{self.get_algorithm_name()} job completed successfully",
            result_path=result_path
        )
        
        # Send to all sites
        print(f"📤 Protocol: Notifying {len(job['connected_sites'])} sites of job {job_id} completion")
        for site_id in job["connected_sites"]:
            try:
                await self.manager.send_to_site(completion_message, site_id)
                print(f"✅ Protocol: Sent completion notification to site {site_id}")
            except Exception as e:
                print(f"💥 Protocol: Error sending completion to site {site_id}: {e}")
    
    async def initialize_job_state(self, job_id: int, db_job: models.Job, initial_status: str = None) -> None:
        """
        Initialize job state with standardized structure.
        
        Args:
            job_id: ID of the job
            db_job: Job database record
            initial_status: Initial job status (default: JobStatus.WAITING.value)
        """
        if initial_status is None:
            initial_status = JobStatus.WAITING.value
            
        self.jobs[job_id] = {
            "status": initial_status,
            "participants": list(set(db_job.participants or [])),
            "connected_sites": [],
            "ready_sites": [],
            "creation_time": asyncio.get_event_loop().time(),
            "parameters": db_job.parameters
        }
        
        # Let subclasses add algorithm-specific state
        self._initialize_algorithm_state(job_id, db_job)
    
    def _initialize_algorithm_state(self, job_id: int, db_job: models.Job) -> None:
        """
        Initialize algorithm-specific state for a job.
        To be overridden by subclasses.
        
        Args:
            job_id: ID of the job
            db_job: Job database record
        """
        pass
    
    async def _handle_algorithm_message(self, site_id: str, job_id: int, message_type: str, data: Dict[str, Any]) -> None:
        """
        Handle algorithm-specific messages.
        To be implemented by subclasses.
        
        Args:
            site_id: ID of the remote site
            job_id: ID of the job
            message_type: Type of the message
            data: Message data
        """
        raise NotImplementedError("Subclasses must implement _handle_algorithm_message")
    
    async def _handle_algorithm_connection(self, site_id: str, job_id: int) -> None:
        """
        Handle algorithm-specific connection logic.
        To be overridden by subclasses.
        
        Args:
            site_id: ID of the remote site
            job_id: ID of the job
        """
        pass
    
    async def _handle_algorithm_start(self, job_id: int) -> None:
        """
        Handle algorithm-specific start logic when all sites are ready.
        To be overridden by subclasses.
        
        Args:
            job_id: ID of the job
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """
        Get the name of the algorithm.
        Must be implemented by subclasses.
        
        Returns:
            Algorithm name as string
        """
        pass
