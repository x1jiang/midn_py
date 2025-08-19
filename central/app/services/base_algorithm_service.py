"""
Base algorithm service for central site.
Provides core functionality for managing algorithm jobs.
"""

from abc import ABC, abstractmethod
import asyncio
import json
import pandas as pd
from typing import Dict, Any, List, Set, Optional, Type

from common.algorithm.base import CentralAlgorithm
from common.algorithm.job_protocol import create_message, parse_message, ProtocolMessageType

from ..websockets.connection_manager import ConnectionManager
from .. import models
from .job_status import JobStatusTracker


class BaseAlgorithmService(ABC):
    """
    Base service for all algorithm implementations on the central site.
    Handles common aspects of job management and communication.
    """
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.jobs: Dict[int, Dict[str, Any]] = {}
        self.site_to_job: Dict[str, int] = {}
        self.job_status_tracker = JobStatusTracker()
        
    @abstractmethod
    async def start_job(self, db_job: models.Job, central_data: pd.DataFrame) -> None:
        """
        Start a new algorithm job.
        
        Args:
            db_job: Job database record
            central_data: Data from the central site
        """
        pass
    
    async def wait_for_participants(self, job_id: int) -> None:
        """
        Wait for all expected participants to connect.
        
        Args:
            job_id: ID of the job
        """
        job_info = self.jobs[job_id]
        expected_participants = set(job_info["participants"])
        
        while not expected_participants.issubset(set(job_info["connected_sites"])):
            await asyncio.sleep(1)
    
    async def handle_site_message(self, site_id: str, message: str) -> None:
        """
        Handle a message from a remote site.
        
        Args:
            site_id: ID of the remote site
            message: Message content
        """
        try:
            # Parse the message to check for job conflicts
            message_data = parse_message(message)
            message_type = message_data.get('type')
            job_id = message_data.get('job_id')
            
            # Check for concurrent jobs if this is a connect message
            if message_type == "connect":
                # Check if there's already a running job
                running_job_id = self.job_status_tracker.get_first_running_job_id()
                if running_job_id and running_job_id != job_id:
                    # There's a DIFFERENT job running, send error response
                    error_message = create_message(
                        ProtocolMessageType.ERROR,
                        message=f"Job {running_job_id} is already running. Please wait for completion.",
                        code="JOB_CONFLICT"
                    )
                    await self.manager.send_to_site(error_message, site_id)
                    print(f"❌ Rejected connection from {site_id}: Job {running_job_id} is already running (requested job {job_id})")
                    return
                elif running_job_id and running_job_id == job_id:
                    # The SAME job is running, allow the connection (participant joining their job)
                    print(f"✅ Allowing connection from {site_id}: joining their assigned job {job_id}")
            
            # Handle the message (to be implemented by subclasses)
            await self._handle_site_message_impl(site_id, message)
            
        except Exception as e:
            print(f"💥 Error handling message from site {site_id}: {e}")
            # Send error response to the site
            error_message = create_message(
                ProtocolMessageType.ERROR,
                message=f"Error processing message: {str(e)}"
            )
            await self.manager.send_to_site(error_message, site_id)
    
    @abstractmethod
    async def _handle_site_message_impl(self, site_id: str, message: str) -> None:
        """
        Implementation of site message handling (to be overridden by subclasses).
        
        Args:
            site_id: ID of the remote site
            message: Message content
        """
        pass
    
    async def send_to_all_sites(self, job_id: int, message_type: ProtocolMessageType, **payload) -> None:
        """
        Send a message to all sites participating in a job.
        
        Args:
            job_id: ID of the job
            message_type: Type of message to send
            **payload: Message payload
        """
        job_info = self.jobs[job_id]
        message = create_message(message_type, **payload)
        
        send_tasks = [
            self.manager.send_to_site(message, site_id)
            for site_id in job_info["connected_sites"]
        ]
        
        await asyncio.gather(*send_tasks)
    
    def add_status_message(self, job_id: int, message: str) -> None:
        """
        Add a status message for a job.
        
        Args:
            job_id: ID of the job
            message: Status message
        """
        self.job_status_tracker.add_message(job_id, message)
