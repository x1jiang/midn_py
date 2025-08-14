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
from common.algorithm.protocol import create_message, parse_message, MessageType

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
    
    @abstractmethod
    async def handle_site_message(self, site_id: str, message: str) -> None:
        """
        Handle a message from a remote site.
        
        Args:
            site_id: ID of the remote site
            message: Message content
        """
        pass
    
    async def send_to_all_sites(self, job_id: int, message_type: MessageType, **payload) -> None:
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
