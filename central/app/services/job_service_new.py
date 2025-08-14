"""
Updated job service to use the new algorithm architecture.
"""

import pandas as pd
from sqlalchemy.orm import Session

from .. import models
from ..websockets.connection_manager import ConnectionManager
from ..services.algorithm_factory import AlgorithmServiceFactory

# Import service initialization to ensure all algorithms and services are registered
import central.app.services.init_services


class JobServiceNew:
    """
    Service for managing algorithm jobs.
    """
    
    def __init__(self, db: Session, manager: ConnectionManager):
        """
        Initialize job service.
        
        Args:
            db: Database session
            manager: WebSocket connection manager
        """
        self.db = db
        self.manager = manager
    
    async def start_job(self, job_id: int, central_data_path: str = None) -> None:
        """
        Start a job.
        
        Args:
            job_id: ID of the job
            central_data_path: Path to central data file
        """
        # Get the job from the database
        job = self.db.query(models.Job).filter(models.Job.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Check if the job is already running
        if job.status == "running":
            raise ValueError(f"Job {job_id} is already running")
        
        # Update job status
        job.status = "running"
        self.db.commit()
        
        # Load central data if provided
        central_data = None
        if central_data_path:
            central_data = pd.read_csv(central_data_path)
        
        # Get the algorithm service
        algorithm = job.algorithm
        service = AlgorithmServiceFactory.create_service(algorithm, self.manager)
        
        # Start the job
        await service.start_job(job, central_data)
