"""
Job status tracking for the central server.
This module provides a way to track job status and messages.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

class JobStatus:
    """Status of a job including messages and state"""
    def __init__(self, job_id: int):
        self.job_id = job_id
        self.status = "Starting"
        self.messages: List[str] = ["Job started"]
        self.start_time = datetime.now()
        self.completed = False
        self.error: Optional[str] = None
        self.result: Any = None
        
    def add_message(self, message: str):
        """Add a status message"""
        self.messages.append(message)
        self.status = message
        
    def complete(self, result=None):
        """Mark the job as completed"""
        self.completed = True
        self.status = "Completed"
        self.messages.append("Job completed successfully")
        self.result = result
        
    def fail(self, error: str):
        """Mark the job as failed"""
        self.completed = True
        self.status = f"Failed: {error}"
        self.error = error
        self.messages.append(f"Error: {error}")
        
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "messages": self.messages,
            "start_time": self.start_time.isoformat(),
            "completed": self.completed,
            "error": self.error,
            "result": self.result
        }

class JobStatusTracker:
    """Singleton to track all job statuses"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobStatusTracker, cls).__new__(cls)
            cls._instance.jobs = {}  # Dict[int, JobStatus]
        return cls._instance
    
    def start_job(self, job_id: int) -> JobStatus:
        """Start tracking a new job"""
        job_status = JobStatus(job_id)
        self.jobs[job_id] = job_status
        return job_status
    
    def get_job_status(self, job_id: int) -> Optional[JobStatus]:
        """Get the status of a job"""
        return self.jobs.get(job_id)
    
    def add_message(self, job_id: int, message: str):
        """Add a message to a job's status"""
        job = self.get_job_status(job_id)
        if job:
            job.add_message(message)
    
    def complete_job(self, job_id: int, result=None):
        """Mark a job as completed"""
        job = self.get_job_status(job_id)
        if job:
            job.complete(result)
    
    def fail_job(self, job_id: int, error: str):
        """Mark a job as failed"""
        job = self.get_job_status(job_id)
        if job:
            job.fail(error)
    
    def list_jobs(self):
        """List all tracked jobs"""
        return self.jobs
    
    def get_first_running_job_id(self):
        """Get the ID of the first running job, or None if no jobs are running"""
        for job_id, job in self.jobs.items():
            if not job.completed:
                return job_id
        return None
