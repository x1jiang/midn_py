from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any

from ..db import get_db
from ..core.security import get_current_user, require_admin
from ..core.csrf import get_csrf_token
from ..services.job_status import JobStatusTracker

router = APIRouter()

# Get API to access job status
@router.get("/status/{job_id}")
async def get_job_status(job_id: int):
    """Get the status of a running job"""
    tracker = JobStatusTracker()
    job_status = tracker.get_job_status(job_id)
    
    if job_status:
        response = job_status.to_dict()
        
        # Double-check if there's an imputed dataset path available for completed jobs
        if job_status.completed and not response.get('imputed_dataset_path'):
            # Check the database for the imputed_dataset_path
            from ..db.database import SessionLocal
            from ..models.job import Job
            
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job and job.imputed_dataset_path:
                    response['imputed_dataset_path'] = job.imputed_dataset_path
            finally:
                db.close()
                
        return response
    else:
        return {"status": "Job not found", "completed": True, "messages": ["Job not found or not started"]}

# Get API to check for any running jobs
@router.get("/check_running", response_model=dict)
def check_running_jobs():
    """Check if there are any running jobs"""
    try:
        # First, try a simple response to check if the endpoint works at all
        return {"running_job_id": None}
    except Exception as e:
        print(f"Error in check_running_jobs: {str(e)}")
        return {"error": str(e)}

# Alternative endpoint to check running jobs
@router.get("/active")
def get_active_jobs():
    """Alternative endpoint to check running jobs"""
    from typing import Dict, Any, Optional
    
    # Create a very simple response with explicit types
    response: Dict[str, Optional[int]] = {"running_job_id": None}
    return response

# API to stop a job
@router.post("/stop/{job_id}")
async def stop_job(job_id: int, request: Request):
    """Stop a running job"""
    # Validate CSRF token from header
    header_token = request.headers.get("X-CSRF-Token")
    cookie_token = get_csrf_token(request)
    
    if not header_token or header_token != cookie_token:
        return {"success": False, "error": "CSRF validation failed"}
    
    tracker = JobStatusTracker()
    job_status = tracker.get_job_status(job_id)
    
    if job_status and not job_status.completed:
        # TODO: Implement actual job stopping mechanism
        # For now, just mark it as completed
        job_status.add_message("Job stopping requested by user")
        job_status.fail("Job stopped by user")
        return {"success": True}
    else:
        return {"success": False, "error": "Job not found or already completed"}
