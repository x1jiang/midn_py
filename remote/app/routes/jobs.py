from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, List, Optional

router = APIRouter()

@router.get("/job_status")
async def get_job_status(request: Request, job_id: int, site_id: str = None):
    """Get the status of a running job for a specific site"""
    # If no site_id is provided, try to find the job in any site's job list
    if site_id:
        # Look for the job in the specified site's job dictionary
        site_jobs_attr = f'running_jobs_{site_id}'
        site_jobs = getattr(request.app.state, site_jobs_attr, {})
        job_status = site_jobs.get(job_id)
    else:
        # Look through all attributes that start with running_jobs_
        job_status = None
        for attr_name in dir(request.app.state):
            if attr_name.startswith('running_jobs_'):
                site_jobs = getattr(request.app.state, attr_name, {})
                if job_id in site_jobs:
                    job_status = site_jobs[job_id]
                    break
    
    if job_status:
        # Only consider a job completed if it's explicitly marked as completed
        # Otherwise, it's still in progress even if rounds have completed
        return {
            "status": "\n".join(job_status.get('messages', [])),
            "completed": job_status.get('completed', False),  # Only true if explicitly stopped
            "round_completed": job_status.get('round_completed', False),  # Round completion status
            "site_id": job_status.get('site_id', '')
        }
    else:
        # Log a more detailed message for debugging
        print(f"Job status request for job_id={job_id}, site_id={site_id} - Job not found")
        if site_id:
            print(f"Looking in site_jobs_{site_id} dictionary")
            # Check what jobs are available for this site
            site_jobs_attr = f'running_jobs_{site_id}'
            site_jobs = getattr(request.app.state, site_jobs_attr, {})
            print(f"Available jobs for site {site_id}: {list(site_jobs.keys())}")
        
        # Still return the not found status
        return {"status": "Job not found", "completed": True}

@router.post("/stop_job")
async def stop_job(request: Request, job_id: int, site_id: str = None):
    """Stop a running job"""
    job_status = None
    
    # If site_id is provided, look in that site's job dictionary
    if site_id:
        site_jobs_attr = f'running_jobs_{site_id}'
        site_jobs = getattr(request.app.state, site_jobs_attr, {})
        job_status = site_jobs.get(job_id)
    else:
        # Look through all site job dictionaries
        for attr_name in dir(request.app.state):
            if attr_name.startswith('running_jobs_'):
                site_jobs = getattr(request.app.state, attr_name, {})
                if job_id in site_jobs:
                    job_status = site_jobs[job_id]
                    break
    
    if job_status and not job_status.get('completed'):
        # TODO: Implement actual job stopping mechanism
        # For now, just mark it as completed
        job_status['completed'] = True
        job_status['status'] = "Job stopped by user"
        job_status['messages'].append("Job stopped by user")
        return {"success": True, "site_id": job_status.get('site_id', '')}
    else:
        return {"success": False, "error": "Job not found or already completed"}

@router.get("/check_running_jobs")
async def check_running_jobs(request: Request, site_id: Optional[str] = None):
    """Check if there are any running jobs for a specific site or any site"""
    running_job_id = None
    
    # If site_id is provided, only check that site's jobs
    if site_id:
        site_jobs_attr = f'running_jobs_{site_id}'
        site_jobs = getattr(request.app.state, site_jobs_attr, {})
        
        # Find the first running job
        for job_id, job in site_jobs.items():
            if not job.get('completed', False):
                running_job_id = job_id
                break
                
        print(f"Checked for running jobs on site {site_id}: {running_job_id or 'None'}")
    else:
        # Check all sites for running jobs
        for attr_name in dir(request.app.state):
            if attr_name.startswith('running_jobs_'):
                site_jobs = getattr(request.app.state, attr_name, {})
                for job_id, job in site_jobs.items():
                    if not job.get('completed', False):
                        running_job_id = job_id
                        site_id = attr_name[13:]  # Extract site_id from attribute name
                        break
                if running_job_id:
                    break
                    
        print(f"Checked for running jobs on all sites: {running_job_id or 'None'}")
    
    return {"running_job_id": running_job_id, "site_id": site_id}
