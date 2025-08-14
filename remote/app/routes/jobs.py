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
    found_site_jobs_attr = None
    
    # If site_id is provided, look in that site's job dictionary
    if site_id:
        site_jobs_attr = f'running_jobs_{site_id}'
        site_jobs = getattr(request.app.state, site_jobs_attr, {})
        job_status = site_jobs.get(job_id)
        if job_status:
            found_site_jobs_attr = site_jobs_attr
    else:
        # Look through all site job dictionaries
        for attr_name in dir(request.app.state):
            if attr_name.startswith('running_jobs_'):
                site_jobs = getattr(request.app.state, attr_name, {})
                if job_id in site_jobs:
                    job_status = site_jobs[job_id]
                    found_site_jobs_attr = attr_name
                    break
    
    if job_status:
        # Mark job as completed
        job_status['completed'] = True
        job_status['status'] = "Job stopped by user"
        
        # Log job stopping
        print(f"Job {job_id} marked as stopped")
        
        # Remove the job from the dictionary to prevent memory leaks
        # This also ensures has_running_jobs won't encounter this job again
        if found_site_jobs_attr:
            site_jobs = getattr(request.app.state, found_site_jobs_attr)
            if job_id in site_jobs:
                print(f"Removing job {job_id} from {found_site_jobs_attr}")
                del site_jobs[job_id]
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
        
        # Clean up completed jobs to prevent memory leaks and state issues
        jobs_to_remove = []
        for job_id, job in site_jobs.items():
            if job.get('completed', False):
                jobs_to_remove.append(job_id)
            elif running_job_id is None:
                # Found first running job
                running_job_id = job_id
        
        # Remove completed jobs
        for job_id in jobs_to_remove:
            print(f"Cleaning up completed job {job_id} from {site_jobs_attr}")
            del site_jobs[job_id]
                
        print(f"Checked for running jobs on site {site_id}: {running_job_id or 'None'}")
    else:
        # Check all sites for running jobs
        for attr_name in dir(request.app.state):
            if attr_name.startswith('running_jobs_'):
                site_jobs = getattr(request.app.state, attr_name, {})
                
                # Clean up completed jobs for this site
                jobs_to_remove = []
                for job_id, job in site_jobs.items():
                    if job.get('completed', False):
                        jobs_to_remove.append(job_id)
                    elif running_job_id is None:
                        # Found first running job
                        running_job_id = job_id
                        site_id = attr_name[13:]  # Extract site_id from attribute name
                
                # Remove completed jobs
                for job_id in jobs_to_remove:
                    print(f"Cleaning up completed job {job_id} from {attr_name}")
                    del site_jobs[job_id]
                
                # If we found a running job, no need to check other sites
                if running_job_id:
                    break
                    
        print(f"Checked for running jobs on all sites: {running_job_id or 'None'}")
    
    return {"running_job_id": running_job_id, "site_id": site_id}
    
@router.get("/debug_job_status")
async def debug_job_status(request: Request):
    """Debug endpoint to show all job states"""
    result = {}
    
    # Collect all site job dictionaries
    for attr_name in dir(request.app.state):
        if attr_name.startswith('running_jobs_'):
            site_id = attr_name[13:]  # Extract site_id from attribute name
            site_jobs = getattr(request.app.state, attr_name, {})
            
            # Convert to a serializable format
            site_data = {}
            for job_id, job in site_jobs.items():
                # Convert any non-serializable values
                job_data = {}
                for key, value in job.items():
                    if key == 'start_time' and hasattr(value, 'isoformat'):
                        job_data[key] = value.isoformat()
                    else:
                        job_data[key] = str(value)
                site_data[str(job_id)] = job_data
                
            result[site_id] = site_data
    
    return {"job_state": result}
