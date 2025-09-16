from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, List, Optional
import os
import signal
import asyncio
import time

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
        # Job not found in memory - this indicates service restart or crash
        print(f"Job status request for job_id={job_id}, site_id={site_id} - Job not found (likely service restart)")
        if site_id:
            print(f"Looking in site_jobs_{site_id} dictionary")
            # Check what jobs are available for this site
            site_jobs_attr = f'running_jobs_{site_id}'
            site_jobs = getattr(request.app.state, site_jobs_attr, {})
            print(f"Available jobs for site {site_id}: {list(site_jobs.keys())}")
        
        # Return job as failed due to service restart/crash
        return {
            "status": "Job failed: Service was restarted or crashed. Please start a new job.",
            "completed": True,
            "error": True,  # Indicate this is an error state
            "site_id": site_id or ''
        }

@router.post("/stop_job")
async def stop_job(request: Request, job_id: int, site_id: str = None):
    """Stop a running job.

    Previous implementation only flipped 'completed' and removed the entry without
    terminating the spawned child process (multiprocessing.Process) referenced by 'pid'.
    This allowed the remote algorithm process to continue running (and still accept
    / produce messages) giving the illusion that cancel had no effect.

    New logic:
      1. Find job record.
      2. Attempt graceful SIGTERM (POSIX) if pid present.
      3. Await up to grace_period for exit (non-blocking with asyncio.sleep).
      4. If still alive, escalate to SIGKILL.
      5. Update messages / status accordingly.
      6. Mark completed & remove from registry only after process is confirmed dead
         or we have exhausted escalation attempts.
    """
    job_status = None
    found_site_jobs_attr = None

    # Locate job
    if site_id:
        site_jobs_attr = f'running_jobs_{site_id}'
        site_jobs = getattr(request.app.state, site_jobs_attr, {})
        job_status = site_jobs.get(job_id)
        if job_status:
            found_site_jobs_attr = site_jobs_attr
    else:
        for attr_name in dir(request.app.state):
            if attr_name.startswith('running_jobs_'):
                site_jobs = getattr(request.app.state, attr_name, {})
                if job_id in site_jobs:
                    job_status = site_jobs[job_id]
                    found_site_jobs_attr = attr_name
                    break

    if not job_status:
        return {"success": False, "error": "Job not found or already completed"}

    # Async task cancellation path (new async-only execution model)
    if job_status.get('async') and 'task' in job_status:
        task = job_status.get('task')
        # Avoid duplicate cancellation
        if job_status.get('completed'):
            return {"success": True, "site_id": job_status.get('site_id', ''), "already_completed": True}
        job_status.setdefault('messages', []).append('Cancellation requested (async task)')
        if task and not task.done():
            job_status['status'] = 'Cancelling'
            try:
                task.cancel()
                # Yield so cancellation can propagate
                await asyncio.sleep(0)
            except Exception as e:
                job_status['messages'].append(f'Error issuing cancel: {e}')
                return {"success": False, "error": f"Failed to cancel async task: {e}"}
            # Do NOT mark completed or remove entry; lifecycle callback in start_job will finalize
            return {"success": True, "site_id": job_status.get('site_id', ''), "async": True, "cancelled": True}
        else:
            # Task already finished; rely on lifecycle callback having set status
            return {"success": True, "site_id": job_status.get('site_id', ''), "async": True, "already_done": True}

    # Avoid double stop
    if job_status.get('completed'):
        return {"success": True, "site_id": job_status.get('site_id', ''), "already_completed": True}

    pid = job_status.get('pid')
    job_status.setdefault('messages', [])
    job_status['messages'].append('Stop requested by user')
    job_status['status'] = 'Cancelling...'
    print(f"[cancel] Stop requested for job {job_id} (pid={pid})")

    # Helper to test if process alive (POSIX). If no pid, treat as already stopped.
    def _is_alive(p):
        if not p or not isinstance(p, int):
            return False
        try:
            # signal 0 does not kill, raises OSError if process does not exist
            os.kill(p, 0)
        except OSError:
            return False
        return True

    termination_result = {
        'sent_term': False,
        'sent_kill': False,
        'alive_after': None
    }

    # Attempt graceful termination
    grace_period = 3.0  # seconds total to wait after SIGTERM
    check_interval = 0.2
    elapsed = 0.0

    if _is_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            termination_result['sent_term'] = True
            job_status['messages'].append(f'Sent SIGTERM to pid {pid}')
            print(f"[cancel] Sent SIGTERM to pid {pid}")
        except ProcessLookupError:
            pass
        except Exception as e:
            job_status['messages'].append(f'Error sending SIGTERM: {e}')
            print(f"[cancel] Error sending SIGTERM to pid {pid}: {e}")

        # Wait for graceful exit
        while elapsed < grace_period and _is_alive(pid):
            await asyncio.sleep(check_interval)
            elapsed += check_interval

    # Escalate if still alive
    if _is_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
            termination_result['sent_kill'] = True
            job_status['messages'].append(f'Sent SIGKILL to pid {pid}')
            print(f"[cancel] Escalated to SIGKILL for pid {pid}")
        except ProcessLookupError:
            pass
        except Exception as e:
            job_status['messages'].append(f'Error sending SIGKILL: {e}')
            print(f"[cancel] Error sending SIGKILL to pid {pid}: {e}")

        # Brief wait to allow OS to reap
        await asyncio.sleep(0.1)

    still_alive = _is_alive(pid)
    termination_result['alive_after'] = still_alive

    if still_alive:
        # Could not fully terminate; mark as completed anyway but inform user
        job_status['messages'].append('Warning: Process may still be alive; manual cleanup might be required.')
        job_status['status'] = 'Cancellation attempted (process may still run)'
        print(f"[cancel] WARNING: pid {pid} still alive after escalation")
    else:
        job_status['messages'].append('Job process terminated')
        job_status['status'] = 'Job stopped by user'
        print(f"[cancel] Job {job_id} (pid={pid}) terminated successfully")

    # Mark completed only after termination attempts
    job_status['completed'] = True

    # Remove entry so new jobs can start
    if found_site_jobs_attr:
        site_jobs = getattr(request.app.state, found_site_jobs_attr)
        if job_id in site_jobs:
            del site_jobs[job_id]
            print(f"[cancel] Removed job {job_id} from registry {found_site_jobs_attr}")

    return {
        "success": True,
        "site_id": job_status.get('site_id', ''),
        "termination": termination_result
    }

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
