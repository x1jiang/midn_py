from fastapi import FastAPI, WebSocket, Depends, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pandas as pd
import asyncio
import secrets
from datetime import datetime, timedelta

from . import models, schemas, services
from .db import get_db, engine
from .core import security
from .core.config import settings
from .core.csrf import get_csrf_token, gen_csrf_token, CSRF_COOKIE
from .websockets.connection_manager import ConnectionManager
from .api import users as sites_api, jobs, remote as remote_api
from .services.algorithm_factory import AlgorithmServiceFactory
from starlette.middleware.base import BaseHTTPMiddleware

# Initialize algorithm services
from .services.init_services import *

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GUI setup
app.mount("/static", StaticFiles(directory="central/app/static"), name="static")
templates = Jinja2Templates(directory="central/app/templates")

# API routers (kept for programmatic access)
app.include_router(sites_api.router, prefix="/api/sites", tags=["sites"])
app.include_router(remote_api.router, prefix="/api/remote", tags=["remote"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])

# Import job status router
from .api import job_status
app.include_router(job_status.router, prefix="/api/jobs", tags=["jobs"])

# Direct test endpoint for checking running jobs
@app.get("/test/job-check")
def test_job_check():
    try:
        from .services.job_status import JobStatusTracker
        tracker = JobStatusTracker()
        try:
            # Get the first running job ID
            running_job_id = tracker.get_first_running_job_id()
            
            # Initialize result with running job ID
            result = {"running_job_id": int(running_job_id) if running_job_id is not None else None}
            
            # Add more detailed job status information
            if running_job_id is not None:
                job_status = tracker.get_job_status(running_job_id)
                if job_status:
                    result["job_details"] = {
                        "job_id": job_status.job_id,
                        "status": job_status.status,
                        "completed": job_status.completed,
                        "error": job_status.error,
                        "last_message": job_status.messages[-1] if job_status.messages else "No messages",
                        "message_count": len(job_status.messages),
                    }
                else:
                    result["job_details"] = {"status": "Job found in tracker but no details available"}
            
            # Also show all jobs currently being tracked
            all_jobs = {}
            for jid, jstatus in tracker.jobs.items():
                all_jobs[jid] = {
                    "status": jstatus.status,
                    "completed": jstatus.completed,
                    "last_message": jstatus.messages[-1] if jstatus.messages else "No messages",
                }
            
            result["all_tracked_jobs"] = all_jobs
            print(f"Checking for running jobs: {result}")
            
            # Check if there are any registered instances for algorithms in the factory
            from .services.algorithm_factory import AlgorithmServiceFactory
            result["registered_algorithms"] = list(AlgorithmServiceFactory._service_classes.keys())
            result["active_algorithm_instances"] = list(AlgorithmServiceFactory._service_instances.keys())
            
            # For debugging, try to get the SIMICE service and check actual job dictionary
            try:
                # Use the global manager instance instead of creating a new one
                simice_service = AlgorithmServiceFactory.create_service("SIMICE", manager)
                result["simice_service_check"] = "OK"
                
                # DEBUG: Get actual job dictionary information
                if running_job_id and hasattr(simice_service, 'jobs') and running_job_id in simice_service.jobs:
                    actual_job = simice_service.jobs[running_job_id]
                    result["actual_job_dict"] = {
                        "status": actual_job.get("status"),
                        "participants": actual_job.get("participants", []),
                        "connected_sites": actual_job.get("connected_sites", []),
                        "ready_sites": actual_job.get("ready_sites", []),
                        "parameters": actual_job.get("parameters", {}),
                        "creation_time": actual_job.get("creation_time")
                    }
                    # Also add detailed comparison for debugging
                    from common.algorithm.job_protocol import JobStatus
                    result["debug_info"] = {
                        "job_status_is_waiting": actual_job.get("status") == JobStatus.WAITING.value,
                        "waiting_value": JobStatus.WAITING.value,
                        "actual_status_value": actual_job.get("status"),
                        "all_sites_ready": set(actual_job.get("ready_sites", [])) == set(actual_job.get("participants", [])),
                        "ready_sites_set": list(set(actual_job.get("ready_sites", []))),
                        "participants_set": list(set(actual_job.get("participants", [])))
                    }
                else:
                    result["actual_job_dict"] = "No job found in SIMICE service jobs dictionary"
                    
            except Exception as e:
                result["simice_service_error"] = str(e)
            print(f"Checking for running jobs: {result}")
            return result
        except Exception as e:
            print(f"Error checking for running jobs: {str(e)}")
            # Fall back to a simple response if there's an error
            return {"running_job_id": None, "error": str(e)}
    except Exception as e:
        print(f"Critical error in test_job_check: {str(e)}")
        # Ultra fallback
        return {"running_job_id": None, "critical_error": str(e)}

# Debug endpoint to manually trigger start computation
@app.get("/debug/trigger-start-computation/{job_id}")
async def debug_trigger_start_computation(job_id: int):
    """Manually trigger start computation for debugging."""
    try:
        print(f"🔧 DEBUG: Manually triggering start computation for job {job_id}")
        
        # Get the algorithm service
        from .services.algorithm_factory import AlgorithmServiceFactory
        from common.algorithm.job_protocol import JobStatus
        
        # Use the global manager instance instead of creating a new one
        
        # Check what services are available
        print(f"📋 Available services: {list(AlgorithmServiceFactory._service_instances.keys())}")
        
        # Try to get SIMICE service
        if "SIMICE" not in AlgorithmServiceFactory._service_instances:
            return {"error": "SIMICE service not available", "available_services": list(AlgorithmServiceFactory._service_instances.keys())}
        
        simice_service = AlgorithmServiceFactory._service_instances["SIMICE"]
        
        if job_id not in simice_service.jobs:
            return {"error": f"Job {job_id} not found", "available_jobs": list(simice_service.jobs.keys())}
        
        job = simice_service.jobs[job_id]
        
        # Log current state
        current_state = {
            "status": job.get("status"),
            "participants": job.get("participants", []),
            "connected_sites": job.get("connected_sites", []),
            "ready_sites": job.get("ready_sites", [])
        }
        print(f"🔍 Current job state: {current_state}")
        
        # Manually fix the job status and ready sites
        site_ids = ["224bdbc5", "863a2efd"]
        
        # Ensure sites are connected
        for site_id in site_ids:
            if site_id not in job.get("connected_sites", []):
                job.setdefault("connected_sites", []).append(site_id)
                print(f"✅ Added {site_id} to connected sites")
        
        # Set job status to waiting
        if job["status"] != JobStatus.WAITING.value:
            print(f"🔧 Fixing job status from '{job['status']}' to '{JobStatus.WAITING.value}'")
            job["status"] = JobStatus.WAITING.value
        
        # Manually trigger site_ready for both sites
        for site_id in site_ids:
            site_ready_data = {
                "type": "site_ready",
                "job_id": job_id,
                "site_id": site_id,
                "status": "ready"
            }
            print(f"📤 Triggering site_ready for {site_id}...")
            await simice_service._handle_site_ready(site_id, site_ready_data)
        
        # Return final state
        final_state = {
            "status": job.get("status"),
            "participants": job.get("participants", []),
            "connected_sites": job.get("connected_sites", []),
            "ready_sites": job.get("ready_sites", [])
        }
        
        return {
            "message": "Start computation triggered",
            "initial_state": current_state,
            "final_state": final_state
        }
        
    except Exception as e:
        print(f"❌ Error in debug trigger: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Debug endpoint to restart SIMICE iteration
@app.get("/debug/restart-iteration/{job_id}")
async def debug_restart_iteration(job_id: int):
    try:
        print(f"🔧 DEBUG: Starting restart for job {job_id}")
        
        from .services.algorithm_factory import AlgorithmServiceFactory
        from .services.job_status import JobStatusTracker
        
        # Check if job is running
        tracker = JobStatusTracker()
        running_job_id = tracker.get_first_running_job_id()
        print(f"🔧 DEBUG: Running job ID: {running_job_id}")
        
        if running_job_id != job_id:
            return {"error": f"Job {job_id} is not currently running (running job: {running_job_id})"}
        
        # Get the algorithm service from the factory's cache
        print(f"🔧 DEBUG: Available service instances: {list(AlgorithmServiceFactory._service_instances.keys())}")
        
        if "SIMICE" not in AlgorithmServiceFactory._service_instances:
            return {"error": "SIMICE service instance not found"}
            
        service = AlgorithmServiceFactory._service_instances["SIMICE"]
        print(f"🔧 DEBUG: Got SIMICE service instance")
        print(f"🔧 DEBUG: Available job data: {list(service.job_data.keys())}")
        
        if job_id not in service.job_data:
            return {"error": f"Job {job_id} data not found in SIMICE service"}
        
        job_data = service.job_data[job_id]
        print(f"🔧 DEBUG: Current job state - waiting_for_statistics: {job_data.get('waiting_for_statistics')}")
        print(f"🔧 DEBUG: Current job state - waiting_for_updates: {job_data.get('waiting_for_updates')}")
        print(f"🔧 DEBUG: Current job state - connected_sites: {job_data.get('connected_sites')}")
        
        # Clear the waiting states to reset the algorithm
        job_data['waiting_for_statistics'] = set()
        job_data['waiting_for_updates'] = set()
        print(f"🔧 DEBUG: Cleared waiting states")
        
        # Force restart the current iteration
        print(f"🔧 DEBUG: Calling _run_simice_iteration for job {job_id}")
        try:
            await service._run_simice_iteration(job_id)
            print(f"🔧 DEBUG: _run_simice_iteration completed successfully")
        except Exception as iter_error:
            print(f"💥 DEBUG: Error in _run_simice_iteration: {iter_error}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error in iteration: {str(iter_error)}", "job_data": str(job_data)}
        
        return {"message": f"Restarted iteration for job {job_id}", "job_data": str(job_data)}
        
    except Exception as e:
        print(f"💥 DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Additional debug endpoint to directly send statistics requests
@app.get("/debug/send-stats-request/{job_id}")
async def debug_send_stats_request(job_id: int):
    try:
        print(f"🔧 DEBUG STATS: Starting stats request for job {job_id}")
        
        from .services.algorithm_factory import AlgorithmServiceFactory
        from common.algorithm.job_protocol import create_message
        
        if "SIMICE" not in AlgorithmServiceFactory._service_instances:
            return {"error": "SIMICE service instance not found"}
            
        service = AlgorithmServiceFactory._service_instances["SIMICE"]
        
        if job_id not in service.job_data:
            return {"error": f"Job {job_id} data not found"}
        
        job_data = service.job_data[job_id]
        connected_sites = job_data.get('connected_sites', set())
        target_column_indexes = job_data.get('target_column_indexes', [])
        is_binary = job_data.get('is_binary', [])
        
        if not connected_sites:
            return {"error": "No connected sites"}
        
        # Send statistics request for first target column
        target_col_idx = target_column_indexes[0] - 1  # Convert to 0-based
        method = "logistic" if is_binary[0] else "gaussian"
        
        print(f"🔧 DEBUG STATS: Sending stats request for column {target_col_idx} ({method})")
        
        results = []
        for site_id in connected_sites:
            try:
                message = create_message(
                    "compute_statistics",
                    job_id=job_id,
                    target_col_idx=target_col_idx,
                    method=method
                )
                
                print(f"📤 DEBUG STATS: Sending to site {site_id}: {message}")
                await service.manager.send_to_site(message, site_id)
                results.append(f"Sent to {site_id}: SUCCESS")
                print(f"✅ DEBUG STATS: Successfully sent to site {site_id}")
                
            except Exception as send_error:
                results.append(f"Sent to {site_id}: ERROR - {str(send_error)}")
                print(f"💥 DEBUG STATS: Error sending to site {site_id}: {send_error}")
        
        # Update job state
        job_data['waiting_for_statistics'] = set(connected_sites)
        job_data['statistics'][f"{target_col_idx}_{method}"] = {}
        
        return {
            "message": f"Sent statistics requests for job {job_id}",
            "target_column": target_col_idx,
            "method": method,
            "results": results,
            "connected_sites": list(connected_sites)
        }
        
    except Exception as e:
        print(f"💥 DEBUG STATS ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Debug endpoint to directly send update_imputations messages
@app.get("/debug/send-updates/{job_id}")
async def debug_send_updates(job_id: int):
    try:
        print(f"🔧 DEBUG UPDATES: Starting updates for job {job_id}")
        
        from .services.algorithm_factory import AlgorithmServiceFactory
        from common.algorithm.job_protocol import create_message
        import numpy as np
        
        if "SIMICE" not in AlgorithmServiceFactory._service_instances:
            return {"error": "SIMICE service instance not found"}
            
        service = AlgorithmServiceFactory._service_instances["SIMICE"]
        
        if job_id not in service.job_data:
            return {"error": f"Job {job_id} data not found"}
        
        job_data = service.job_data[job_id]
        connected_sites = job_data.get('connected_sites', set())
        target_column_indexes = job_data.get('target_column_indexes', [])
        
        if not connected_sites:
            return {"error": "No connected sites"}
        
        # Send dummy update_imputations for first target column
        target_col_idx = target_column_indexes[0] - 1  # Convert to 0-based
        
        print(f"🔧 DEBUG UPDATES: Sending updates for column {target_col_idx}")
        
        # Create dummy imputation values (random numbers for testing)
        np.random.seed(42)  # For reproducible results
        dummy_imputations = np.random.normal(0, 1, 100).tolist()  # 100 dummy values
        
        results = []
        for site_id in connected_sites:
            try:
                message = create_message(
                    "update_imputations",
                    job_id=job_id,
                    target_col_idx=target_col_idx,
                    imputations=dummy_imputations[:50]  # Send 50 values per site
                )
                
                print(f"📤 DEBUG UPDATES: Sending to site {site_id}: update_imputations with {len(dummy_imputations[:50])} values")
                await service.manager.send_to_site(message, site_id)
                results.append(f"Sent to {site_id}: SUCCESS")
                print(f"✅ DEBUG UPDATES: Successfully sent to site {site_id}")
                
            except Exception as send_error:
                results.append(f"Sent to {site_id}: ERROR - {str(send_error)}")
                print(f"💥 DEBUG UPDATES: Error sending to site {site_id}: {send_error}")
        
        # Update job state
        job_data['waiting_for_updates'] = set()  # Clear waiting state
        
        return {
            "message": f"Sent update_imputations for job {job_id}",
            "target_column": target_col_idx,
            "results": results,
            "connected_sites": list(connected_sites),
            "imputation_count": len(dummy_imputations[:50])
        }
        
    except Exception as e:
        print(f"💥 DEBUG UPDATES ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


manager = ConnectionManager()

# Algorithm services will be created dynamically using the factory

# ---------------- GUI ROUTES ----------------
# Simple admin auth + CSRF
ADMIN_COOKIE = "admin_auth"

def is_admin(request: Request) -> bool:
    return request.cookies.get(ADMIN_COOKIE) == "1"

# Middleware to restrict /api to admin + CSRF (header)
@app.middleware("http")
async def api_guard_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api") and not path.startswith("/api/remote"):
        # Protect all API endpoints except the remote JWT-based endpoints
        if not is_admin(request):
            # For APIs, return 401 JSON instead of redirect
            return HTMLResponse("Admin authentication required", status_code=401)
        
        # Skip CSRF check for job status API since it's called by JavaScript
        if path.startswith("/api/jobs/status/"):
            # This is a GET request for job status and should be allowed without CSRF
            pass
        elif request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            # For the stop job endpoint, we'll accept the token from multiple places
            if path.startswith("/api/jobs/stop/"):
                # We'll check the CSRF in the route handler
                pass
            else:
                # For other API endpoints, require CSRF header
                header_token = request.headers.get("X-CSRF-Token")
                if not header_token or header_token != get_csrf_token(request):
                    return HTMLResponse("CSRF validation failed", status_code=403)
    
    response = await call_next(request)
    return response

@app.get("/gui/login", response_class=HTMLResponse)
async def gui_login_get(request: Request):
    # Ensure a CSRF token exists for the login form (optional)
    token = get_csrf_token(request) or gen_csrf_token()
    resp = templates.TemplateResponse("login.html", {"request": request, "csrf_token": token})
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

@app.post("/gui/login")
async def gui_login_post(request: Request, password: str = Form(...)):
    if password == settings.ADMIN_PASSWORD:
        # Set admin cookie and (re)issue CSRF token
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(ADMIN_COOKIE, "1", httponly=True, samesite="lax")
        resp.set_cookie(CSRF_COOKIE, gen_csrf_token(), httponly=False, samesite="lax")
        return resp
    return templates.TemplateResponse("confirm.html", {"request": request, "message": "Invalid admin password."})

@app.get("/", response_class=HTMLResponse)
async def gui_home(request: Request):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    token = get_csrf_token(request) or gen_csrf_token()
    resp = templates.TemplateResponse("index.html", {"request": request, "csrf_token": token})
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

# User Registration (GUI)
@app.get("/gui/users/register", response_class=HTMLResponse)
async def gui_register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/gui/users/register", response_class=HTMLResponse)
async def gui_register_post(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    institution: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Enforce unique site name; allow duplicate emails
    if services.user_service.get_user_by_username(db, username=username):
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Site name already registered."})
    user = schemas.UserCreate(username=username, email=email, institution=institution, password=password)
    db_user = services.user_service.create_user(db, user)
    return templates.TemplateResponse("confirm.html", {"request": request, "message": f"Site created. ID={db_user.id}, site_id={db_user.site_id}. Await approval."})

# Admin: list users and approve
@app.get("/gui/admin/users", response_class=HTMLResponse)
async def gui_admin_users(request: Request, db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    token = get_csrf_token(request) or gen_csrf_token()
    users_list = services.user_service.get_users(db, 0, 1000)
    resp = templates.TemplateResponse("admin_users.html", {"request": request, "users": users_list, "csrf_token": token})
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

@app.post("/gui/admin/users/{user_id}/approve", response_class=HTMLResponse)
async def gui_admin_approve(request: Request, user_id: int, csrf_token: str = Form(""), db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    if csrf_token != get_csrf_token(request):
        return HTMLResponse("CSRF validation failed", status_code=403)
    db_user = services.user_service.approve_user(db, user_id)
    if not db_user:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "User not found."})
    return templates.TemplateResponse("confirm.html", {"request": request, "message": f"Approved user {db_user.username} (site_id={db_user.site_id}). Email sent."})

# Create Job (GUI)
@app.get("/gui/jobs/create", response_class=HTMLResponse)
async def gui_jobs_create_get(request: Request, db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    token = get_csrf_token(request) or gen_csrf_token()
    approved = [u for u in services.user_service.get_users(db, 0, 1000) if u.is_approved]
    resp = templates.TemplateResponse("create_job.html", {
        "request": request,
        "csrf_token": token,
        "algorithms": settings._ALG,
        "approved_users": approved,
    })
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

@app.post("/gui/jobs/create", response_class=HTMLResponse)
async def gui_jobs_create_post(
    request: Request,
    algorithm: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    participants: list[str] = Form([]),
    target_column_index: str = Form(None),
    is_binary: bool = Form(False),
    target_column_indexes: str = Form("") ,
    is_binary_list: str = Form("") ,
    iteration_before_first_imputation: int = Form(None),
    iteration_between_imputations: int = Form(None),
    imputation_trials: int = Form(10),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db)
):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    if csrf_token != get_csrf_token(request):
        return HTMLResponse("CSRF validation failed", status_code=403)
    algo = algorithm.upper()
    if algo not in settings._ALG:
        return HTMLResponse("Unsupported algorithm", status_code=400)

    # Build parameters based on algorithm
    if algo == 'SIMI':
        try:
            idx = int(target_column_index) if target_column_index not in (None, "") else None
        except ValueError:
            return HTMLResponse("Target column index must be an integer", status_code=400)
        if idx is None or idx < 1:
            return HTMLResponse("Target column index is required (1-based)", status_code=400)
        params = {"target_column_index": idx, "is_binary": is_binary}
        missing_spec = {"target_column_index": idx, "": is_binary}
    elif algo == 'SIMICE':
        try:
            idxs = [int(x.strip()) for x in target_column_indexes.split(',') if x.strip()]
            bins = [s.strip().lower() in ('true','1','yes') for s in is_binary_list.split(',') if s.strip()]
        except ValueError:
            return HTMLResponse("Invalid indexes or boolean list", status_code=400)
        if not idxs or len(bins) != len(idxs):
            return HTMLResponse("Indexes and is_binary list must be same length and non-empty", status_code=400)
        if iteration_before_first_imputation is None or iteration_between_imputations is None:
            return HTMLResponse("Iteration fields are required for SIMICE", status_code=400)
        params = {
            "target_column_indexes": idxs,
            "is_binary": bins,
            "iteration_before_first_imputation": iteration_before_first_imputation,
            "iteration_between_imputations": iteration_between_imputations
        }
        # For multi-feature, missing_spec can mirror params specific fields
        missing_spec = {
            "target_column_indexes": idxs,
            "is_binary": bins,
            "iteration_before_first_imputation": iteration_before_first_imputation,
            "iteration_between_imputations": iteration_between_imputations
        }
    else:
        return HTMLResponse("Unsupported algorithm", status_code=400)

    job = schemas.JobCreate(
        name=name,
        description=description,
        algorithm=algo,
        parameters=params,
        participants=participants,
        missing_spec=missing_spec,
        imputation_trials=imputation_trials,
    )
    db_job = services.job_service.create_job(db, job, owner_id=None)
    return templates.TemplateResponse("confirm.html", {"request": request, "message": f"Job created. ID={db_job.id}"})

# Start Job (GUI)
@app.get("/gui/jobs/start", response_class=HTMLResponse)
async def gui_jobs_start_get(request: Request, db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    token = get_csrf_token(request) or gen_csrf_token()
    
    # Get all jobs and convert them to serializable dictionaries
    db_jobs = services.job_service.get_jobs(db, 0, 1000)
    jobs = []
    for job in db_jobs:
        # Create job dict with fields in a consistent order matching the job creation form
        job_dict = {
            # Basic job information (top of form)
            "id": job.id,
            "algorithm": job.algorithm,
            "name": job.name,
            "description": job.description,
            "participants": job.participants,
            "status": job.status,
            
            # Algorithm-specific parameters in order they appear in the form
            "parameters": {},
        }
        
        # Copy all parameters with proper ordering based on algorithm
        if job.parameters:
            if job.algorithm == "SIMI":
                # Order SIMI parameters
                ordered_params = {}
                if "target_column_index" in job.parameters:
                    ordered_params["target_column_index"] = job.parameters["target_column_index"]
                if "is_binary" in job.parameters:
                    ordered_params["is_binary"] = job.parameters["is_binary"]
                # Add any remaining parameters
                for key, value in job.parameters.items():
                    if key not in ordered_params:
                        ordered_params[key] = value
                job_dict["parameters"] = ordered_params
            elif job.algorithm == "SIMICE":
                # Order SIMICE parameters
                ordered_params = {}
                if "target_column_indexes" in job.parameters:
                    ordered_params["target_column_indexes"] = job.parameters["target_column_indexes"]
                if "is_binary_list" in job.parameters:
                    ordered_params["is_binary_list"] = job.parameters["is_binary_list"]
                # Add any remaining parameters
                for key, value in job.parameters.items():
                    if key not in ordered_params:
                        ordered_params[key] = value
                job_dict["parameters"] = ordered_params
            else:
                # For other algorithms, just use the parameters as-is
                job_dict["parameters"] = job.parameters
        
        # Add the remaining job properties at the end (bottom of form)
        job_dict["iteration_before_first_imputation"] = job.iteration_before_first_imputation
        job_dict["iteration_between_imputations"] = job.iteration_between_imputations
        job_dict["imputation_trials"] = job.imputation_trials
        job_dict["missing_spec"] = job.missing_spec
        job_dict["owner_id"] = job.owner_id
        
        jobs.append(job_dict)
    
    resp = templates.TemplateResponse("start_job.html", {"request": request, "csrf_token": token, "jobs": jobs})
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

@app.post("/gui/jobs/start", response_class=HTMLResponse)
async def gui_jobs_start_post(
    request: Request,
    job_id: int = Form(...),
    central_data_file: UploadFile = File(...),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db)
):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    if csrf_token != get_csrf_token(request):
        return HTMLResponse("CSRF validation failed", status_code=403)
    
    db_job = services.job_service.get_job(db, job_id=job_id)
    if not db_job:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Job not found."})
    
    try:
        central_data = pd.read_csv(central_data_file.file)
        
        # Get all jobs and convert them to serializable dictionaries - for reloading the form
        db_jobs = services.job_service.get_jobs(db, 0, 1000)
        jobs = []
        for job in db_jobs:
            # Create job dict with fields in a consistent order (same as the GET handler)
            job_dict = {
                "id": job.id,
                "algorithm": job.algorithm,
                "name": job.name,
                "description": job.description,
                "participants": job.participants,
                "status": job.status,
                "parameters": {},
            }
            
            # Copy parameters with proper ordering based on algorithm
            if job.parameters:
                if job.algorithm == "SIMI":
                    ordered_params = {}
                    if "target_column_index" in job.parameters:
                        ordered_params["target_column_index"] = job.parameters["target_column_index"]
                    if "is_binary" in job.parameters:
                        ordered_params["is_binary"] = job.parameters["is_binary"]
                    for key, value in job.parameters.items():
                        if key not in ordered_params:
                            ordered_params[key] = value
                    job_dict["parameters"] = ordered_params
                elif job.algorithm == "SIMICE":
                    ordered_params = {}
                    if "target_column_indexes" in job.parameters:
                        ordered_params["target_column_indexes"] = job.parameters["target_column_indexes"]
                    if "is_binary_list" in job.parameters:
                        ordered_params["is_binary_list"] = job.parameters["is_binary_list"]
                    for key, value in job.parameters.items():
                        if key not in ordered_params:
                            ordered_params[key] = value
                    job_dict["parameters"] = ordered_params
                else:
                    job_dict["parameters"] = job.parameters
            
            # Add the remaining job properties at the end
            job_dict["iteration_before_first_imputation"] = job.iteration_before_first_imputation
            job_dict["iteration_between_imputations"] = job.iteration_between_imputations
            job_dict["imputation_trials"] = job.imputation_trials
            job_dict["missing_spec"] = job.missing_spec
            job_dict["owner_id"] = job.owner_id
            
            jobs.append(job_dict)
        
        # Dispatch based on algorithm
        algorithm_name = (db_job.algorithm or "").upper()
        try:
            algorithm_service = AlgorithmServiceFactory.create_service(algorithm_name, manager)
            asyncio.create_task(algorithm_service.start_job(db_job, central_data))
            # Return to the same page with job_id pre-selected
            return templates.TemplateResponse("start_job.html", {
                "request": request, 
                "csrf_token": csrf_token, 
                "jobs": jobs,
                "active_job_id": job_id,  # Pass the active job ID to highlight it
                "message": f"Job {job_id} started. Monitoring status..."
            })
        except ValueError as e:
            return templates.TemplateResponse("start_job.html", {
                "request": request, 
                "csrf_token": csrf_token, 
                "jobs": jobs,
                "error": str(e)
            })
    except Exception as e:
        # Handle errors and return to the same page
        return templates.TemplateResponse("start_job.html", {
            "request": request, 
            "csrf_token": csrf_token, 
            "jobs": jobs if 'jobs' in locals() else [],
            "error": f"Error starting job: {str(e)}"
        })

# List Jobs (GUI)
@app.get("/gui/jobs", response_class=HTMLResponse)
async def gui_jobs_list(request: Request, db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    token = get_csrf_token(request) or gen_csrf_token()
    jobs_list = services.job_service.get_jobs(db, 0, 1000)
    resp = templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs_list, "csrf_token": token})
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

# Edit Job (GUI)
@app.get("/gui/jobs/{job_id}/edit", response_class=HTMLResponse)
async def gui_jobs_edit_get(request: Request, job_id: int, db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    job = services.job_service.get_job(db, job_id)
    if not job:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Job not found."})
    token = get_csrf_token(request) or gen_csrf_token()
    approved = [u for u in services.user_service.get_users(db, 0, 1000) if u.is_approved]
    resp = templates.TemplateResponse("edit_job.html", {
        "request": request, 
        "job": job, 
        "csrf_token": token,
        "approved_users": approved
    })
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

@app.post("/gui/jobs/{job_id}/edit", response_class=HTMLResponse)
async def gui_jobs_edit_post(request: Request, job_id: int,
                             name: str = Form(None), description: str = Form(None),
                             participants: list[str] = Form([]),
                             target_column_index: str = Form(None),
                             is_binary: bool = Form(False),
                             target_column_indexes: str = Form(""),
                             is_binary_list: str = Form(""),
                             iteration_before_first_imputation: int = Form(None),
                             iteration_between_imputations: int = Form(None),
                             imputation_trials: int = Form(None),
                             csrf_token: str = Form(""),
                             db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    if csrf_token != get_csrf_token(request):
        return HTMLResponse("CSRF validation failed", status_code=403)
    job = services.job_service.get_job(db, job_id)
    if not job:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Job not found."})

    # Update name/desc/participants/trials
    updated = services.job_service.update_job(db, job_id, name=name, description=description,
                                              participants=participants, imputation_trials=imputation_trials)
    if not updated:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Job not found."})

    # Update parameters based on algorithm
    algo = (job.algorithm or '').upper()
    # Make a copy of parameters to modify
    params = dict(updated.parameters or {})
    
    if algo == 'SIMI':
        try:
            idx = int(target_column_index) if target_column_index not in (None, "") else None
        except ValueError:
            return HTMLResponse("Target column index must be an integer", status_code=400)
        if idx is not None:
            params['target_column_index'] = idx
        params['is_binary'] = bool(is_binary)
    elif algo == 'SIMICE':
        if target_column_indexes:
            try:
                idxs = [int(x.strip()) for x in target_column_indexes.split(',') if x.strip()]
            except ValueError:
                return HTMLResponse("Invalid indexes list", status_code=400)
            params['target_column_indexes'] = idxs
        if is_binary_list:
            bins = [s.strip().lower() in ('true','1','yes') for s in is_binary_list.split(',') if s.strip()]
            params['is_binary'] = bins
        if iteration_before_first_imputation is not None:
            params['iteration_before_first_imputation'] = iteration_before_first_imputation
        if iteration_between_imputations is not None:
            params['iteration_between_imputations'] = iteration_between_imputations
    
    # Reassign the entire dictionary to trigger SQLAlchemy change detection
    updated.parameters = params
    db.add(updated)
    db.commit()
    db.refresh(updated)
    return templates.TemplateResponse("confirm.html", {"request": request, "message": f"Job {job_id} updated."})

@app.post("/gui/jobs/{job_id}/delete", response_class=HTMLResponse)
async def gui_jobs_delete_post(request: Request, job_id: int, csrf_token: str = Form(""), db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    if csrf_token != get_csrf_token(request):
        return HTMLResponse("CSRF validation failed", status_code=403)
    ok = services.job_service.delete_job(db, job_id)
    if not ok:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Job not found."})
    return templates.TemplateResponse("confirm.html", {"request": request, "message": f"Job {job_id} deleted."})

# --------------- Existing API/WebSocket ----------------
@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: int, central_data_file: UploadFile = File(...), db: Session = Depends(get_db), request: Request = None):
    if request and not is_admin(request):
        raise HTTPException(status_code=401, detail="Admin authentication required")
    db_job = services.job_service.get_job(db, job_id=job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")

    central_data = pd.read_csv(central_data_file.file)

    # Dispatch based on algorithm
    algorithm_name = (db_job.algorithm or "").upper()
    try:
        algorithm_service = AlgorithmServiceFactory.create_service(algorithm_name, manager)
        asyncio.create_task(algorithm_service.start_job(db_job, central_data))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Job started"}

@app.websocket("/ws/{site_id}")
async def websocket_endpoint(websocket: WebSocket, site_id: str, token: str = Depends(security.get_token_ws)):
    # WebSocket remains JWT-guarded for remotes
    print(f"🔌 WebSocket: New connection from site {site_id}")
    print(f"🎫 WebSocket: Token validated for site {site_id}")
    
    # Check if there's a running job (for informational purposes)
    from .services.job_status import JobStatusTracker
    tracker = JobStatusTracker()
    running_job_id = tracker.get_first_running_job_id()
    
    if running_job_id:
        print(f"📋 WebSocket: Job {running_job_id} is currently running - allowing site {site_id} to connect")
    else:
        print(f"📋 WebSocket: No jobs currently running - site {site_id} will wait for jobs")
    
    await manager.connect(websocket, site_id)
    print(f"✅ WebSocket: Connection established for site {site_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            print(f"📨 WebSocket: Received message from site {site_id}")
            print(f"💬 WebSocket: Message preview: {data[:200]}{'...' if len(data) > 200 else ''}")
            
            # Route message to the appropriate algorithm service
            await route_websocket_message(site_id, data)
            print(f"✅ WebSocket: Message routed successfully for site {site_id}")
            
    except Exception as e:
        print(f"💥 WebSocket Error for site {site_id}: {e}")
        print(f"💥 WebSocket Error details: {type(e).__name__}: {str(e)}")
    finally:
        print(f"🔌 WebSocket: Disconnecting site {site_id}")
        manager.disconnect(websocket, site_id)
        print(f"❌ WebSocket: Site {site_id} disconnected")

async def route_websocket_message(site_id: str, data: str):
    """
    Route WebSocket message to the appropriate algorithm service.
    """
    # Focused logging for stats messages
    if "stats" in data:
        print(f"📊 Router: STATS from {site_id}")
    # else:
    #     print(f"🎯 Router: Processing message from site {site_id}")  # Reduced noise
    
    try:
        # Parse message to determine which algorithm/job it belongs to
        import json
        from common.algorithm.job_protocol import parse_message
        
        # print(f"📝 Router: Parsing message from site {site_id}")  # Reduced noise
        # parse_message returns a dictionary, not a tuple
        message_data = parse_message(data)
        message_type = message_data.get('type')
        job_id = message_data.get('job_id')
        
        if message_type == "stats":
            print(f"🔍 Router: STATS message - job_id: {job_id}")
        # else:
        #     print(f"🔍 Router: Parsed message - type: {message_type}, job_id: {job_id}")  # Reduced noise
        
        if job_id:
            # Look up the job to determine which algorithm service to use
            from .db import get_db
            from . import services
            
            print(f"🔍 Router: Looking up job {job_id}")
            db = next(get_db())
            try:
                db_job = services.job_service.get_job(db, job_id=job_id)
                if db_job:
                    # Check if job is completed - reject connection if it is
                    if db_job.status == "completed":
                        print(f"🚫 Router: Job {job_id} is already completed, rejecting message from site {site_id}")
                        # Send a rejection message to the site
                        rejection_message = {
                            "type": "error",
                            "job_id": job_id,
                            "message": f"Job {job_id} is already completed. No further processing needed."
                        }
                        await manager.send_to_site(json.dumps(rejection_message), site_id)
                        
                        # Also close the connection with the site
                        await manager.disconnect_site(site_id)
                        print(f"🔌 Router: Disconnected site {site_id} for completed job {job_id}")
                        return
                        
                    algorithm_name = (db_job.algorithm or "").upper()
                    print(f"🎯 Router: Found job {job_id}, algorithm: {algorithm_name}")
                    
                    algorithm_service = AlgorithmServiceFactory.create_service(algorithm_name, manager)
                    print(f"🏭 Router: Created {algorithm_name} service for job {job_id}")
                    
                    await algorithm_service.handle_site_message(site_id, data)
                    print(f"✅ Router: Message handled by {algorithm_name} service")
                else:
                    print(f"❌ Router: Job {job_id} not found for message from site {site_id}")
                    # Send a rejection message to the site
                    rejection_message = {
                        "type": "error",
                        "job_id": job_id,
                        "message": f"Job {job_id} not found. Connection rejected."
                    }
                    await manager.send_to_site(json.dumps(rejection_message), site_id)
            except Exception as e:
                print(f"💥 Router: Error routing message from site {site_id}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                db.close()
        else:
            print(f"❌ Router: No job_id in message from site {site_id}: {data[:100]}{'...' if len(data) > 100 else ''}")
            
    except Exception as e:
        print(f"💥 Router: Error parsing message from site {site_id}: {e}")
        print(f"📝 Router: Problematic message: {data[:200]}{'...' if len(data) > 200 else ''}")
        import traceback
        traceback.print_exc()

@app.get("/health")
def read_health():
    return {"message": "Central Server is running"}

@app.get("/gui/logout")
async def gui_logout():
    resp = RedirectResponse(url="/gui/login", status_code=303)
    # clear admin and csrf cookies
    resp.delete_cookie("admin_auth")
    resp.delete_cookie("csrf_token")
    return resp

@app.get("/gui/admin/sites", response_class=HTMLResponse)
async def gui_admin_sites(request: Request, db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    token = get_csrf_token(request) or gen_csrf_token()
    sites_list = services.user_service.get_users(db, 0, 1000)
    resp = templates.TemplateResponse("admin_sites.html", {"request": request, "users": sites_list, "csrf_token": token})
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

@app.post("/gui/admin/sites/{user_id}/approve", response_class=HTMLResponse)
async def gui_admin_sites_approve(request: Request, user_id: int, csrf_token: str = Form(""), expires_days: int = Form(30), db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    if csrf_token != get_csrf_token(request):
        return HTMLResponse("CSRF validation failed", status_code=403)
    db_user = services.user_service.approve_user(db, user_id, expires_days=expires_days)
    if not db_user:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Site not found."})
    exp_ts = (datetime.utcnow() + timedelta(days=expires_days)).strftime("%Y-%m-%d %H:%M:%S UTC")
    return templates.TemplateResponse("confirm.html", {"request": request, "message": f"Approved site {db_user.username} (site_id={db_user.site_id}). JWT expires at {exp_ts}. Email sent."})
