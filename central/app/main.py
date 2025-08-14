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
from .services.simi_service import SIMIService
from starlette.middleware.base import BaseHTTPMiddleware

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
            running_job_id = tracker.get_first_running_job_id()
            # Convert to explicit int or None to avoid serialization issues
            result = {"running_job_id": int(running_job_id) if running_job_id is not None else None}
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


manager = ConnectionManager()
simi_service = SIMIService(manager)

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
        if (db_job.algorithm or "").upper() == "SIMI":
            asyncio.create_task(simi_service.start_job(db_job, central_data))
            # Return to the same page with job_id pre-selected
            return templates.TemplateResponse("start_job.html", {
                "request": request, 
                "csrf_token": csrf_token, 
                "jobs": jobs,
                "active_job_id": job_id,  # Pass the active job ID to highlight it
                "message": f"Job {job_id} started. Monitoring status..."
            })
        else:
            return templates.TemplateResponse("start_job.html", {
                "request": request, 
                "csrf_token": csrf_token, 
                "jobs": jobs,
                "error": f"Unsupported algorithm: {db_job.algorithm}"
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
    if algo == 'SIMI':
        try:
            idx = int(target_column_index) if target_column_index not in (None, "") else None
        except ValueError:
            return HTMLResponse("Target column index must be an integer", status_code=400)
        if idx is not None:
            updated.parameters['target_column_index'] = idx
        updated.parameters['is_binary'] = bool(is_binary)
    elif algo == 'SIMICE':
        if target_column_indexes:
            try:
                idxs = [int(x.strip()) for x in target_column_indexes.split(',') if x.strip()]
            except ValueError:
                return HTMLResponse("Invalid indexes list", status_code=400)
            updated.parameters['target_column_indexes'] = idxs
        if is_binary_list:
            bins = [s.strip().lower() in ('true','1','yes') for s in is_binary_list.split(',') if s.strip()]
            updated.parameters['is_binary'] = bins
        if iteration_before_first_imputation is not None:
            updated.parameters['iteration_before_first_imputation'] = iteration_before_first_imputation
        if iteration_between_imputations is not None:
            updated.parameters['iteration_between_imputations'] = iteration_between_imputations
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
    if (db_job.algorithm or "").upper() == "SIMI":
        asyncio.create_task(simi_service.start_job(db_job, central_data))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {db_job.algorithm}")

    return {"message": "Job started"}

@app.websocket("/ws/{site_id}")
async def websocket_endpoint(websocket: WebSocket, site_id: str, token: str = Depends(security.get_token_ws)):
    # WebSocket remains JWT-guarded for remotes
    await manager.connect(websocket, site_id)
    try:
        while True:
            data = await websocket.receive_text()
            await simi_service.handle_message(site_id, data)
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        manager.disconnect(websocket, site_id)

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
