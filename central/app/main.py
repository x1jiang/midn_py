from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Request, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, Any, List
from pathlib import Path
import json
import logging
import numpy as np
import zipfile
import io
import sys

from . import models, schemas, services
from .db import get_db, engine
from .core.config import settings
from .core.csrf import get_csrf_token, gen_csrf_token, CSRF_COOKIE
from .api import users as sites_api, jobs, remote as remote_api
from starlette.middleware.base import BaseHTTPMiddleware

# Keep DB models

models.Base.metadata.create_all(bind=engine)

# Startup migration: normalize legacy SIMICE 'is_binary' list key to 'is_binary_list'
try:  # non-critical
    from .db import SessionLocal  # type: ignore
    with SessionLocal() as _m:  # type: ignore
        jobs_fix = _m.query(models.Job).filter(models.Job.algorithm == "SIMICE").all()  # type: ignore
        changed = False
        for j in jobs_fix:
            if isinstance(j.parameters, dict) and "is_binary_list" not in j.parameters and isinstance(j.parameters.get("is_binary"), list):
                j.parameters["is_binary_list"] = j.parameters.pop("is_binary")
                changed = True
        if changed:
            _m.commit()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Legacy job status router removed; using lightweight /api/jobs/status/{job_id} defined below

# -------- New lightweight WS + job runner (mimic MIDN_R_PY/run_imputation.py) --------
# Allow importing algorithm modules under MIDN_R_PY without modifying them
sys.path.append(str(Path(__file__).resolve().parents[2] / "MIDN_R_PY"))

# Global WS state and job tracking
remote_websockets: Dict[str, WebSocket] = {}
site_locks: Dict[str, asyncio.Lock] = {}
expected_sites: Optional[Set[str]] = None
all_sites_connected = asyncio.Event()
imputation_running = asyncio.Event()
jobs_runtime: Dict[int, Dict[str, Any]] = {}
# Store results under the publicly served static directory so they can be accessed if needed
# Path: central/app/static/results (user requirement: central/static/results; project layout uses app/static)
RESULTS_DIR = Path(__file__).resolve().parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Direct test endpoint for checking running jobs
@app.get("/test/job-check")
def test_job_check():
    for jid, info in jobs_runtime.items():
        if info.get("running") and not info.get("completed"):
            return {"running_job_id": jid}
    return {"running_job_id": None}

# Debug endpoint to manually trigger start computation
@app.get("/debug/trigger-start-computation/{job_id}")
async def debug_trigger_start_computation(job_id: int):
    return {"message": "Not applicable with new runner"}


# Debug endpoint to restart SIMICE iteration
@app.get("/debug/restart-iteration/{job_id}")
async def debug_restart_iteration(job_id: int):
    return {"message": "Not applicable with new runner"}

# Additional debug endpoint to directly send statistics requests
@app.get("/debug/send-stats-request/{job_id}")
async def debug_send_stats_request(job_id: int):
    return {"message": "Not applicable with new runner"}

# Debug endpoint to directly send update_imputations messages
@app.get("/debug/send-updates/{job_id}")
async def debug_send_updates(job_id: int):
    return {"message": "Not applicable with new runner"}


# --- WebSocket endpoint (no ConnectionManager, direct registry) ---
def _set_expected_sites(sites: List[str]):
    global expected_sites
    expected_sites = set(sites)
    all_sites_connected.clear()

@app.websocket("/ws/{site_id}")
async def websocket_endpoint(websocket: WebSocket, site_id: str):
    # Import here so dynamic sys.path modification above is in effect; avoids static analysis error
    try:
        from Core.transfer import write_string  # type: ignore
    except Exception:
        write_string = None  # Fallback; will skip heartbeat if protocol helper not available
    if site_id not in site_locks:
        site_locks[site_id] = asyncio.Lock()
    await websocket.accept()
    remote_websockets[site_id] = websocket
    try:
        if expected_sites and expected_sites.issubset(set(remote_websockets.keys())):
            all_sites_connected.set()
        # Lightweight keepalive; algorithm coroutines own send/recv
        while True:
            await asyncio.sleep(30)
            # Don't ping during active imputation to avoid interfering with recv
            if not imputation_running.is_set() and write_string:
                try:
                    await write_string("ping", websocket)
                except Exception:
                    break
    finally:
        remote_websockets.pop(site_id, None)

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
            "is_binary_list": bins,
            "iteration_before_first_imputation": iteration_before_first_imputation,
            "iteration_between_imputations": iteration_between_imputations
        }
        # For multi-feature, missing_spec can mirror params specific fields
        missing_spec = {
            "target_column_indexes": idxs,
            "is_binary_list": bins,
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

# Start Job (GUI) - list all jobs for selection
@app.get("/gui/jobs/start", response_class=HTMLResponse)
async def gui_jobs_start_get(request: Request, db: Session = Depends(get_db)):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    token = get_csrf_token(request) or gen_csrf_token()

    db_jobs = services.job_service.get_jobs(db, 0, 1000)
    jobs: List[Dict[str, Any]] = []
    for job in db_jobs:
        job_dict: Dict[str, Any] = {
            "id": job.id,
            "algorithm": job.algorithm,
            "name": job.name,
            "description": job.description,
            "participants": job.participants,
            "status": job.status,
            "parameters": {},
        }
        if job.parameters:
            if job.algorithm == "SIMI":
                ordered_params: Dict[str, Any] = {}
                if "target_column_index" in job.parameters:
                    ordered_params["target_column_index"] = job.parameters["target_column_index"]
                if "is_binary" in job.parameters:
                    ordered_params["is_binary"] = job.parameters["is_binary"]
                for k, v in job.parameters.items():
                    if k not in ordered_params:
                        ordered_params[k] = v
                job_dict["parameters"] = ordered_params
            elif job.algorithm == "SIMICE":
                ordered_params = {}
                if "target_column_indexes" in job.parameters:
                    ordered_params["target_column_indexes"] = job.parameters["target_column_indexes"]
                if "is_binary_list" in job.parameters:
                    ordered_params["is_binary_list"] = job.parameters["is_binary_list"]
                for k, v in job.parameters.items():
                    if k not in ordered_params:
                        ordered_params[k] = v
                job_dict["parameters"] = ordered_params
            else:
                job_dict["parameters"] = job.parameters
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

# Start Job using new runner
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
        print(f"db_job: {db_job}")
        # Only serialize the selected job (no need to load ALL jobs)
        job = db_job  # alias
        jobs = []
        job_dict = {
            "id": job.id,
            "algorithm": job.algorithm,
            "name": job.name,
            "description": job.description,
            "participants": job.participants,
            "status": job.status,
            "parameters": {},
        }
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
        job_dict["iteration_before_first_imputation"] = job.iteration_before_first_imputation
        job_dict["iteration_between_imputations"] = job.iteration_between_imputations
        job_dict["imputation_trials"] = job.imputation_trials
        job_dict["missing_spec"] = job.missing_spec
        job_dict["owner_id"] = job.owner_id
        jobs.append(job_dict)
        
        # Start with SIMI/SIMICE by calling algorithm central directly
        algorithm_name = (db_job.algorithm or "").lower()
        if algorithm_name not in ("simi", "simice"):
            return templates.TemplateResponse("start_job.html", {
                "request": request,
                "csrf_token": csrf_token,
                "jobs": jobs,
                "error": f"Unsupported algorithm {db_job.algorithm}"
            })
        print(f"Starting job {job_id} with algorithm {db_job.algorithm} for sites {db_job.participants}")
        print(f"DB Job parameters: {db_job.parameters}")
        
        # Build base config from stored parameters then normalize per algorithm expectations
        raw_params: Dict[str, Any] = dict(db_job.parameters or {})
        config: Dict[str, Any] = {}
        if algorithm_name == "simi":
            # Expected keys for SIMI central: M, mvar (0-based), method (Gaussian|logistic)
            # Source fields from stored params / job fields:
            # - imputation_trials -> M (fallback 1)
            # - target_column_index (assumed 1-based from UI) -> mvar (0-based)
            # - is_binary -> method logistic else Gaussian
            target_idx = raw_params.get("target_column_index")
            if target_idx is not None:
                # Accept either 0-based or 1-based: treat ints >0; if user provided 0 assume already 0-based
                try:
                    t_int = int(target_idx)
                    mvar = t_int - 1 if t_int > 0 else t_int
                except Exception:
                    mvar = target_idx
            else:
                mvar = raw_params.get("mvar")  # allow direct provision
            method = None
            if "is_binary" in raw_params:
                method = "logistic" if raw_params.get("is_binary") else "Gaussian"
            elif "method" in raw_params:
                method = raw_params["method"]
            config["M"] = db_job.imputation_trials or raw_params.get("M") or 1
            if mvar is not None:
                config["mvar"] = mvar
            if method is not None:
                config["method"] = method
            # Copy any other untouched params (avoid overwriting the normalized ones)
            for k, v in raw_params.items():
                if k not in ("target_column_index", "is_binary", "mvar", "method") and k not in config:
                    config[k] = v
            print(f"SIMI config constructed: {config}")        
        elif algorithm_name == "simice":
            # Simplified mapping for SIMICE expected keys: M, mvar (0-based list), type_list, iter_val, iter0_val
            def _int_list(val):
                if val is None:
                    return []
                if isinstance(val, str):
                    parts = [p.strip() for p in val.split(',') if p.strip()]
                else:
                    parts = list(val)
                out: List[int] = []
                for p in parts:
                    try:
                        iv = int(p)
                        out.append(iv - 1 if iv > 0 else iv)
                    except Exception:
                        continue
                return out

            # mvar sources: target_column_indexes (UI, 1-based) or existing mvar
            mvar_list = _int_list(raw_params.get("target_column_indexes") or raw_params.get("mvar"))

            # type_list: direct or derive from is_binary_list / is_binary
            if "type_list" in raw_params and isinstance(raw_params.get("type_list"), list):
                type_list = raw_params["type_list"]
            else:
                bin_list = None
                if "is_binary_list" in raw_params:
                    bin_list = raw_params["is_binary_list"]
                elif "is_binary" in raw_params and isinstance(raw_params.get("is_binary"), list):
                    bin_list = raw_params["is_binary"]
                elif "is_binary" in raw_params and mvar_list:
                    bin_list = [raw_params["is_binary"]] * len(mvar_list)
                type_list = ["logistic" if b else "Gaussian" for b in (bin_list or [])]

            # Iterations: prefer DB columns, then canonical keys, then GUI keys
            iter_val = (db_job.iteration_between_imputations
                        if db_job.iteration_between_imputations is not None else
                        raw_params.get("iter_val", raw_params.get("iteration_between_imputations")))
            iter0_val = (db_job.iteration_before_first_imputation
                         if db_job.iteration_before_first_imputation is not None else
                         raw_params.get("iter0_val", raw_params.get("iteration_before_first_imputation")))

            config.update({
                "M": db_job.imputation_trials or raw_params.get("M") or 1,
            })
            if mvar_list:
                config["mvar"] = mvar_list
            if type_list:
                config["type_list"] = type_list
            if iter_val is not None:
                config["iter_val"] = iter_val
            if iter0_val is not None:
                config["iter0_val"] = iter0_val

            # Pass through any extra keys (excluding GUI-only / duplicates)
            skip = {"target_column_indexes", "is_binary_list", "is_binary",
                    "iteration_before_first_imputation", "iteration_between_imputations",
                    "mvar", "type_list", "iter_val", "iter0_val"}
            for k, v in raw_params.items():
                if k in skip or k in config:
                    continue
                config[k] = v
            print(f"SIMICE config constructed: {config}")
        else:
            config = raw_params
        # participants hold site IDs
        site_ids: List[str] = list(db_job.participants or [])
        # prepare numpy matrix
        D = central_data.values

        # init job runtime record
        jobs_runtime[job_id] = {
            "running": True,
            "completed": False,
            "algorithm": db_job.algorithm,
            "sites": site_ids,
            "messages": ["Job created. Waiting for remotes..."]
        }

        async def run_central(job_id_local: int):
            # wait for remotes
            _set_expected_sites(site_ids)
            await all_sites_connected.wait()
            jobs_runtime[job_id_local]["messages"].append("All remotes connected. Starting...")
            imputation_running.set()
            try:
                import importlib
                # Debug: record raw and normalized config before invocation
                jobs_runtime[job_id_local]["messages"].append(f"DEBUG raw_params={raw_params}")
                jobs_runtime[job_id_local]["messages"].append(f"DEBUG config_pre_call={config}")
                # Fallback repair if SIMI missing required keys
                if algorithm_name == "simi":
                    missing_keys = [k for k in ("M","mvar","method") if k not in config]
                    if missing_keys:
                        # Attempt repair
                        tgt = raw_params.get("target_column_index")
                        if tgt is not None and "mvar" not in config:
                            try:
                                ti = int(tgt)
                                config["mvar"] = ti - 1 if ti > 0 else ti
                            except Exception:
                                pass
                        if "method" not in config:
                            if "is_binary" in raw_params:
                                config["method"] = "logistic" if raw_params.get("is_binary") else "Gaussian"
                        if "M" not in config:
                            config["M"] = db_job.imputation_trials or raw_params.get("M") or 1
                        jobs_runtime[job_id_local]["messages"].append(f"DEBUG repaired_config={config}")
                if algorithm_name == "simi":
                    algorithm_central = importlib.import_module("SIMI.SIMICentral").simi_central
                elif algorithm_name == "simice":
                    algorithm_central = importlib.import_module("SIMICE.SIMICECentral").simice_central
                else:
                    raise ValueError(f"Unsupported algorithm {algorithm_name}")
                imputed = await algorithm_central(D=D, config=config, site_ids=site_ids, websockets=remote_websockets)
                jobs_runtime[job_id_local]["messages"].append("Imputation completed.")
                # Persist results
                try:
                    # Normalize list of datasets
                    if isinstance(imputed, list):
                        datasets = imputed
                    else:
                        datasets = [imputed]
                    # Create a unique timestamped directory per run
                    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                    job_dir_name = f"job_{job_id_local}_{ts}"
                    job_dir = RESULTS_DIR / job_dir_name
                    job_dir.mkdir(exist_ok=True)
                    csv_files = []
                    for i, arr in enumerate(datasets, start=1):
                        try:
                            np_arr = np.asarray(arr)
                        except Exception:
                            continue
                        csv_path = job_dir / f"imputed_{i}.csv"
                        # If includes header row elsewhere we skip; here just raw matrix
                        np.savetxt(csv_path, np_arr, delimiter=",", fmt="%g")
                        csv_files.append(csv_path)
                    # Save metadata
                    meta = {
                        "algorithm": algorithm_name,
                        "config": config,
                        "num_imputations": len(csv_files),
                    }
                    with open(job_dir / "metadata.json", "w") as mf:
                        json.dump(meta, mf, indent=2)
                    # Create zip alongside directory with same timestamped base name
                    zip_path = RESULTS_DIR / f"{job_dir_name}.zip"
                    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                        for f in csv_files:
                            zf.write(f, arcname=f.name)
                        zf.write(job_dir / "metadata.json", arcname="metadata.json")
                    jobs_runtime[job_id_local]["result_info"] = {
                        "imputations": len(csv_files),
                        "zip_path": str(zip_path),
                        "run_timestamp": ts,
                        "job_dir": str(job_dir)
                    }
                    jobs_runtime[job_id_local]["zip_path"] = str(zip_path)
                    jobs_runtime[job_id_local]["job_dir"] = str(job_dir)
                    jobs_runtime[job_id_local]["run_timestamp"] = ts
                    # For legacy frontend naming expectation
                    jobs_runtime[job_id_local]["imputed_dataset_path"] = str(zip_path)
                    jobs_runtime[job_id_local]["messages"].append(f"Results saved: {zip_path.name} (dir {job_dir_name})")
                except Exception as save_e:
                    jobs_runtime[job_id_local]["messages"].append(f"Result save error: {save_e}")
            except Exception as e:
                jobs_runtime[job_id_local]["messages"].append(f"Error: {e}")
                jobs_runtime[job_id_local]["error"] = str(e)
            finally:
                jobs_runtime[job_id_local]["completed"] = True
                jobs_runtime[job_id_local]["running"] = False
                imputation_running.clear()

        asyncio.create_task(run_central(job_id))

        return templates.TemplateResponse("start_job.html", {
            "request": request, 
            "csrf_token": csrf_token, 
            "jobs": jobs,
            "active_job_id": job_id,
            "message": f"Job {job_id} started. Monitoring status..."
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
            params['is_binary_list'] = bins
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
# Removed deprecated /api/jobs/{job_id}/start endpoint (duplicate of GUI POST /gui/jobs/start)

@app.get("/api/jobs/status/{job_id}")
async def api_job_status(job_id: int):
    info = jobs_runtime.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return info

@app.get("/api/jobs/{job_id}/download")
async def api_job_download(job_id: int):
    info = jobs_runtime.get(job_id)
    # Primary: use in-memory runtime if available and has existing zip
    print(f"Download request job_id={job_id} info={info}")
    
    if info:
        zip_path = info.get("zip_path")
        if zip_path and Path(zip_path).exists():
            logger.info(f"Download (primary) job_id={job_id} path={zip_path}")
            return FileResponse(zip_path, filename=Path(zip_path).name, media_type="application/zip")
    # Fallback: search results directory for latest matching zip (survives server restarts)
    pattern = f"job_{job_id}_*.zip"
    matches = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.name, reverse=True)
    logger.info(f"Download fallback search job_id={job_id} pattern={pattern} matches={[m.name for m in matches]}")
    if matches:
        latest = matches[0]
        # Optionally repopulate runtime cache
        jobs_runtime.setdefault(job_id, {"completed": True, "running": False})
        jobs_runtime[job_id]["zip_path"] = str(latest)
        jobs_runtime[job_id]["imputed_dataset_path"] = str(latest)
        # Persist to DB legacy field if job exists (best-effort)
        try:
            from .services import job_service
            with next(get_db()) as _db:
                db_job = job_service.get_job(_db, job_id)
                if db_job:
                    db_job.imputed_dataset_path = str(latest)
                    if db_job.status != "completed":
                        db_job.status = "completed"
                    _db.add(db_job)
                    _db.commit()
        except Exception:
            pass
        logger.info(f"Download (fallback) job_id={job_id} path={latest}")
        return FileResponse(str(latest), filename=latest.name, media_type="application/zip")
    logger.warning(f"Download 404 job_id={job_id} RESULTS_DIR={RESULTS_DIR} contents={[p.name for p in RESULTS_DIR.glob('*.zip')]}")
    raise HTTPException(status_code=404, detail="Result archive not found")

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
