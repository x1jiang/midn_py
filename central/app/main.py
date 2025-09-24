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
import contextlib

from . import models, schemas, services
from .db import get_db, engine
from .core.config import settings
from .core.alg_config import load_all_algorithm_schemas, validate_parameters
from .core.csrf import get_csrf_token, gen_csrf_token, CSRF_COOKIE
from .api import users as sites_api, jobs, remote as remote_api
from starlette.middleware.base import BaseHTTPMiddleware

# Keep DB models

models.Base.metadata.create_all(bind=engine)

"""All SIMICE jobs now store multi-column binary flags under 'is_binary_list'.
Legacy automatic migration from 'is_binary' removed (schema updated). Re-create
old jobs if they referenced the deprecated key."""

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

# Helper: order parameters according to algorithm schema ui:order (similar to create_job rendering)
def _order_parameters(algo: str, params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return parameters ordered according to the algorithm schema ui:order.

    Previous compatibility that aliased SIMICE 'is_binary' -> 'is_binary_list' removed.
    Jobs must already use 'is_binary_list'. We keep keys exactly as stored.
    """
    if not params:
        return {}
    schemas = load_all_algorithm_schemas()
    schema = schemas.get(algo)
    if not schema:
        return params
    order = schema.get('ui:order') or []
    if not isinstance(order, list) or not order:
        return params
    ordered: Dict[str, Any] = {}
    seen: Set[str] = set()
    for key in order:
        if key in params:
            ordered[key] = params[key]
            seen.add(key)
    for k in sorted(params.keys()):
        if k not in seen:
            ordered[k] = params[k]
    print(f"_order_parameters algo={algo} order={order} input_keys={list(params.keys())} ordered_keys={list(ordered.keys())}")
    return ordered

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
# Track asyncio Task objects for running jobs so we can cancel them
job_tasks: Dict[int, asyncio.Task] = {}
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
    # Clear first (fresh wait state) then immediately evaluate current connections
    all_sites_connected.clear()
    try:
        if expected_sites and expected_sites.issubset(set(remote_websockets.keys())):
            # All required sites already connected (they connected before job start)
            all_sites_connected.set()
            logger.info(f"All expected sites already connected: {sorted(expected_sites)}; proceeding without additional waits.")
    except Exception as e:
        # Non-critical safeguard; failure here should not block job start
        logger.warning(f"_set_expected_sites early evaluation error: {e}")

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

# Demo page (GUI)
@app.get("/demo", response_class=HTMLResponse)
async def gui_demo(request: Request):
    if not is_admin(request):
        return RedirectResponse(url="/gui/login")
    token = get_csrf_token(request) or gen_csrf_token()
    resp = templates.TemplateResponse("demo.html", {"request": request, "csrf_token": token})
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
    # Derive external base URL (scheme+host)
    base = str(request.base_url).rstrip('/')
    if base.startswith("http://") and request.headers.get("X-Forwarded-Proto") == "https":
        base = base.replace("http://", "https://", 1)
    db_user = services.user_service.approve_user(db, user_id, central_url=base)
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
        "alg_schemas": load_all_algorithm_schemas(),
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
    # Dynamic params will arrive; keep legacy fields but treat generically
    # legacy field removed; if needed must be in schema now
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

    # Collect raw form data for dynamic validation
    form = await request.form()
    raw_map = {k: v for k, v in form.items() if k not in {"algorithm","name","description","participants","csrf_token"}}
    try:
        params = validate_parameters(algo, raw_map)
    except ValueError as ve:
        return HTMLResponse(str(ve), status_code=400)
    # Basic missing spec concept removed; all data stays in parameters

    job = schemas.JobCreate(
        name=name,
        description=description,
        algorithm=algo,
        parameters=params,
        participants=participants,
    # removed missing_spec/imputation_trials
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
            ordered_schema = _order_parameters(job.algorithm, dict(job.parameters))
            job_dict["parameters"] = ordered_schema
            job_dict["param_order"] = list(ordered_schema.keys())
        # include owner id per job and append within loop so all jobs retained
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
            ordered_schema = _order_parameters(job.algorithm, dict(job.parameters))
            job_dict["parameters"] = ordered_schema
            job_dict["param_order"] = list(ordered_schema.keys())
    # legacy iteration/imputation/missing_spec removed
        job_dict["owner_id"] = job.owner_id
        jobs.append(job_dict)
        
        # Start with valid algorithm by calling algorithm central directly
        algorithm_name = (db_job.algorithm or "").lower()
        if algorithm_name.lower() not in [a.lower() for a in settings._ALG]:
             return templates.TemplateResponse("start_job.html", {
                "request": request,
                "csrf_token": csrf_token,
                "jobs": jobs,
                "error": f"Unsupported algorithm {db_job.algorithm}"
            })
        print(f"Starting job {job_id} with algorithm {db_job.algorithm} for sites {db_job.participants}")
        print(f"DB Job parameters: {db_job.parameters}")
        
        # Build base config from stored parameters; pass raw to algorithm modules
        raw_params: Dict[str, Any] = dict(db_job.parameters or {})
        # Pass raw parameters directly; algorithm modules will handle any normalization
        config = raw_params
        print(f"Passing raw parameters to algorithm ({algorithm_name}): {config}")
        # participants hold site IDs
        site_ids: List[str] = list(db_job.participants or [])
        # prepare numpy matrix
        # Coerce to float64 early; if non-numeric columns exist, attempt conversion and raise clear error
        try:
            # Identify object / string dtypes and attempt to coerce
            non_numeric_cols = []
            for c in central_data.columns:
                if central_data[c].dtype == object:
                    non_numeric_cols.append(c)
            if non_numeric_cols:
                coercion_errors: List[str] = []
                for c in non_numeric_cols:
                    try:
                        central_data[c] = pd.to_numeric(central_data[c], errors='raise')
                    except Exception as ce:  # keep original for debugging
                        coercion_errors.append(f"{c}: {ce}")
                if coercion_errors:
                    return templates.TemplateResponse("confirm.html", {"request": request, "message": (
                        "Non-numeric column(s) detected that could not be converted to numeric required for SIMICE/SIMI: "
                        + "; ".join(coercion_errors)
                    )})
            D = central_data.to_numpy(dtype=float, copy=False)
        except Exception as conv_e:
            return templates.TemplateResponse("confirm.html", {"request": request, "message": f"Failed to coerce data to float64: {conv_e}"})

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
            print(f"Job {job_id_local} waiting for remotes: {site_ids}", flush=True)
            print( f"Currently connected remotes: {list(remote_websockets.keys())}", flush=True)
            
            _set_expected_sites(site_ids)
            await all_sites_connected.wait()
            jobs_runtime[job_id_local]["messages"].append("All remotes connected. Starting...")
            imputation_running.set()
            try:
                import importlib
                # Debug: record raw and normalized config before invocation
                #jobs_runtime[job_id_local]["messages"].append(f"DEBUG raw_params={raw_params}")
                #jobs_runtime[job_id_local]["messages"].append(f"DEBUG config_pre_call={config}")
                if algorithm_name == "simi":
                    algorithm_central = importlib.import_module("SIMI.SIMICentral").simi_central
                elif algorithm_name == "simice":
                    algorithm_central = importlib.import_module("SIMICE.SIMICECentral").simice_central
                elif algorithm_name == "avgmmi":
                    algorithm_central = importlib.import_module("AVGMMI.AVGMMICentral").avgmmi_central        
                elif algorithm_name == "avgmmice":
                    algorithm_central = importlib.import_module("AVGMMICE.AVGMMICECentral").avgmmice_central   
                elif algorithm_name == "hdmi":
                    algorithm_central = importlib.import_module("HDMI.HDMICentral").hdmi_central        
                elif algorithm_name == "cslmi":
                    algorithm_central = importlib.import_module("CSLMI.CSLMICentral").cslmi_central        
                elif algorithm_name == "cslmice":
                    algorithm_central = importlib.import_module("CSLMICE.CSLMICECentral").cslmice_central        
                elif algorithm_name == "imi":
                    algorithm_central = importlib.import_module("IMI.IMICentral").imi_central        
                elif algorithm_name == "imice":
                    algorithm_central = importlib.import_module("IMICE.IMICECentral").imice_central
                else:
                    raise ValueError(f"Unsupported algorithm {algorithm_name}")
                # Real-time streaming capture of algorithm prints
                class _StreamingCapture:
                    def __init__(self, job_id: int, kind: str):
                        self._job_id = job_id
                        self._kind = kind
                        self._buf = ""
                        self.encoding = "utf-8"
                    def write(self, s: str):
                        if s is None:
                            return 0
                        self._buf += s
                        while "\n" in self._buf:
                            line, self._buf = self._buf.split("\n", 1)
                            line = line.rstrip()
                            if line.strip():
                                try:
                                    jobs_runtime[self._job_id]["messages"].append(f"[algo {self._kind}] {line}")
                                except Exception:
                                    pass
                        return len(s)
                    def flush(self):
                        if self._buf.strip():
                            try:
                                jobs_runtime[self._job_id]["messages"].append(f"[algo {self._kind}] {self._buf.strip()}")
                            except Exception:
                                pass
                            self._buf = ""
                    def isatty(self):
                        return False
                orig_stdout, orig_stderr = sys.stdout, sys.stderr
                sys.stdout = _StreamingCapture(job_id_local, "stdout")  # type: ignore
                sys.stderr = _StreamingCapture(job_id_local, "stderr")  # type: ignore
                try:
                    # All normalization of config is deferred to the algorithm implementation
                    imputed = await algorithm_central(D=D, config=config, site_ids=site_ids, websockets=remote_websockets,debug=True)
                finally:
                    # Flush remaining partial lines
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception:
                        pass
                    sys.stdout, sys.stderr = orig_stdout, orig_stderr
                jobs_runtime[job_id_local]["messages"].append("Imputation completed.")
                # Persist results
                try:
                    # Normalize list of datasets
                    if isinstance(imputed, list):
                        datasets = imputed
                    else:
                        datasets = [imputed]
                    # Create a unique timestamped directory per run
                    # Use local server time instead of UTC for directory/zip naming
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
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
                    # Persist path + mark completed early (best-effort) so UI tables update promptly
                    try:
                        from .services import job_service  # local import to avoid cycles
                        with next(get_db()) as _db:
                            db_job_early = job_service.get_job(_db, job_id_local)
                            if db_job_early:
                                db_job_early.imputed_dataset_path = str(zip_path)
                                # Set status only if not already error/cancelled
                                if not jobs_runtime[job_id_local].get("error") and not jobs_runtime[job_id_local].get("cancelled"):
                                    db_job_early.status = "completed"
                                _db.add(db_job_early)
                                _db.commit()
                                jobs_runtime[job_id_local]["messages"].append("Database updated with result path.")
                    except Exception as early_db_e:
                        jobs_runtime[job_id_local]["messages"].append(f"Early DB update failed: {early_db_e}")
                except Exception as save_e:
                    jobs_runtime[job_id_local]["messages"].append(f"Result save error: {save_e}")
            except asyncio.CancelledError:
                jobs_runtime[job_id_local]["messages"].append("Job cancelled by user.")
                jobs_runtime[job_id_local]["cancelled"] = True
                jobs_runtime[job_id_local]["error"] = "cancelled"
                raise
            except Exception as e:
                jobs_runtime[job_id_local]["messages"].append(f"Error: {e}")
                jobs_runtime[job_id_local]["error"] = str(e)
            finally:
                jobs_runtime[job_id_local]["completed"] = True
                jobs_runtime[job_id_local]["running"] = False
                imputation_running.clear()
                # Persist status to DB (best effort)
                try:
                    from .services import job_service
                    with next(get_db()) as _db:
                        db_job_local = job_service.get_job(_db, job_id_local)
                        if db_job_local:
                            if jobs_runtime[job_id_local].get("cancelled"):
                                db_job_local.status = "cancelled"
                            elif jobs_runtime[job_id_local].get("error") and jobs_runtime[job_id_local]["error"] != "cancelled":
                                db_job_local.status = "error"
                            else:
                                db_job_local.status = "completed"
                            _db.add(db_job_local)
                            _db.commit()
                except Exception:
                    pass

        task = asyncio.create_task(run_central(job_id))
        job_tasks[job_id] = task

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
        "approved_users": approved,
        "alg_schemas": load_all_algorithm_schemas(),
    })
    if not get_csrf_token(request):
        resp.set_cookie(CSRF_COOKIE, token, httponly=False, samesite="lax")
    return resp

@app.post("/gui/jobs/{job_id}/edit", response_class=HTMLResponse)
async def gui_jobs_edit_post(request: Request, job_id: int,
                             name: str = Form(None), description: str = Form(None),
                             participants: list[str] = Form([]),
                            # legacy imputation_trials removed
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
    # Update only supported simple fields; trials now live inside parameters if defined by schema
    updated = services.job_service.update_job(db, job_id, name=name, description=description,
                                              participants=participants)
    if not updated:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Job not found."})

    algo = (job.algorithm or '').upper()
    form = await request.form()
    raw_map = {k: v for k, v in form.items() if k not in {"name","description","participants","csrf_token"}}
    try:
        params = validate_parameters(algo, raw_map)
    except ValueError as ve:
        return HTMLResponse(str(ve), status_code=400)
    # legacy imputation_trials/missing_spec removed

    # Assign updated fields
    updated.parameters = params
    # removed legacy assignments
    
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
    # Redirect back to jobs list after deletion
    return RedirectResponse(url="/gui/jobs", status_code=303)

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

@app.post("/api/jobs/stop/{job_id}")
async def api_job_stop(job_id: int, request: Request):
    """Cancel a running job.

    CSRF header required (X-CSRF-Token). Middleware skips generic check for this path,
    so we validate explicitly here to allow custom error codes.
    """
    token = request.headers.get("X-CSRF-Token")
    if not token or token != get_csrf_token(request):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    info = jobs_runtime.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    task = job_tasks.get(job_id)
    if info.get("completed") or not task or task.done():
        # Already finished; normalize cancelled flag if error == cancelled
        if info.get("cancelled") and info.get("running"):
            info["running"] = False
        return {"detail": "Job already finished", "job": info}
    # Mark intent immediately so UI reflects change even if task is mid-await
    info["messages"].append("Cancellation requested by user...")
    info["cancelled"] = True
    info["running"] = False
    # We set completed True so polling UI stops; backend may still be unwinding
    info["completed"] = True
    info["error"] = "cancelled"
    task.cancel()
    # Fire-and-wait briefly but don't block indefinitely
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    # Persist status to DB best effort
    try:
        from .services import job_service
        with next(get_db()) as _db:
            db_job_local = job_service.get_job(_db, job_id)
            if db_job_local:
                db_job_local.status = "cancelled"
                _db.add(db_job_local)
                _db.commit()
    except Exception:
        pass
    return {"detail": "Job cancellation processed", "job": info}

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
    base = str(request.base_url).rstrip('/')
    if base.startswith("http://") and request.headers.get("X-Forwarded-Proto") == "https":
        base = base.replace("http://", "https://", 1)
    db_user = services.user_service.approve_user(db, user_id, central_url=base, expires_days=expires_days)
    if not db_user:
        return templates.TemplateResponse("confirm.html", {"request": request, "message": "Site not found."})
    exp_ts = (datetime.utcnow() + timedelta(days=expires_days)).strftime("%Y-%m-%d %H:%M:%S UTC")
    return templates.TemplateResponse("confirm.html", {"request": request, "message": f"Approved site {db_user.username} (site_id={db_user.site_id}). JWT expires at {exp_ts}. Email sent."})
