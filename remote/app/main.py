from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import pandas as pd
import httpx
from datetime import datetime, timezone
from jose import jwt
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from pathlib import Path
import os
import sys
import traceback

from . import config
from .custom_templates import templates
from .job_status import JobStatusCallback
from .routes import jobs as jobs_routes

app = FastAPI()

app.mount("/static", StaticFiles(directory="remote/app/static"), name="static")

# Include job status routes
app.include_router(jobs_routes.router)

# ---- Child-process launchers to avoid nested event loop issues ----
# Legacy process launch helpers removed; async tasks are now always used.

def _token_expiry_str(token: str | None) -> str | None:
    if not token:
        return None
    try:
        claims = jwt.get_unverified_claims(token)
        exp = claims.get("exp")
        if exp:
            return datetime.fromtimestamp(int(exp), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return None
    return None


async def get_site_info(site_id, active_site=None):
    """Try to get the site information from the central server, using the active site's token."""
    try:
        if site_id and site_id != "my_site_id" and active_site and active_site.TOKEN:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{active_site.HTTP_URL}/api/remote/info",
                    params={"site_id": site_id},
                    headers={"Authorization": f"Bearer {active_site.TOKEN}"}
                )
                if r.status_code == 200:
                    return r.json()
    except Exception as e:
        print(f"Error fetching site info: {e}")
    return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, refresh: bool = False, site_index: int = 0):
    # Reload settings from disk to reflect external updates
    try:
        config.settings = config.load_settings()
    except Exception:
        pass
    jobs = []
    site_info = None
    message = None
    
    # Determine the active site without modifying the sites list
    active_site = None
    if 0 <= site_index < len(config.settings.sites):
        active_site = config.settings.sites[site_index]
    else:
        # Default to first site if index is invalid
        active_site = config.settings.sites[0] if config.settings.sites else None
    
    # If active site is configured and has valid credentials, fetch available jobs
    if active_site and active_site.SITE_ID and active_site.SITE_ID != "my_site_id" and active_site.TOKEN and active_site.TOKEN != "my_jwt_token":
        try:
            # Use HTTP URL for API calls
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{active_site.HTTP_URL}/api/remote/jobs",
                                    params={"site_id": active_site.SITE_ID},
                                    headers={"Authorization": f"Bearer {active_site.TOKEN}"})
                if r.status_code == 200:
                    jobs = r.json()
                    print(f"Successfully fetched {len(jobs)} jobs for site {active_site.SITE_ID}")
                    if refresh:
                        message = f"Data refreshed successfully. Found {len(jobs)} jobs assigned to this site."
                    # Only try to get site information if jobs API call succeeds (token is valid)
                    site_info = await get_site_info(active_site.SITE_ID, active_site)
                else:
                    print(f"Failed to fetch jobs: HTTP {r.status_code}")
                    if refresh:
                        message = f"Failed to refresh data. Server returned status code {r.status_code}."
        except Exception as e:
            print(f"Error fetching jobs: {e}")
            if refresh:
                message = f"Error refreshing data: {str(e)}"
    else:
        print("Site not configured properly. SITE_ID or TOKEN is not set.")
        if refresh:
            message = "Site not configured properly. Please set your Site ID and Token in Settings."
    
    site_name = None
    institution = None
    
    if site_info:
        site_name = site_info.get("name")
        institution = site_info.get("institution")
    
    active_site_name = active_site.name if active_site else "Default Site"
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "settings": active_site if active_site else config.settings,  # Use active site settings
        "sites": config.settings.sites,
        "active_site": active_site_name,
        "active_site_index": site_index,  # Pass the current site index
        "jobs": jobs, 
        "site_name": site_name or "Site Not Authenticated",
        "institution": institution or "",
        "token_expiry": _token_expiry_str(active_site.TOKEN if active_site else None),
        "message": message
    })

@app.get("/settings", response_class=HTMLResponse)
async def settings_get(request: Request, site_index: int = 0):
    # Always reload settings from disk to reflect external updates
    # (the settings file may be shared across multiple instances)
    try:
        config.settings = config.load_settings()
    except Exception as _e:
        # Best-effort reload; fall back to existing in-memory settings
        pass

    # Determine the active site without modifying the sites list
    active_site = None
    if 0 <= site_index < len(config.settings.sites):
        active_site = config.settings.sites[site_index]
    else:
        # Default to first site if index is invalid
        active_site = config.settings.sites[0] if config.settings.sites else None
    
    active_site_name = active_site.name if active_site else "Default Site"
    return templates.TemplateResponse("settings.html", {
        "request": request, 
        "settings": active_site if active_site else config.settings, 
        "sites": config.settings.sites,
        "active_site": active_site_name,
        "active_site_index": site_index,
        "token_expiry": _token_expiry_str(active_site.TOKEN if active_site else None)
    })

@app.post("/settings", response_class=HTMLResponse)
async def settings_post(request: Request,
                        site_name: str = Form(...),
                        central_url: str = Form(...),
                        site_id: str = Form(...),
                        token: str = Form(...)):
    # Parse the URL to determine protocol (http/https)
    if central_url.startswith('http://') or central_url.startswith('https://'):
        # Convert HTTP URL to WS URL
        ws_url = central_url.replace('http://', 'ws://').replace('https://', 'wss://')
        http_url = central_url
    else:
        # If no protocol specified, default to http/ws
        http_url = f"http://{central_url}"
        ws_url = f"ws://{central_url}"
    
    # Find or create the site configuration
    site_found = False
    for site in config.settings.sites:
        if site.name == site_name:
            # Update existing site
            site.CENTRAL_URL = ws_url
            site.HTTP_URL = http_url
            site.SITE_ID = site_id
            site.TOKEN = token
            site_found = True
            break
    
    if not site_found:
        # Create a new site
        new_site = config.SiteConfig(
            name=site_name,
            central_url=ws_url,
            http_url=http_url,
            site_id=site_id,
            token=token
        )
        config.settings.sites.append(new_site)
    
    # Save settings to the config file
    config.save_settings(config.settings)
    
    # Redirect to the index page with the site_index query parameter
    # Find the index of the site that was just created or updated
    for i, site in enumerate(config.settings.sites):
        if site.name == site_name:
            return RedirectResponse(url=f"/?site_index={i}", status_code=303)
    
    return RedirectResponse(url="/", status_code=303)

@app.get("/get_jobs")
async def get_jobs(site_id: Optional[str] = None, site_index: int = 0):
    """Endpoint that returns job data as JSON for AJAX requests"""
    # Reload settings from disk to reflect external updates
    try:
        config.settings = config.load_settings()
    except Exception:
        pass
    jobs = []
    
    # Determine the active site
    active_site = None
    if 0 <= site_index < len(config.settings.sites):
        active_site = config.settings.sites[site_index]
    else:
        # Default to first site if index is invalid
        active_site = config.settings.sites[0] if config.settings.sites else None
    
    # Use provided site_id, or get it from the active site
    site_id_to_use = site_id or (active_site.SITE_ID if active_site else None)
    
    if active_site and site_id_to_use and site_id_to_use != "my_site_id" and active_site.TOKEN and active_site.TOKEN != "my_jwt_token":
        try:
            # Use HTTP URL for API calls
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{active_site.HTTP_URL}/api/remote/jobs",
                                    params={"site_id": site_id_to_use},
                                    headers={"Authorization": f"Bearer {active_site.TOKEN}"})
                if r.status_code == 200:
                    jobs = r.json()
                    print(f"Successfully fetched {len(jobs)} jobs for site {site_id_to_use} via API")
        except Exception as e:
            print(f"Error fetching jobs via API: {e}")
    
    return jobs

@app.get("/jobs", response_class=HTMLResponse)
async def list_jobs(request: Request, refresh: bool = False, site_index: int = 0):
    # Reload settings from disk to reflect external updates
    try:
        config.settings = config.load_settings()
    except Exception:
        pass
    jobs = []
    site_info = None
    message = None
    
    # Determine the active site without modifying the sites list
    active_site = None
    if 0 <= site_index < len(config.settings.sites):
        active_site = config.settings.sites[site_index]
    else:
        # Default to first site if index is invalid
        active_site = config.settings.sites[0] if config.settings.sites else None
    
    if not active_site:
        message = "No site configuration available."
    else:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{active_site.HTTP_URL}/api/remote/jobs",
                                    params={"site_id": active_site.SITE_ID},
                                    headers={"Authorization": f"Bearer {active_site.TOKEN}"})
                if r.status_code == 200:
                    jobs = r.json()
                    if refresh:
                        message = f"Data refreshed successfully. Found {len(jobs)} assigned jobs."
                    
                    # Only try to get site information if jobs API call succeeds (token is valid)
                    site_info = await get_site_info(active_site.SITE_ID, active_site.TOKEN)
                elif refresh:
                    message = f"Failed to refresh data. Server returned status code {r.status_code}."
        except Exception as e:
            print(f"Error fetching jobs: {e}")
            if refresh:
                message = f"Error refreshing data: {str(e)}"
    
    site_name = None
    institution = None
    
    if site_info:
        site_name = site_info.get("name")
        institution = site_info.get("institution")
    
    active_site_name = active_site.name if active_site else "Default Site"
    
    return templates.TemplateResponse("jobs.html", {
        "request": request, 
        "jobs": jobs, 
        "settings": active_site if active_site else config.settings,
        "sites": config.settings.sites,
        "active_site": active_site_name,
        "active_site_index": site_index,
        "site_name": site_name or "Site Not Authenticated",
        "institution": institution or "",
        "token_expiry": _token_expiry_str(active_site.TOKEN if active_site else None),
        "message": message
    })

def has_running_jobs(app, site_id=None):
    """Check if there are any running jobs for a specific site or any site"""
    # If site_id is provided, only check that site's jobs
    if site_id:
        site_jobs_attr = f'running_jobs_{site_id}'
        site_jobs = getattr(app.state, site_jobs_attr, {})
        return any(not job.get('completed', False) for job in site_jobs.values())
    
    # Otherwise check all sites for running jobs
    running_jobs = False
    for attr_name in dir(app.state):
        if attr_name.startswith('running_jobs_'):
            site_jobs = getattr(app.state, attr_name, {})
            if any(not job.get('completed', False) for job in site_jobs.values()):
                running_jobs = True
                break
    return running_jobs

@app.get("/debug/all_jobs")
async def debug_all_jobs(request: Request):
    """Debug endpoint to show all jobs across all sites"""
    result = {
        'app_state_attributes': [attr for attr in dir(request.app.state) if not attr.startswith('_')],
        'running_jobs_attributes': [],
        'sites': {}
    }
    
    # Instead of using dir(), try to access known site IDs from config
    for site in config.settings.sites:
        site_id = site.SITE_ID
        site_jobs_attr = f'running_jobs_{site_id}'
        
        # Check if this site has any jobs
        if hasattr(request.app.state, site_jobs_attr):
            result['running_jobs_attributes'].append(site_jobs_attr)
            site_jobs = getattr(request.app.state, site_jobs_attr, {})
            result['sites'][site_id] = {
                'attribute_name': site_jobs_attr,
                'job_count': len(site_jobs),
                'jobs': {job_id: {
                    'status': job.get('status', 'Unknown'),
                    'completed': job.get('completed', False),
                    'start_time': str(job.get('start_time', 'Unknown'))
                } for job_id, job in site_jobs.items()}
            }
        else:
            result['sites'][site_id] = {
                'attribute_name': site_jobs_attr,
                'job_count': 0,
                'jobs': {}
            }
    
    return result

@app.get("/debug/app_state")
async def debug_app_state(request: Request):
    """Debug endpoint to show raw app state"""
    # Test if app.state works at all
    if not hasattr(request.app.state, 'test_attribute'):
        request.app.state.test_attribute = "app.state is working"
    
    state_dict = {}
    for attr_name in dir(request.app.state):
        if not attr_name.startswith('_'):
            attr_value = getattr(request.app.state, attr_name)
            if hasattr(attr_value, '__dict__'):
                state_dict[attr_name] = str(attr_value)
            elif isinstance(attr_value, (dict, list, str, int, float, bool)):
                state_dict[attr_name] = attr_value
            else:
                state_dict[attr_name] = str(type(attr_value))
    return {
        'state_attributes': state_dict,
        'total_attributes': len([attr for attr in dir(request.app.state) if not attr.startswith('_')]),
        'app_state_type': str(type(request.app.state))
    }

@app.post("/start_job")
async def start_job(
    request: Request,
    job_id: int = Form(...),
    mvar: int = Form(...),
    csv_file: UploadFile = File(...),
    job_type: Optional[str] = Form(None),
    iteration_before_first_imputation: Optional[int] = Form(None),
    iteration_between_imputations: Optional[int] = Form(None),
    site_index: Optional[int] = Form(0),
    current_site_id: Optional[str] = Form(None)
):
    print(f"🚀 START_JOB CALLED: ID={job_id}, Type={job_type}, Missing var={mvar}, Site index={site_index}")
    print(f"🚀 Request type: {type(request)}")
    print(f"🚀 App instance: {id(request.app)}")
    
    # Get the active site without modifying the sites list
    active_site = None
    if 0 <= site_index < len(config.settings.sites):
        active_site = config.settings.sites[site_index]
    else:
        # Default to first site if index is invalid
        active_site = config.settings.sites[0] if config.settings.sites else None
        
    if not active_site:
        error_message = "No site configuration available."
        print(f"Error starting job: {error_message}")
        return {"error": error_message}
        
    print(f"Using site configuration: {active_site.name}, ID={active_site.SITE_ID}")
            
    # Verify that current site ID matches the expected site ID
    if current_site_id and current_site_id != active_site.SITE_ID:
        print(f"Warning: Current site ID from form ({current_site_id}) doesn't match active site ID ({active_site.SITE_ID})")
    
    # Check for running jobs - if any job is running, block starting a new one
    if has_running_jobs(app, active_site.SITE_ID):
        error_message = "A job is already running on this site. Please wait for it to complete or stop it before starting a new job."
        print(f"Error starting job: {error_message}")
        
        # Check if the request is AJAX or regular form submission
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Accept', '').startswith('application/json')
        if is_ajax:
            return {"success": False, "error": error_message}
        else:
            return templates.TemplateResponse("error.html", {
                "request": request, 
                "error": error_message
            })
    
    # Check if the request is AJAX or regular form submission
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Accept', '').startswith('application/json')
    
    try:
        df = pd.read_csv(csv_file.file)
        print(f"CSV file loaded successfully: {csv_file.filename}, Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        # Return appropriate error response
        if is_ajax:
            return {"success": False, "error": f"Error loading CSV file: {str(e)}"}
        else:
            return templates.TemplateResponse("error.html", {
                "request": request, 
                "error": f"Error loading CSV file: {str(e)}"
            })
    
    # Get the job details to access additional parameters
    job_details = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{active_site.HTTP_URL}/api/remote/jobs",
                                params={"site_id": active_site.SITE_ID},
                                headers={"Authorization": f"Bearer {active_site.TOKEN}"})
            if r.status_code == 200:
                jobs = r.json()
                for job in jobs:
                    if job["id"] == job_id:
                        job_details = job
                        print(f"Found job details: Algorithm={job['algorithm']}, Parameters={job['parameters']}")
                        break
    except Exception as e:
        print(f"Error fetching job details: {e}")
    
    # Determine high-level algorithm type early (job_details preferred)
    algo = (job_details.get("algorithm") if job_details else job_type or "SIMI").upper()
    params_from_job = (job_details or {}).get("parameters", {}) if job_details else {}
    # Build a parameters dict to pass directly to async_run_remote_client; SIMI will validate internally now.
    # For backward compatibility we still keep the originally submitted mvar in parameters if central didn't supply one.
    parameters: Dict[str, Any] = dict(params_from_job)  # shallow copy
     
    if df.shape[1] == 0:
        return templates.TemplateResponse("error.html", {"request": request, "error": "Uploaded CSV has no columns."})
    # Do not perform detailed column-bound validation here; moved to SIMIRemote.async_run_remote_client.
    print(f"[start_job] Delegated parameter validation to remote client. Parameters passed: {parameters}")
    
    # Create task based on job type with streaming capture & proper error handling
    try:
        # Prepare sys.path to import MIDN_R_PY implementations
        sys.path.append(str(Path(__file__).resolve().parents[2] / "MIDN_R_PY"))

        # Parse CENTRAL_URL for host/port
        ws = urlparse(active_site.CENTRAL_URL)
        central_host = ws.hostname or "localhost"
        central_port = ws.port or (443 if ws.scheme == "wss" else 80)
        central_proto = ws.scheme or "ws"

        # Persist CSV to a temp file so algorithm can read it
        jobs_dir = Path("remote_runtime/jobs")
        jobs_dir.mkdir(parents=True, exist_ok=True)
        data_path = jobs_dir / f"job_{job_id}_{active_site.SITE_ID}.csv"
        df.to_csv(data_path, index=False)
        print(f"Saved job data to {data_path}")

        # Registry for site jobs (create early so capture can write)
        site_jobs_attr = f'running_jobs_{active_site.SITE_ID}'
        if not hasattr(request.app.state, site_jobs_attr):
            setattr(request.app.state, site_jobs_attr, {})
        site_jobs = getattr(request.app.state, site_jobs_attr)
        site_jobs[job_id] = {
            'status': 'Running',
            'messages': ['Job started successfully'],
            'start_time': datetime.now(),
            'completed': False,
            'end_time': None,
            'site_id': active_site.SITE_ID,
            'data_path': str(data_path),
            'pid': None,
            'async': True,
            'task': None
        }
        job_entry = site_jobs[job_id]

        class _StreamingCapture:
            def __init__(self, kind: str):
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
                            job_entry['messages'].append(f"[remote {self._kind}] {line}")
                        except Exception:
                            pass
                return len(s)
            def flush(self):
                if self._buf.strip():
                    try:
                        job_entry['messages'].append(f"[remote {self._kind}] {self._buf.strip()}")
                    except Exception:
                        pass
                    self._buf = ""
            def isatty(self):
                return False

        import importlib
        # Use the shared Core runner with the algorithm-specific client class
        core_mod = importlib.import_module("Core.remote_core")
        run_core_async = getattr(core_mod, "run_remote_client_async")

        if algo == "SIMICE":
            parameters.setdefault("job_id", job_id)
            parameters.setdefault("site_id", active_site.SITE_ID)
            simice_mod = importlib.import_module("SIMICE.SIMICERemote")
            ClientClass = getattr(simice_mod, "SIMICERemoteClient")
            remote_coro = run_core_async(
                ClientClass,
                str(data_path),
                central_host,
                central_port,
                central_proto,
                active_site.SITE_ID,
                parameters,
            )
        elif algo == "SIMI":
            parameters.setdefault("job_id", job_id)
            parameters.setdefault("site_id", active_site.SITE_ID)
            simi_mod = importlib.import_module("SIMI.SIMIRemote")
            ClientClass = getattr(simi_mod, "SIMIRemoteClient")
            remote_coro = run_core_async(
                ClientClass,
                str(data_path),
                central_host,
                central_port,
                central_proto,
                active_site.SITE_ID,
                parameters,
            )
        else:
            raise ValueError(f"Unsupported algorithm type: {algo}")

        async def _run_with_capture():
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            sys.stdout = _StreamingCapture("stdout")  # type: ignore
            sys.stderr = _StreamingCapture("stderr")  # type: ignore
            try:
                job_entry['messages'].append(f"Starting remote client (algo={algo})...")
                return await remote_coro
            except asyncio.CancelledError:
                job_entry['messages'].append("Remote job cancelled by user.")
                raise
            except Exception as e:
                job_entry['messages'].append(f"Remote job error: {e}")
                raise
            finally:
                try:
                    sys.stdout.flush(); sys.stderr.flush()
                except Exception:
                    pass
                sys.stdout, sys.stderr = orig_stdout, orig_stderr

        task = asyncio.create_task(_run_with_capture())
        job_entry['task'] = task

        def _task_done_cb(t: asyncio.Task, _app=app, _site_id=active_site.SITE_ID, _job_id=job_id):
            try:
                site_jobs_attr_inner = f'running_jobs_{_site_id}'
                if not hasattr(_app.state, site_jobs_attr_inner):
                    return
                jobs_dict = getattr(_app.state, site_jobs_attr_inner)
                je = jobs_dict.get(_job_id)
                if not je:
                    return
                je['end_time'] = datetime.now()
                if t.cancelled():
                    je['status'] = 'Cancelled'
                    je['messages'].append('Task was cancelled')
                else:
                    exc = t.exception()
                    if exc:
                        je['status'] = 'Error'
                        je['messages'].append(f"Task error: {exc}")
                        je['traceback'] = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    else:
                        je['status'] = 'Completed'
                        je['messages'].append('Task completed successfully')
                je['completed'] = je['status'] in ('Cancelled', 'Completed') or je['status'] == 'Error'
            except Exception as cb_e:
                print(f"Error in task done callback for job {_job_id}: {cb_e}")
        task.add_done_callback(_task_done_cb)

        # Log & aggregate job dictionaries
        print(f"🔄 Job {job_id} registered in request.app.state for site {active_site.SITE_ID}")
        print(f"📊 Jobs for this site: {list(site_jobs.keys())}")
        all_running_jobs = []
        for site in config.settings.sites:
            check_attr = f'running_jobs_{site.SITE_ID}'
            if hasattr(request.app.state, check_attr):
                jobs_dict = getattr(request.app.state, check_attr)
                if jobs_dict:
                    all_running_jobs.append(f"{check_attr}({len(jobs_dict)} jobs)")
        print(f"🏘️ All active job dictionaries: {all_running_jobs}")

        if is_ajax:
            return {"success": True, "job_id": job_id}
        else:
            return RedirectResponse(url="/jobs", status_code=303)
    except Exception as e:
        print(f"Error starting job: {e}")
        if is_ajax:
            return {"success": False, "error": f"Error starting job: {str(e)}"}
        else:
            return templates.TemplateResponse("error.html", {"request": request, "error": f"Error starting job: {str(e)}"})

@app.post("/cancel_job")
async def cancel_job(request: Request, job_id: int = Form(...), site_id: Optional[str] = Form(None), site_index: Optional[int] = Form(0)):
    """Cancel a running job by job_id for a specific site.
    Resolution rules:
      1. If site_id provided use it.
      2. Else derive from site_index active site config.
    Returns JSON with cancellation status (for AJAX) or simple dict.
    """
    # Derive active site if site_id not provided
    resolved_site_id = site_id
    if not resolved_site_id:
        if 0 <= (site_index or 0) < len(config.settings.sites):
            resolved_site_id = config.settings.sites[site_index].SITE_ID
    if not resolved_site_id:
        return {"success": False, "error": "No site_id provided and unable to resolve from site_index"}

    site_jobs_attr = f'running_jobs_{resolved_site_id}'
    if not hasattr(request.app.state, site_jobs_attr):
        return {"success": False, "error": f"No jobs found for site {resolved_site_id}"}
    jobs_dict = getattr(request.app.state, site_jobs_attr)
    job_entry = jobs_dict.get(job_id)
    if not job_entry:
        return {"success": False, "error": f"Job {job_id} not found for site {resolved_site_id}"}
    if job_entry.get('completed'):
        return {"success": True, "message": f"Job {job_id} already {job_entry.get('status')}"}
    task: asyncio.Task | None = job_entry.get('task')
    if task is None:
        return {"success": False, "error": "No task handle available to cancel"}
    # Set status to Cancelling prior to actual cancellation
    job_entry['status'] = 'Cancelling'
    job_entry['messages'].append('Cancellation requested')
    try:
        cancelled = task.cancel()
        # Optionally give event loop a chance to process cancellation
        await asyncio.sleep(0)  # yield control
        return {"success": True, "job_id": job_id, "site_id": resolved_site_id, "cancelled": cancelled}
    except Exception as e:
        return {"success": False, "error": f"Error attempting cancellation: {e}"}

@app.get("/job_status")
async def job_status(job_id: int, site_id: str):
    """Retrieve current status for a job (polling helper)."""
    site_jobs_attr = f'running_jobs_{site_id}'
    if not hasattr(app.state, site_jobs_attr):
        return {"success": False, "error": f"No jobs tracked for site {site_id}"}
    jobs_dict = getattr(app.state, site_jobs_attr)
    job_entry = jobs_dict.get(job_id)
    if not job_entry:
        return {"success": False, "error": f"Job {job_id} not found for site {site_id}"}
    # Serialize datetime objects
    def _ser_dt(v):
        return v.isoformat() if isinstance(v, datetime) else v
    serialized = {k: _ser_dt(v) for k, v in job_entry.items() if k != 'task'}
    return {"success": True, "job": serialized}
