from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import pandas as pd
import httpx
from datetime import datetime, timezone
from jose import jwt
from typing import Optional, Dict, Any

from . import config
from .services.algorithm_factory import AlgorithmClientFactory
from .custom_templates import templates
from . import services  # Ensure algorithm clients are registered
import algorithms  # Ensure algorithm implementations are registered
from .job_status import JobStatusCallback
from .routes import jobs as jobs_routes

app = FastAPI()

app.mount("/static", StaticFiles(directory="remote/app/static"), name="static")

# Initialize the running_jobs state
app.state.running_jobs = {}

# Include job status routes
app.include_router(jobs_routes.router)

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

async def get_site_info(site_id):
    """Try to get the site information from the central server"""
    try:
        if site_id and site_id != "my_site_id":
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{config.settings.HTTP_URL}/api/remote/info",
                                    params={"site_id": site_id},
                                    headers={"Authorization": f"Bearer {config.settings.TOKEN}"})
                if r.status_code == 200:
                    return r.json()
    except Exception as e:
        print(f"Error fetching site info: {e}")
    
    return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, refresh: bool = False, site_index: int = 0):
    jobs = []
    site_info = None
    message = None
    
    # Set the site based on the index parameter
    if 0 <= site_index < len(config.settings.sites):
        # Temporarily change the sites order to make the selected site first
        if site_index > 0:
            selected_site = config.settings.sites.pop(site_index)
            config.settings.sites.insert(0, selected_site)
    
    # If site_id is configured and config file exists, fetch available jobs
    if config.settings.SITE_ID and config.settings.SITE_ID != "my_site_id" and config.settings.TOKEN and config.settings.TOKEN != "my_jwt_token":
        try:
            # Use HTTP URL for API calls
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{config.settings.HTTP_URL}/api/remote/jobs",
                                    params={"site_id": config.settings.SITE_ID},
                                    headers={"Authorization": f"Bearer {config.settings.TOKEN}"})
                if r.status_code == 200:
                    jobs = r.json()
                    print(f"Successfully fetched {len(jobs)} jobs for site {config.settings.SITE_ID}")
                    if refresh:
                        message = f"Data refreshed successfully. Found {len(jobs)} jobs assigned to this site."
                    
                    # Only try to get site information if jobs API call succeeds (token is valid)
                    site_info = await get_site_info(config.settings.SITE_ID)
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
    
    current_site_name = config.settings.current_site.name if config.settings.current_site else "Default Site"
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "settings": config.settings,
        "sites": config.settings.sites,
        "active_site": current_site_name,
        "jobs": jobs, 
        "site_name": site_name or "Site Not Authenticated",
        "institution": institution or "",
        "token_expiry": _token_expiry_str(config.settings.TOKEN),
        "message": message
    })

@app.get("/settings", response_class=HTMLResponse)
async def settings_get(request: Request):
    current_site_name = config.settings.current_site.name if config.settings.current_site else "Default Site"
    return templates.TemplateResponse("settings.html", {
        "request": request, 
        "settings": config.settings, 
        "sites": config.settings.sites,
        "active_site": current_site_name,
        "token_expiry": _token_expiry_str(config.settings.TOKEN)
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
async def get_jobs(site_id: Optional[str] = None):
    """Endpoint that returns job data as JSON for AJAX requests"""
    jobs = []
    
    # Use the provided site_id or fallback to the current site
    site_id_to_use = site_id or config.settings.SITE_ID
    
    if site_id_to_use and site_id_to_use != "my_site_id" and config.settings.TOKEN and config.settings.TOKEN != "my_jwt_token":
        try:
            # Use HTTP URL for API calls
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{config.settings.HTTP_URL}/api/remote/jobs",
                                    params={"site_id": site_id_to_use},
                                    headers={"Authorization": f"Bearer {config.settings.TOKEN}"})
                if r.status_code == 200:
                    jobs = r.json()
                    print(f"Successfully fetched {len(jobs)} jobs for site {site_id_to_use} via API")
        except Exception as e:
            print(f"Error fetching jobs via API: {e}")
    
    return jobs

@app.get("/jobs", response_class=HTMLResponse)
async def list_jobs(request: Request, refresh: bool = False, site_index: int = 0):
    jobs = []
    site_info = None
    message = None
    
    # Set the site based on the index parameter
    if 0 <= site_index < len(config.settings.sites):
        # Temporarily change the sites order to make the selected site first
        if site_index > 0:
            selected_site = config.settings.sites.pop(site_index)
            config.settings.sites.insert(0, selected_site)
    
    try:
        central_url = f"{config.settings.CENTRAL_URL.replace('ws://','http://').replace('wss://','https://')}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{central_url}/api/remote/jobs",
                                params={"site_id": config.settings.SITE_ID},
                                headers={"Authorization": f"Bearer {config.settings.TOKEN}"})
            if r.status_code == 200:
                jobs = r.json()
                if refresh:
                    message = f"Data refreshed successfully. Found {len(jobs)} assigned jobs."
                
                # Only try to get site information if jobs API call succeeds (token is valid)
                site_info = await get_site_info(config.settings.SITE_ID)
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
    
    current_site_name = config.settings.current_site.name if config.settings.current_site else "Default Site"
    
    return templates.TemplateResponse("jobs.html", {
        "request": request, 
        "jobs": jobs, 
        "settings": config.settings,
        "sites": config.settings.sites,
        "active_site": current_site_name, 
        "site_name": site_name or "Site Not Authenticated",
        "institution": institution or "",
        "token_expiry": _token_expiry_str(config.settings.TOKEN),
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
    print(f"Starting job: ID={job_id}, Type={job_type}, Missing var={mvar}, Site index={site_index}")
    
    # Set the site based on the index parameter to ensure correct site ID and token are used
    if 0 <= site_index < len(config.settings.sites):
        # Temporarily change the sites order to make the selected site first
        if site_index > 0:
            selected_site = config.settings.sites.pop(site_index)
            config.settings.sites.insert(0, selected_site)
            print(f"Using site configuration: {config.settings.current_site.name}, ID={config.settings.SITE_ID}")
            
    # Verify that current site ID matches the expected site ID
    if current_site_id and current_site_id != config.settings.SITE_ID:
        print(f"Warning: Current site ID from form ({current_site_id}) doesn't match active site ID ({config.settings.SITE_ID})")
    
    # Check for running jobs - if any job is running, block starting a new one
    if has_running_jobs(app, config.settings.SITE_ID):
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
            r = await client.get(f"{config.settings.CENTRAL_URL.replace('ws://','http://').replace('wss://','https://')}/api/remote/jobs",
                                params={"site_id": config.settings.SITE_ID},
                                headers={"Authorization": f"Bearer {config.settings.TOKEN}"})
            if r.status_code == 200:
                jobs = r.json()
                for job in jobs:
                    if job["id"] == job_id:
                        job_details = job
                        print(f"Found job details: Algorithm={job['algorithm']}, Parameters={job['parameters']}")
                        break
    except Exception as e:
        print(f"Error fetching job details: {e}")
    
    # Determine the appropriate missing variable index based on the job type
    try:
        # Get job parameters if available
        is_binary = False
        if job_details and "parameters" in job_details:
            # For SIMI jobs, use the target column index from parameters
            if job_details["algorithm"] == "SIMI" and "target_column_index" in job_details["parameters"]:
                mvar = job_details["parameters"]["target_column_index"]
                # Check if the job is for binary imputation
                if "is_binary" in job_details["parameters"]:
                    is_binary = job_details["parameters"]["is_binary"]
                    print(f"Using SIMI with binary flag: {is_binary}")
                print(f"Using SIMI target column index from job parameters: {mvar}")
            # For SIMICE jobs, use the first target column index from parameters
            elif job_details["algorithm"] == "SIMICE" and "target_column_indexes" in job_details["parameters"]:
                target_columns = job_details["parameters"]["target_column_indexes"]
                if target_columns and len(target_columns) > 0:
                    mvar = target_columns[0]
                    print(f"Using SIMICE first target column from job parameters: {mvar}")
        
        # Convert to integer and adjust to 0-based indexing
        mvar_index = int(mvar) - 1
        print(f"Using missing variable index: {mvar} (1-based) -> {mvar_index} (0-based)")
        
        # Validate if the column index is valid for the dataset
        if mvar_index < 0 or mvar_index >= df.shape[1]:
            print(f"Invalid column index: {mvar_index} (0-based) for dataset with {df.shape[1]} columns")
            return templates.TemplateResponse("error.html", {
                "request": request, 
                "error": f"Invalid column index: {mvar}. Dataset has {df.shape[1]} columns (1-{df.shape[1]})."
            })
            
    except ValueError as e:
        print(f"Invalid mvar value: {mvar}, Error: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request, 
            "error": "Invalid value for Missing Variable Index. Please check your data file format."
        })
    
    # Create task based on job type
    try:
        if job_type == "SIMICE" or (job_details and job_details.get("algorithm") == "SIMICE"):
            # For SIMICE algorithm
            extra_params = {}
            if iteration_before_first_imputation is not None:
                extra_params["iteration_before_first_imputation"] = int(iteration_before_first_imputation)
            if iteration_between_imputations is not None:
                extra_params["iteration_between_imputations"] = int(iteration_between_imputations)
            
            # Debug info about current site configuration
            print(f"Starting SIMICE job with site ID: {config.settings.SITE_ID}, site name: {config.settings.current_site.name}")
            print(f"Extra params: {extra_params}")
            
            # Create a status callback with site ID
            status_callback = JobStatusCallback(app, job_id, config.settings.SITE_ID)
            
            # Start SIMICE job with status updates (using SIMI for now)
            client = AlgorithmClientFactory.create_client("SIMI")
            asyncio.create_task(client.run_algorithm(
                data=df.values,
                target_column=mvar_index,
                job_id=job_id,
                site_id=config.settings.SITE_ID,
                central_url=config.settings.CENTRAL_URL,
                token=config.settings.TOKEN,
                extra_params=extra_params,
                status_callback=status_callback
            ))
        else:
            # Default to SIMI algorithm
            print(f"Starting SIMI job (is_binary={is_binary})")
            print(f"Using site ID: {config.settings.SITE_ID}, site name: {config.settings.current_site.name}")
            
            # Create a status callback with site ID
            status_callback = JobStatusCallback(app, job_id, config.settings.SITE_ID)
            
            # Start SIMI job with status updates
            client = AlgorithmClientFactory.create_client("SIMI")
            asyncio.create_task(client.run_algorithm(
                data=df.values,
                target_column=mvar_index,
                job_id=job_id,
                site_id=config.settings.SITE_ID,
                central_url=config.settings.CENTRAL_URL,
                token=config.settings.TOKEN,
                is_binary=is_binary,  # Pass the binary flag
                status_callback=status_callback
            ))
        
        print("Job started successfully")
        
        # Get or create site-specific job dictionary
        site_jobs_attr = f'running_jobs_{config.settings.SITE_ID}'
        if not hasattr(app.state, site_jobs_attr):
            setattr(app.state, site_jobs_attr, {})
        
        # Track the running job in the site-specific state
        site_jobs = getattr(app.state, site_jobs_attr)
        site_jobs[job_id] = {
            'status': 'Running',
            'messages': ['Job started successfully'],
            'start_time': datetime.now(),
            'completed': False,
            'site_id': config.settings.SITE_ID  # Store site ID with the job
        }
        
        # Log for debugging
        print(f"Job {job_id} registered in app state for site {config.settings.SITE_ID}")
        print(f"Jobs for this site: {list(site_jobs.keys())}")
        
        if is_ajax:
            return {"success": True, "job_id": job_id}
        else:
            return RedirectResponse(url="/jobs", status_code=303)
            
    except Exception as e:
        print(f"Error starting job: {e}")
        if is_ajax:
            return {"success": False, "error": f"Error starting job: {str(e)}"}
        else:
            return templates.TemplateResponse("error.html", {
                "request": request, 
                "error": f"Error starting job: {str(e)}"
            })
