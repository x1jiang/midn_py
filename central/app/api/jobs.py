from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import json
import os
from jsonschema import validate, ValidationError

from .. import schemas, services
from ..db import get_db
from ..core.security import require_admin

router = APIRouter()

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'algorithms')


def load_params_schema(algorithm: str):
    algo_dir = os.path.join(SCHEMA_DIR, algorithm)
    schema_path = os.path.join(algo_dir, 'params.json')
    if not os.path.exists(schema_path):
        raise HTTPException(status_code=400, detail=f"Parameter schema not found for algorithm: {algorithm}")
    with open(schema_path, 'r') as f:
        return json.load(f)


@router.post("/", response_model=schemas.Job)
def create_job(job: schemas.JobCreate, request: Request, db: Session = Depends(get_db)):
    require_admin(request)
    algorithm = (job.algorithm or '').upper()
    if algorithm not in {"SIMI"}:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {job.algorithm}")

    schema = load_params_schema(algorithm)
    try:
        validate(instance=job.parameters, schema=schema)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Parameter validation failed: {e.message}")

    return services.job_service.create_job(db=db, job=job)


@router.get("/", response_model=list[schemas.Job])
def read_jobs(skip: int = 0, limit: int = 100, request: Request = None, db: Session = Depends(get_db)):
    require_admin(request)
    jobs = services.job_service.get_jobs(db, skip=skip, limit=limit)
    return jobs


@router.get("/{job_id}", response_model=schemas.Job)
def read_job(job_id: int, request: Request, db: Session = Depends(get_db)):
    require_admin(request)
    db_job = services.job_service.get_job(db, job_id=job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return db_job


@router.get("/{job_id}/download")
def download_imputed_dataset(job_id: int, request: Request, db: Session = Depends(get_db)):
    """Download the imputed dataset for a job."""
    require_admin(request)
    db_job = services.job_service.get_job(db, job_id=job_id)
    
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not db_job.imputed_dataset_path:
        raise HTTPException(status_code=404, detail="No imputed dataset available for this job")
    
    # Construct the full path to the imputed dataset
    file_path = os.path.join("central", "app", db_job.imputed_dataset_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Imputed dataset file not found")
    
    # Extract filename from path
    filename = os.path.basename(file_path)
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/zip'
    )
