from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict

from ..db import get_db
from ..core.security import get_current_user
from .. import models

router = APIRouter()

@router.get("/info", response_model=Dict)
async def get_site_info(site_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get information about a remote site"""
    # user is the JWT subject; must match site_id
    if user != site_id:
        raise HTTPException(status_code=403, detail="Forbidden")
        
    # Find the user record for this site
    site_user = db.query(models.User).filter(models.User.site_id == site_id).first()
    if not site_user:
        raise HTTPException(status_code=404, detail="Site not found")
        
    return {
        "id": site_user.id,
        "name": site_user.username or f"Remote Site {site_id}",
        "institution": site_user.institution or "",
        "email": site_user.email,
        "site_id": site_id,
        "is_approved": site_user.is_approved
    }

@router.get("/jobs", response_model=list[dict])
async def list_site_jobs(site_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    # user is the JWT subject; must match site_id
    if user != site_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    jobs = db.query(models.Job).all()
    result = []
    for j in jobs:
        try:
            parts = j.participants or []
            if site_id in parts:
                result.append({
                    "id": j.id,
                    "name": j.name,
                    "algorithm": j.algorithm,
                    "parameters": j.parameters,
                    "status": j.status,
                    "missing_spec": j.missing_spec,
                    "iteration_before_first_imputation": j.iteration_before_first_imputation,
                    "iteration_between_imputations": j.iteration_between_imputations,
                    "imputation_trials": j.imputation_trials,
                })
        except Exception as e:
            print(f"Error serializing job {j.id}: {str(e)}")
            continue
    return result
