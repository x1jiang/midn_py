from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from .. import schemas, services
from ..db import get_db
from ..core.security import require_admin

router = APIRouter()

@router.post("/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Enforce unique site name; allow duplicate emails
    existing = services.user_service.get_user_by_username(db, user.username)
    if existing:
        raise HTTPException(status_code=400, detail="Site name already registered")
    return services.user_service.create_user(db=db, user=user)

@router.get("/{user_id}", response_model=schemas.User)
def read_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    require_admin(request)
    db_user = services.user_service.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.post("/{user_id}/approve", response_model=schemas.User)
def approve_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    require_admin(request)
    db_user = services.user_service.approve_user(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user
