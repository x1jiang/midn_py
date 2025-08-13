from sqlalchemy.orm import Session

from .. import models, schemas


def get_job(db: Session, job_id: int):
    return db.query(models.Job).filter(models.Job.id == job_id).first()


def get_jobs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Job).offset(skip).limit(limit).all()


def create_job(db: Session, job: schemas.JobCreate, owner_id: int | None = None):
    data = job.dict()
    if owner_id is not None:
        data["owner_id"] = owner_id
    db_job = models.Job(**data)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def update_job(db: Session, job_id: int, *, name: str | None = None, description: str | None = None,
               participants: list[str] | None = None, imputation_trials: int | None = None):
    db_job = get_job(db, job_id)
    if not db_job:
        return None
    if name is not None:
        db_job.name = name
    if description is not None:
        db_job.description = description
    if participants is not None:
        db_job.participants = participants
    if imputation_trials is not None:
        db_job.imputation_trials = imputation_trials
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def delete_job(db: Session, job_id: int):
    db_job = get_job(db, job_id)
    if not db_job:
        return False
    db.delete(db_job)
    db.commit()
    return True
