from sqlalchemy.orm import Session

from .. import models, schemas

# No-op placeholders to avoid import errors

def get_algorithms(db: Session, skip: int = 0, limit: int = 100):
    return []
