from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from ..db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)  # site name must be unique
    email = Column(String, index=True)  # duplicates allowed
    hashed_password = Column(String)
    institution = Column(String)
    is_active = Column(Boolean, default=True)
    is_approved = Column(Boolean, default=False)
    site_id = Column(String, unique=True, index=True)
    jwt_token = Column(String, nullable=True)  # issued JWT for this site
    jwt_expires_at = Column(DateTime, nullable=True)  # expiry timestamp (UTC)

    jobs = relationship("Job", back_populates="owner")
