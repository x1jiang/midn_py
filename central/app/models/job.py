from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import relationship

from ..db.database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    status = Column(String, default="created")

    algorithm = Column(String, index=True)  # e.g., "SIMI"
    parameters = Column(JSON)

    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="jobs")

    participants = Column(JSON)  # List of site_ids

    missing_spec = Column(JSON)

    iteration_before_first_imputation = Column(Integer)
    iteration_between_imputations = Column(Integer)

    imputation_trials = Column(Integer, default=10)

    imputed_dataset_path = Column(String)
