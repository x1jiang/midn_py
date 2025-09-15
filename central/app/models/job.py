from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import relationship

from ..db.database import Base

class Job(Base):
    """Job metadata record.

    Canonical algorithm parameters are stored exclusively in `parameters` JSON
    using config-driven schemas (see config/*.json and alg_config loader).
    Legacy iteration/imputation columns and `missing_spec` have been removed
    (migration required if upgrading from older schema versions).
    """
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

    imputed_dataset_path = Column(String)
