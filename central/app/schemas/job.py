from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional

class JobBase(BaseModel):
    name: str
    description: str
    algorithm: str  # algorithm name, e.g., "SIMI"
    parameters: Dict  # validated against algorithms/<algorithm>/params.json
    participants: List[str]
    # legacy metadata removed (missing_spec, iteration*, imputation_trials)

class JobCreate(JobBase):
    pass

class Job(JobBase):
    id: int
    status: str
    owner_id: int

    model_config = ConfigDict(from_attributes=True)
