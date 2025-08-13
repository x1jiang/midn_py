from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional

class JobBase(BaseModel):
    name: str
    description: str
    algorithm: str  # algorithm name, e.g., "SIMI"
    parameters: Dict  # validated against algorithms/<algorithm>/params.json
    participants: List[str]
    missing_spec: Optional[Dict] = None
    iteration_before_first_imputation: Optional[int] = None
    iteration_between_imputations: Optional[int] = None
    imputation_trials: int = 10  # number of imputation trials (all algorithms)

class JobCreate(JobBase):
    pass

class Job(JobBase):
    id: int
    status: str
    owner_id: int

    model_config = ConfigDict(from_attributes=True)
