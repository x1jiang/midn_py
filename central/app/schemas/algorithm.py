from pydantic import BaseModel, ConfigDict

class Algorithm(BaseModel):
    id: int | None = None
    name: str | None = None
    description: str | None = None
    version: str | None = None
    schema_str: str | None = None

    model_config = ConfigDict(from_attributes=True)
