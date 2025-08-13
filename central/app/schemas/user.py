from pydantic import BaseModel, EmailStr, ConfigDict

class UserBase(BaseModel):
    username: str
    email: EmailStr
    institution: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    is_approved: bool
    site_id: str

    # Pydantic v2 config
    model_config = ConfigDict(from_attributes=True)
