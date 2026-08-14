from datetime import datetime
from typing import Optional
import re
from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator

from app.models import Supervisor


class SupervisorBase(BaseModel):
    username: str = Field(...,max_length=50,min_length=2,pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",description="the unique name for supervisor, is job name.")
    full_name: str = Field(...,max_length=50,min_length=2,pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",description="the full name for supervisor.")
    email:EmailStr = Field(...,max_length=50,pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",description="the email address for supervisor, is unique.")

class SupervisorCreate(SupervisorBase):
    password: str = Field(...,max_length=50,min_length=6,description="the password for supervisor.")

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, value: str) :
        pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"

        if not re.match(pattern, value):
            raise ValueError(
                "Password must contain at least one uppercase letter, "
                "one lowercase letter, one number, and one special character."
            )
        return value

class SupervisorUpdate(BaseModel):
    username: Optional[str] = Field(None, max_length=50, min_length=2, pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",
                          description="the unique name for supervisor, is job name.")
    full_name: Optional[str] = Field(None, max_length=50, min_length=2, pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",
                           description="the full name for supervisor.")
    email: Optional[EmailStr] = Field(None, max_length=50, pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
                            description="the email address for supervisor, is unique.")


class SupervisorResponsePublic(BaseModel):
    id: int
    full_name: str
    username: str
    model_config = ConfigDict(from_attributes=True)
class SupervisorResponsePrivate(SupervisorResponsePublic):
    email:EmailStr
    created_at: datetime
    updated_at: datetime

class LoginRequest(BaseModel):
    email_username: str
    password: str


class ResponseWithoutPassword:
    def __init__(self,data:Supervisor):
        self.id = data.id
        self.full_name = data.full_name
        self.username = data.username
        self.email = data.email
        self.created_at = data.created_at
        self.updated_at = data.updated_at
