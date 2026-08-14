from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field,ConfigDict

class StudentBase(BaseModel):
    first_name: str = Field(...,max_length=50,min_length=2,pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",description="the first name for student, is real name.")
    last_name: str = Field(..., max_length=50, min_length=2, pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",description="the last name for student, is nickname.")
    father_name: str = Field(..., max_length=50, min_length=2, pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",description="the father name for the student, just first name of father.")
    classroom_id: int = Field(...,gt=0,description="the id of the classroom.")
    grade: str = Field(...,min_length=1,max_length=20, pattern=r"^[\u0621-\u064Aa-zA-Z0-9\s'-]+$",description="the grade of the classroom.")
    section: str = Field(...,min_length=1,max_length=20, pattern=r"^[\u0621-\u064Aa-zA-Z0-9\s'-]+$",description="the section of the classroom.")
class StudentCreate(StudentBase):
    face_embeddings: Optional[List[List[float]]] = None     
class StudentUpdate(BaseModel):
    first_name: Optional[str] = Field(None, max_length=50, min_length=2, pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",
                            description="the first name for student, is real name.")
    last_name: Optional[str] = Field(None, max_length=50, min_length=2, pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",
                           description="the last name for student, is nickname.")
    father_name: Optional[str] = Field(None, max_length=50, min_length=2, pattern=r"^[\u0621-\u064Aa-zA-Z\s'-]+$",
                             description="the father name for the student, just first name of father.")
    classroom_id: Optional[int] = Field(None,gt=0, description="the id of the classroom.")
    grade: Optional[str] = Field(None, min_length=1, max_length=10, pattern=r"^[\u0621-\u064Aa-zA-Z0-9\s'-]+$", description="the grade of the classroom.")
    section: Optional[str] = Field(None, min_length=1, max_length=10, pattern=r"^[\u0621-\u064Aa-zA-Z0-9\s'-]+$", description="the section of the classroom.")
    face_embeddings: Optional[List[List[float]]] = None
    is_active: Optional[bool] = None


class StudentResponse(StudentBase):
    id: int
    #محمد2
    is_active: bool
    #محمد2
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

