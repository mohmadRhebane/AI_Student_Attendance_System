from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class AttendanceLogBase(BaseModel):
    session_id: int = Field(...,gt=0,description="the session id.")
    student_id: int = Field(...,gt=0,description="the student id.")
    confidence_score: float = Field(default=0.90,gt=0.5,le=1.0,description="the confidence score.")
    detection_count: int = Field(...,ge=1,le=10,description="the detection count.")

class AttendanceLogCreate(AttendanceLogBase):
    pass
class AttendanceLogUpdate(BaseModel):
    session_id: Optional[int] = Field(default=None,gt=0,description="the session id.")
    student_id: Optional[int] = Field(default=None,gt=0,description="the student id.")
    confidence_score: Optional[float] = Field(default=None,gt=0.5,le=1.0,description="the confidence score.")
    detection_count: Optional[int] = Field(default=None,ge=1,le=10,description="the detection count.")

class AttendanceLogResponse(AttendanceLogBase):
    id:int
    model_config = ConfigDict(from_attributes=True)