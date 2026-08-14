from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from app.enums import AttendanceStatus


class AttendanceRecordBase(BaseModel):
    session_id: int = Field(...,ge=1,description="the session id")
    student_id: int = Field(...,ge=1,description="the student id")
    status: AttendanceStatus = Field(default=AttendanceStatus.ABSENT,description="the status of attendance record (present or absent or late)")
    confidence_score:float = Field(default=0.90,description="the confidence score of attendance record")
    manually_modified: bool = Field(default=False,description="the manually modified status of attendance record")
    notes: Optional[str] = Field(default=None,description="the notes of attendance record")

class AttendanceRecordCreate(AttendanceRecordBase):
    pass
class AttendanceRecordUpdate(BaseModel):
    session_id: Optional[int] = Field(default=None,ge=1,description="the session id")
    student_id: Optional[int] = Field(default=None,ge=1,description="the student id")
    status: Optional[AttendanceStatus] = Field(default=None,description="the status of attendance record (present or absent or late)")
    confidence_score:Optional[float] = Field(default=None,description="the confidence score of attendance record")
    manually_modified: Optional[bool] = Field(default=None,description="the manually modified status of attendance record")
    notes: Optional[str] = Field(default=None,description="the notes of attendance record")

class AttendanceRecordResponse(AttendanceRecordBase):
    id:int
    created_at:datetime
    updated_at:datetime
    model_config = ConfigDict(from_attributes=True)


class ManualAttendanceUpdate(BaseModel):
    status: AttendanceStatus
    notes: Optional[str] = None