from datetime import datetime, date
from typing import Optional

from pydantic import BaseModel, Field,ConfigDict

from app.enums import AttendanceMode

class SystemSetting(BaseModel):
    attendance_mode:AttendanceMode = Field(default=AttendanceMode.AUTOMATIC,description="The mode of attendance (automatic or manual).")
    times_per_session:int = Field(...,ge=1,le=5,description="the number of times a session will be created.")
    min_confidence_score:float = Field(...,gt=0.5,le=1.0,description="the minimum confidence score.")
    is_active: bool = Field(default=True, description="Is this setting configuration currently active?")
    scan_duration_seconds: int = Field(
    default=120,
    gt=0,
    description="Duration of each attendance scan segment in seconds.",
        )

class SystemSettingCreate(SystemSetting):
    pass

class SystemSettingUpdate(BaseModel):
    attendance_mode:Optional[AttendanceMode] = Field(default=AttendanceMode.AUTOMATIC,description="The mode of attendance (automatic or manual).")
    times_per_session:Optional[int] = Field(None,ge=1,le=5,description="the number of times a session will be created.")
    min_confidence_score:Optional[float] = Field(None,gt=0.5,le=1.0,description="the minimum confidence score.")
    is_active:Optional[bool] = Field(default=True,description="Is this setting configuration currently active?")
    scan_duration_seconds: Optional[int] = Field(
    default=None,
    gt=0,
    description="Duration of each attendance scan segment in seconds.",
)
class SystemSettingResponse(SystemSetting):
    id:int
    created_at:datetime
    updated_at:datetime
    model_config = ConfigDict(from_attributes=True)
