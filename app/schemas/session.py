from datetime import datetime, date, time
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

class SessionBase(BaseModel):
    classroom_id: int = Field(..., gt=0, description="the id of the classroom.")
    name: str = Field(..., max_length=50, min_length=2, description="the name of the session")
    date: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] = Field(..., description="the date of the session like (MON, TUE, WED, ...)")
    start_time: time = Field(..., description="the start time of the session.")
    end_time: time = Field(..., description="the end time of the session.")

    @model_validator(mode='after')
    def validate_times(self) -> 'SessionBase':
        if self.start_time >= self.end_time:
            raise ValueError("end_time must be strictly after start_time.")
        return self


class SessionCreate(SessionBase):
    pass

class SessionUpdate(BaseModel):
    classroom_id: Optional[int] = Field(None, gt=0, description="the id of the classroom.")
    name: Optional[str] = Field(None, description="the name of the session")
    date: Optional[Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]] = Field(None, description="the date of the session like (MON, TUE, WED, ...)")
    start_time: Optional[time] = Field(None, description="the start time of the session.")
    end_time: Optional[time] = Field(None, description="the end time of the session.")

    @model_validator(mode='after')
    def validate_times(self) -> 'SessionUpdate':
        if self.start_time is not None and self.end_time is not None:
            if self.start_time >= self.end_time:
                raise ValueError("end_time must be strictly after start_time.")
        return self

class SessionResponse(SessionBase):
    id: int
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)