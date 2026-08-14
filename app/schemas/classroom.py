from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field,ConfigDict

class ClassroomBase(BaseModel):
    name: str = Field(...,max_length=50, min_length=1,description="name of the classroom.")
    supervisor_id: int = Field(None,gt=0,description="supervisor id of the classroom.")

class ClassroomCreate(ClassroomBase):
    pass
class ClassroomUpdate(BaseModel):
    name:Optional[str] = Field(None,max_length=50, min_length=1,description="name of the classroom.")
    supervisor_id: Optional[int] = Field(None,gt=0,description="supervisor id of the classroom.")
class ClassroomResponse(ClassroomBase):
    id:int
    created_at:datetime
    updated_at:datetime
    model_config = ConfigDict(from_attributes=True)
