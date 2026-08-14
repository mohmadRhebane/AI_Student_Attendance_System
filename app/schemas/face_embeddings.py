from datetime import datetime, date
from typing import Optional

from pydantic import BaseModel, Field,ConfigDict

class FaceEmbeddingBase(BaseModel):
    student_id:int = Field(...,gt=0,description="the student id.")
    embedding:list[float] = Field(...,max_length=512,min_length=512,description="the face embedding.")

class FaceEmbeddingCreate(FaceEmbeddingBase):
    pass
class FaceEmbeddingUpdate(BaseModel):
    student_id: Optional[int] = Field(None, gt=0, description="the student id.")
    embedding: Optional[list[float]] = Field(None, max_length=512, min_length=512, description="the face embedding.")

class FaceEmbeddingResponse(FaceEmbeddingBase):
    id:int
    created_at:datetime
    updated_at:datetime
    model_config = ConfigDict(from_attributes=True)
