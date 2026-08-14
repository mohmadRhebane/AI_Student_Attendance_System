from datetime import datetime
from typing import List, TYPE_CHECKING

from sqlalchemy import BigInteger, String, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.supervisor import Supervisor
    from app.models.student import Student
    from app.models.session import Session
    from app.models.scheduler_logs import SchedulerLog
class Classroom(Base):
    __tablename__ = "classrooms"
    __table_args__ = {"schema": "attendance"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    supervisor_id: Mapped[int] = mapped_column(ForeignKey("attendance.supervisors.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    supervisor: Mapped["Supervisor"] = relationship("Supervisor",back_populates="classrooms")
    students: Mapped[List["Student"]] = relationship("Student",back_populates="classroom")
    sessions: Mapped[List["Session"]] = relationship("Session",back_populates="classroom")

