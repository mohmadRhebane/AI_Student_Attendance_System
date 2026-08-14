from datetime import datetime
from typing import List, TYPE_CHECKING
from sqlalchemy import BigInteger, String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.classroom import Classroom
    from app.models.attendance_record import AttendanceRecord
class Supervisor(Base):
    __tablename__ = "supervisors"
    __table_args__ = {"schema": "attendance"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    classrooms: Mapped[List["Classroom"]] = relationship("Classroom",back_populates="supervisor")
    records: Mapped[List["AttendanceRecord"]] = relationship("AttendanceRecord",back_populates="supervisor")

