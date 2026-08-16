from datetime import datetime
from sqlalchemy import BigInteger, String, ForeignKey, DateTime, Boolean, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.classroom import Classroom
    from app.models.supervisor import AttendanceRecord
    from app.models.face_embedding import FaceEmbedding
    from app.models.attendance_logs import AttendanceLogs
class Student(Base):
    __tablename__ = "students"
    __table_args__ = {"schema": "attendance"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    classroom_id: Mapped[int] = mapped_column(ForeignKey("attendance.classrooms.id"))
    first_name: Mapped[str] = mapped_column(String(50), nullable=False)
    last_name: Mapped[str] = mapped_column(String(50), nullable=False)
    father_name: Mapped[str] = mapped_column(String(50), nullable=False)
    grade: Mapped[str] = mapped_column(String(50), nullable=False)
    section: Mapped[str] = mapped_column(String(50), nullable=False)
#محمد1
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
#محمد1
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    classroom: Mapped["Classroom"] = relationship("Classroom",back_populates="students")
    attendance_records: Mapped[List["AttendanceRecord"]] = relationship("AttendanceRecord",back_populates="student")

    face_embeddings: Mapped[List["FaceEmbedding"]] = relationship("FaceEmbedding",back_populates="student",
                                                                  cascade="all, delete-orphan")
    attendance_logs: Mapped[List["AttendanceLogs"]] = relationship("AttendanceLogs",back_populates="student")
