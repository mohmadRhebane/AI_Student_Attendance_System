from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Float, Boolean, Text, Enum, DateTime, func,UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.enums import AttendanceStatus
from typing import Optional

if TYPE_CHECKING:
    from app.models.supervisor import Supervisor
    from app.models.student import Student
    from app.models.session import Session
class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
   # __table_args__ = {"schema": "attendance"}

#محمد3    
    __table_args__ = (
            UniqueConstraint(
                "session_id",
                "student_id",
                name="uq_attendance_record_session_student",
            ),
            {"schema": "attendance"},
        )
#محمد3
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("attendance.students.id"))
    session_id: Mapped[int] = mapped_column(ForeignKey("attendance.sessions.id"))
    supervisor_id: Mapped[int] = mapped_column(ForeignKey("attendance.supervisors.id"))

    #status: Mapped[str] = mapped_column(Enum(AttendanceStatus, native_enum=True, create_type=False)
     #                                                       , default=AttendanceStatus.ABSENT,
     #                                                       nullable=False)  # Present, Absent, Late
 #محمد4
    status: Mapped[AttendanceStatus] = mapped_column(
        Enum(
            AttendanceStatus,
            name="attendancestatus",
            schema="attendance",
            native_enum=True,
        ),
        default=AttendanceStatus.ABSENT,
        nullable=False,
    )
#محمد4
    
    
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    manually_modified: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())

    student: Mapped["Student"] = relationship("Student",back_populates="attendance_records")
    session: Mapped["Session"] = relationship("Session", back_populates="attendance_records")
    supervisor: Mapped["Supervisor"] = relationship("Supervisor",back_populates="records")

