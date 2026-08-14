from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    BigInteger,
    Integer,
    ForeignKey,
    Float,
    DateTime,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.student import Student
    from app.models.session import Session


class AttendanceLogs(Base):
    __tablename__ = "attendance_logs"

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "student_id",
            "scan_number",
            name="uq_attendance_log_session_student_scan",
        ),
        {"schema": "attendance"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[int] = mapped_column(
        ForeignKey("attendance.sessions.id"),
        nullable=False,
    )

    student_id: Mapped[int] = mapped_column(
        ForeignKey("attendance.students.id"),
        nullable=False,
    )

    scan_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    detection_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    session: Mapped["Session"] = relationship(
        "Session",
        back_populates="attendance_logs",
    )

    student: Mapped["Student"] = relationship(
        "Student",
        back_populates="attendance_logs",
    )

# from typing import TYPE_CHECKING, Optional
# from sqlalchemy import BigInteger, Integer, ForeignKey, Float
# from sqlalchemy.orm import Mapped, mapped_column, relationship
# from app.database import Base
# from typing import Optional
# if TYPE_CHECKING:
#     from app.models.student import Student
#     from app.models.session import Session

    
# class AttendanceLogs(Base):
#     __tablename__ = "attendance_logs"
#     __table_args__ = {"schema": "attendance"}

#     id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
#     session_id: Mapped[int] = mapped_column(ForeignKey("attendance.sessions.id"))
#     student_id: Mapped[int] = mapped_column(ForeignKey("attendance.students.id"))

#     confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
#     detection_count: Mapped[int] = mapped_column(Integer, default=1)

#     session: Mapped["Session"] = relationship("Session",back_populates="attendance_logs")
#     student: Mapped["Student"] = relationship("Student",back_populates="attendance_logs")


