from datetime import date, time, datetime
from typing import List, TYPE_CHECKING, Optional
from sqlalchemy import BigInteger, ForeignKey, Date, Time, DateTime, func, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.classroom import Classroom
    from app.models.attendance_logs import AttendanceLogs
    from app.models.attendance_record import AttendanceRecord
    from app.models.scheduler_logs import SchedulerLog
    from app.models.videos import Video

class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = {"schema": "attendance"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    classroom_id: Mapped[int] = mapped_column(ForeignKey("attendance.classrooms.id"))
    name: Mapped[str] = mapped_column(String,nullable=False)

    date: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())

    classroom: Mapped["Classroom"] = relationship("Classroom",back_populates="sessions")
    attendance_logs: Mapped[List["AttendanceLogs"]] = relationship("AttendanceLogs",back_populates="session")
    scheduler_log:Mapped[List["SchedulerLog"]] = relationship("SchedulerLog",back_populates="session")
    attendance_records: Mapped[List["AttendanceRecord"]] = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")
    #video: Mapped["Video"] = relationship("Video", back_populates="session")
#محمد8
    videos: Mapped[List["Video"]] = relationship(
    "Video",
    back_populates="session",
)
#محمد8