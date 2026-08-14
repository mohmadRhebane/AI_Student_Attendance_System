from datetime import datetime

from sqlalchemy import Integer, Float, Boolean, Enum, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base
from app.enums import AttendanceMode


class SystemSetting(Base):
    __tablename__ = "system_settings"
    __table_args__ = {"schema": "attendance"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
   # attendance_mode: Mapped[AttendanceMode] = mapped_column(Enum(AttendanceMode, native_enum=True, create_type=False)
    #                                                        , default=AttendanceMode.AUTOMATIC,
     #                                                       nullable=False)
 #محمد5  
    attendance_mode: Mapped[AttendanceMode] = mapped_column(Enum(AttendanceMode,
                                                                  name="attendancemode", schema="attendance",
                                                                    native_enum=True), default=AttendanceMode.AUTOMATIC, nullable=False)
#محمد5
    times_per_session: Mapped[int] = mapped_column(Integer, default=2)
    min_confidence_score: Mapped[float] = mapped_column(Float, default=0.50)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())
    scan_duration_seconds: Mapped[int] = mapped_column(
    Integer,
    default=120,
    server_default="120",
    nullable=False,
)
