from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Integer, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models import Classroom
    from app.models import Session
class SchedulerLog(Base):
    __tablename__ = "scheduler_logs"
    __table_args__ = {"schema": "attendance"}
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("attendance.sessions.id"))
    scan_count: Mapped[int] = mapped_column(Integer)
    scan_session_count: Mapped[int] = mapped_column(Integer)

    session: Mapped["Session"] = relationship("Session", back_populates="scheduler_log")

