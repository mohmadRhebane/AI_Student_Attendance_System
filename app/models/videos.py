from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import BigInteger, ForeignKey, Float, Boolean, Text, Enum, DateTime, func
from app.database import Base

if TYPE_CHECKING:
    from app.models.session import Session

class Video(Base):
    __tablename__ = "videos"
    __table_args__ = {"schema": "attendance"}
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
  #  session_id: Mapped[int] = mapped_column(ForeignKey("sessions.session_id"))
#محمد6
    session_id: Mapped[int] = mapped_column(ForeignKey("attendance.sessions.id"), nullable=False, unique=True)
#محمد6
    path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())
    #session: Mapped["Session"] = relationship("Session", back_populates="videos")
#محمد7
    session: Mapped["Session"] = relationship("Session", back_populates="video")
#محمد7