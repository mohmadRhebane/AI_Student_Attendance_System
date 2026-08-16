from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Date,
    ForeignKey,
    Integer,
    UniqueConstraint,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.database import Base


if TYPE_CHECKING:
    from app.models.session import Session


class SchedulerLog(Base):

    __tablename__ = "scheduler_logs"

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "attendance_date",
            name="uq_scheduler_log_session_date",
        ),
        {
            "schema": "attendance",
        },
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

    attendance_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    scan_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    scan_session_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    session: Mapped["Session"] = relationship(
        "Session",
        back_populates="scheduler_log",
    )
