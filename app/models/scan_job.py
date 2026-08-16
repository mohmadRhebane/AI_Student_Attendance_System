from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    Date,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


if TYPE_CHECKING:
    from app.models.session import Session


class ScanJob(Base):
    __tablename__ = "scan_jobs"

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "attendance_date",
            "scan_number",
            name="uq_scan_job_session_date_scan",
),
        Index(
            "ix_scan_jobs_status_priority_eligible",
            "status",
            "priority",
            "eligible_at",
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

    scan_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    segment_start_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    segment_duration_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        default=100,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="WAITING",
        nullable=False,
    )

    attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    eligible_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    queued_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    session: Mapped["Session"] = relationship(
        "Session",
    )