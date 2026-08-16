"""add attendance date to attendance runtime tables

Revision ID: 4c9f8e21a601
Revises: 268eec145533
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "4c9f8e21a601"
down_revision: Union[str, Sequence[str], None] = "268eec145533"
branch_labels = None
depends_on = None


def _drop_video_session_unique():
    """
    اسم Unique Constraint على videos.session_id
    قد يختلف حسب PostgreSQL/Alembic.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    constraints = inspector.get_unique_constraints(
        "videos",
        schema="attendance",
    )

    for constraint in constraints:
        columns = constraint.get(
            "column_names"
        ) or []

        if columns == ["session_id"]:
            name = constraint.get("name")

            if name:
                op.drop_constraint(
                    name,
                    "videos",
                    schema="attendance",
                    type_="unique",
                )


def upgrade() -> None:

    # =========================================================
    # 1. Add nullable first
    # =========================================================

    op.add_column(
        "attendance_logs",
        sa.Column(
            "attendance_date",
            sa.Date(),
            nullable=True,
        ),
        schema="attendance",
    )

    op.add_column(
        "attendance_records",
        sa.Column(
            "attendance_date",
            sa.Date(),
            nullable=True,
        ),
        schema="attendance",
    )

    op.add_column(
        "scheduler_logs",
        sa.Column(
            "attendance_date",
            sa.Date(),
            nullable=True,
        ),
        schema="attendance",
    )

    op.add_column(
        "scan_jobs",
        sa.Column(
            "attendance_date",
            sa.Date(),
            nullable=True,
        ),
        schema="attendance",
    )

    op.add_column(
        "videos",
        sa.Column(
            "attendance_date",
            sa.Date(),
            nullable=True,
        ),
        schema="attendance",
    )

    # =========================================================
    # 2. Backfill old development data
    # =========================================================

    op.execute(
        """
        UPDATE attendance.attendance_logs
        SET attendance_date = created_at::date
        WHERE attendance_date IS NULL
        """
    )

    op.execute(
        """
        UPDATE attendance.attendance_records
        SET attendance_date = created_at::date
        WHERE attendance_date IS NULL
        """
    )

    op.execute(
        """
        UPDATE attendance.scan_jobs
        SET attendance_date = created_at::date
        WHERE attendance_date IS NULL
        """
    )

    op.execute(
        """
        UPDATE attendance.videos
        SET attendance_date = created_at::date
        WHERE attendance_date IS NULL
        """
    )

    # scheduler_logs القديمة ليس لديها created_at
    op.execute(
        """
        UPDATE attendance.scheduler_logs
        SET attendance_date = CURRENT_DATE
        WHERE attendance_date IS NULL
        """
    )

    # =========================================================
    # 3. Make NOT NULL
    # =========================================================

    for table in [
        "attendance_logs",
        "attendance_records",
        "scheduler_logs",
        "scan_jobs",
        "videos",
    ]:
        op.alter_column(
            table,
            "attendance_date",
            nullable=False,
            schema="attendance",
        )

    # =========================================================
    # 4. Remove old constraints
    # =========================================================

    op.drop_constraint(
        "uq_attendance_log_session_student_scan",
        "attendance_logs",
        schema="attendance",
        type_="unique",
    )

    op.drop_constraint(
        "uq_attendance_record_session_student",
        "attendance_records",
        schema="attendance",
        type_="unique",
    )

    op.drop_constraint(
        "uq_scan_job_session_scan",
        "scan_jobs",
        schema="attendance",
        type_="unique",
    )

    _drop_video_session_unique()

    # SchedulerLog القديمة يمكن أن تحتوي duplicate
    op.execute(
        """
        DELETE FROM attendance.scheduler_logs a
        USING attendance.scheduler_logs b
        WHERE a.id > b.id
          AND a.session_id = b.session_id
          AND a.attendance_date = b.attendance_date
        """
    )

    # =========================================================
    # 5. New date-aware constraints
    # =========================================================

    op.create_unique_constraint(
        "uq_attendance_log_session_date_student_scan",
        "attendance_logs",
        [
            "session_id",
            "attendance_date",
            "student_id",
            "scan_number",
        ],
        schema="attendance",
    )

    op.create_unique_constraint(
        "uq_attendance_record_session_date_student",
        "attendance_records",
        [
            "session_id",
            "attendance_date",
            "student_id",
        ],
        schema="attendance",
    )

    op.create_unique_constraint(
        "uq_scheduler_log_session_date",
        "scheduler_logs",
        [
            "session_id",
            "attendance_date",
        ],
        schema="attendance",
    )

    op.create_unique_constraint(
        "uq_scan_job_session_date_scan",
        "scan_jobs",
        [
            "session_id",
            "attendance_date",
            "scan_number",
        ],
        schema="attendance",
    )

    op.create_unique_constraint(
        "uq_video_session_date",
        "videos",
        [
            "session_id",
            "attendance_date",
        ],
        schema="attendance",
    )


def downgrade() -> None:

    op.drop_constraint(
        "uq_video_session_date",
        "videos",
        schema="attendance",
        type_="unique",
    )

    op.drop_constraint(
        "uq_scan_job_session_date_scan",
        "scan_jobs",
        schema="attendance",
        type_="unique",
    )

    op.drop_constraint(
        "uq_scheduler_log_session_date",
        "scheduler_logs",
        schema="attendance",
        type_="unique",
    )

    op.drop_constraint(
        "uq_attendance_record_session_date_student",
        "attendance_records",
        schema="attendance",
        type_="unique",
    )

    op.drop_constraint(
        "uq_attendance_log_session_date_student_scan",
        "attendance_logs",
        schema="attendance",
        type_="unique",
    )

    for table in [
        "videos",
        "scan_jobs",
        "scheduler_logs",
        "attendance_records",
        "attendance_logs",
    ]:
        op.drop_column(
            table,
            "attendance_date",
            schema="attendance",
        )