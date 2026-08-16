from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import logger
from app.database import AsyncSessionLocal
from app.enums import AttendanceMode
from app.models.scan_job import ScanJob
from app.models.scheduler_logs import SchedulerLog
from app.models.session import Session
from app.models.system_setting import SystemSetting
from app.services.scan_window_service import build_scan_windows

import asyncio
from pathlib import Path

from app.models.videos import Video
from app.services.video_segment_service import get_video_duration_seconds
DAYS_OF_WEEK = [
    "MON",
    "TUE",
    "WED",
    "THU",
    "FRI",
    "SAT",
    "SUN",
]


STATUS_WAITING = "WAITING"
STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"


FINAL_SCAN_PRIORITY = 0
NORMAL_SCAN_PRIORITY = 100


def _combine_local_datetime(
        current_date,
        session_time,
        tzinfo,
) -> datetime:
    value = datetime.combine(
        current_date,
        session_time,
    )

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=tzinfo,
        )

    return value


async def plan_scan_jobs(
        db: AsyncSession,
        now: Optional[datetime] = None,
        max_attempts: int = 3,
) -> Dict[str, Any]:
    # ---------------------------------------------------------
    # 1. Current local time
    # ---------------------------------------------------------
    if now is None:
        now = datetime.now().astimezone()

    if now.tzinfo is None:
        now = now.astimezone()

    # ---------------------------------------------------------
    # 2. Read system settings
    # ---------------------------------------------------------
    settings_result = await db.execute(
        select(SystemSetting)
    )

    settings = settings_result.scalars().first()

    if settings is None:
        return {
            "enabled": False,
            "reason": "System settings not found.",
            "created_jobs": [],
            "queued_jobs": [],
        }

    if not settings.is_active:
        return {
            "enabled": False,
            "reason": "System settings are inactive.",
            "created_jobs": [],
            "queued_jobs": [],
        }

    if settings.attendance_mode != AttendanceMode.AUTOMATIC:
        return {
            "enabled": False,
            "reason": "Attendance mode is not automatic.",
            "created_jobs": [],
            "queued_jobs": [],
        }

    default_scan_count = int(
        settings.times_per_session
    )

    scan_duration_seconds = int(
        settings.scan_duration_seconds
    )

    if default_scan_count < 1:
        raise ValueError(
            "times_per_session must be greater than zero."
        )

    if scan_duration_seconds < 1:
        raise ValueError(
            "scan_duration_seconds must be greater than zero."
        )

    # ---------------------------------------------------------
    # 3. Find today's sessions
    # ---------------------------------------------------------
    current_date = now.date()

    current_day_str = DAYS_OF_WEEK[
        current_date.weekday()
    ]

    session_result = await db.execute(
        select(Session)
        .where(
            Session.date == current_day_str
        )
        .order_by(
            Session.start_time,
            Session.id,
        )
    )

    sessions = session_result.scalars().all()

    created_jobs: List[Dict[str, Any]] = []
    queued_jobs: List[Dict[str, Any]] = []
    skipped_sessions: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # 4. Process every session independently
    # ---------------------------------------------------------
    for session_obj in sessions:
        session_start = _combine_local_datetime(
            current_date=current_date,
            session_time=session_obj.start_time,
            tzinfo=now.tzinfo,
        )

        session_end = _combine_local_datetime(
            current_date=current_date,
            session_time=session_obj.end_time,
            tzinfo=now.tzinfo,
        )

        if session_end <= session_start:
            skipped_sessions.append(
                {
                    "session_id": session_obj.id,
                    "reason": "Invalid session duration.",
                }
            )
            continue

        session_duration_seconds = int(
            (
                session_end
                - session_start
            ).total_seconds()
        )

        # -----------------------------------------------------
        # 5. SchedulerLog = progress for this session
        # -----------------------------------------------------
        scheduler_result = await db.execute(
            select(SchedulerLog)
            .where(
                SchedulerLog.session_id
                == session_obj.id,
                SchedulerLog.attendance_date
                == current_date,
            )
        )

        scheduler_log = (
            scheduler_result.scalars().first()
        )

        if scheduler_log is None:
            scheduler_log = SchedulerLog(
                session_id=session_obj.id,

                attendance_date=
                    current_date,

                scan_count=0,

                scan_session_count=
                    default_scan_count,
            )

            db.add(scheduler_log)

            await db.flush()

        # Important:
        # once a session has been planned,
        # keep its original number of scans.
        max_scans = int(
            scheduler_log.scan_session_count
            or default_scan_count
        )

        if max_scans < 1:
            skipped_sessions.append(
                {
                    "session_id": session_obj.id,
                    "reason": "Invalid scan_session_count.",
                }
            )
            continue

        # -----------------------------------------------------
        # 6. Calculate fixed windows for the session
        # -----------------------------------------------------
        try:
            video_result = await db.execute(
                
            select(Video).where(
                    Video.session_id
                    == session_obj.id,

                    Video.attendance_date
                    == current_date,
                )
            )

            video = video_result.scalars().first()

            if video is None:
                skipped_sessions.append(
                    {
                        "session_id": session_obj.id,
                        "reason": "VIDEO_NOT_FOUND",
                    }
                )
                continue

            video_path = Path(video.path).resolve()

            try:
                video_duration_seconds = int(
                    await asyncio.to_thread(
                        get_video_duration_seconds,
                        video_path,
                    )
                )
            except Exception as exc:
                skipped_sessions.append(
                    {
                        "session_id": session_obj.id,
                        "reason": f"VIDEO_INVALID: {exc}",
                    }
                )
                continue
            
            windows = build_scan_windows(
                video_duration_seconds=video_duration_seconds,
                scan_count=max_scans,
                scan_duration_seconds=scan_duration_seconds,
            )

        except ValueError as exc:
            skipped_sessions.append(
                {
                    "session_id": session_obj.id,
                    "reason": str(exc),
                }
            )
            continue

        # -----------------------------------------------------
        # 7. Load existing ScanJobs
        # -----------------------------------------------------
        jobs_result = await db.execute(
            select(ScanJob)
            .where(
                ScanJob.session_id
                == session_obj.id,
                ScanJob.attendance_date
    == current_date,

            )
            .order_by(
                ScanJob.scan_number
            )
        )

        existing_jobs = jobs_result.scalars().all()

        jobs_by_scan = {
            int(job.scan_number): job
            for job in existing_jobs
        }

        # -----------------------------------------------------
        # 8. Create missing jobs
        # -----------------------------------------------------
        for window in windows:
            if window.scan_number in jobs_by_scan:
                continue

            eligible_at = (
                session_start
                + timedelta(
                    seconds=window.end_seconds
                )
            )

            priority = (
                FINAL_SCAN_PRIORITY
                if window.scan_number == max_scans
                else NORMAL_SCAN_PRIORITY
            )
            job = ScanJob(
                session_id=session_obj.id,

                attendance_date=
                    current_date,

                scan_number=
                    window.scan_number,

                segment_start_seconds=
                    window.start_seconds,

                segment_duration_seconds=
                    window.duration_seconds,

                priority=priority,

                status=STATUS_WAITING,

                attempts=0,

                eligible_at=eligible_at,
            )

            db.add(job)

            jobs_by_scan[
                window.scan_number
            ] = job

            created_jobs.append(
                {
                    "session_id": session_obj.id,
                    "scan_number": window.scan_number,
                    "segment_start_seconds": window.start_seconds,
                    "segment_end_seconds": window.end_seconds,
                    "eligible_at": eligible_at.isoformat(),
                    "priority": priority,
                }
            )

        await db.flush()

        # -----------------------------------------------------
        # 9. Session already completed
        # -----------------------------------------------------
        current_scan_count = int(
            scheduler_log.scan_count or 0
        )

        if current_scan_count >= max_scans:
            continue

        # -----------------------------------------------------
        # 10. Critical rule:
        # Only the NEXT required scan may enter the queue.
        # -----------------------------------------------------
        next_scan_number = (
            current_scan_count + 1
        )

        next_job = jobs_by_scan.get(
            next_scan_number
        )

        if next_job is None:
            skipped_sessions.append(
                {
                    "session_id": session_obj.id,
                    "reason": (
                        "Next ScanJob does not exist: "
                        f"{next_scan_number}"
                    ),
                }
            )
            continue

        # -----------------------------------------------------
        # 11. Never queue two jobs for same session
        # -----------------------------------------------------
        active_job_exists = any(
            job.status
            in {
                STATUS_PENDING,
                STATUS_RUNNING,
            }
            for job in jobs_by_scan.values()
        )

        if active_job_exists:
            continue

        # -----------------------------------------------------
        # 12. Segment must be fully available first
        # -----------------------------------------------------
        if next_job.eligible_at > now:
            continue

        # -----------------------------------------------------
        # 13. Failed job may retry
        # -----------------------------------------------------
        if (
            next_job.status == STATUS_FAILED
            and int(next_job.attempts or 0) >= max_attempts
        ):
            continue

        # -----------------------------------------------------
        # 14. WAITING / retry FAILED -> PENDING
        # -----------------------------------------------------
        if next_job.status in {
            STATUS_WAITING,
            STATUS_FAILED,
        }:
            next_job.status = STATUS_PENDING

            next_job.priority = (
                FINAL_SCAN_PRIORITY
                if next_scan_number == max_scans
                else NORMAL_SCAN_PRIORITY
            )

            next_job.queued_at = now
            next_job.error_message = None

            queued_jobs.append(
                {
                    "job_id": next_job.id,
                    "session_id": session_obj.id,
                    "scan_number": next_scan_number,
                    "max_scans": max_scans,
                    "priority": next_job.priority,
                    "segment_start_seconds": (
                        next_job.segment_start_seconds
                    ),
                    "segment_duration_seconds": (
                        next_job.segment_duration_seconds
                    ),
                }
            )

    return {
        "enabled": True,
        "planner_time": now.isoformat(),
        "day": current_day_str,
        "sessions_found": len(sessions),
        "created_jobs_count": len(
            created_jobs
        ),
        "queued_jobs_count": len(
            queued_jobs
        ),
        "created_jobs": created_jobs,
        "queued_jobs": queued_jobs,
        "skipped_sessions": skipped_sessions,
    }


async def run_scan_planner() -> Dict[str, Any]:
    async with AsyncSessionLocal() as db:
        try:
            report = await plan_scan_jobs(
                db=db
            )

            await db.commit()

            logger.info(
                "Scan planner completed: "
                f"created={report.get('created_jobs_count', 0)}, "
                f"queued={report.get('queued_jobs_count', 0)}"
            )

            return report

        except Exception:
            await db.rollback()

            logger.exception(
                "Scan planner failed"
            )

            raise