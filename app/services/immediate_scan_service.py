import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import select

from app.config import SCHEDULER_ENABLED
from app.database import AsyncSessionLocal
from app.models.scan_job import ScanJob
from app.models.scheduler_logs import SchedulerLog
from app.models.system_setting import SystemSetting
from app.models.videos import Video
from app.services.scan_window_service import build_scan_windows
from app.services.scan_worker import run_one_scan_job
from app.services.video_segment_service import get_video_duration_seconds
from app.models.attendance_record import AttendanceRecord
from app.enums import AttendanceStatus

MANUAL_SCAN_COUNT = 3

STATUS_WAITING = "WAITING"
STATUS_PENDING = "PENDING"

FINAL_SCAN_PRIORITY = 0
NORMAL_SCAN_PRIORITY = 100

_manual_batch_lock = asyncio.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _prepare_immediate_jobs(
    session_id: int,
) -> List[Dict[str, Any]]:
    async with AsyncSessionLocal() as db:
        settings_result = await db.execute(
            select(SystemSetting)
        )
        settings = settings_result.scalars().first()

        if settings is None:
            raise ValueError(
                "System settings not found."
            )

        video_result = await db.execute(
            select(Video).where(
                Video.session_id == session_id
            )
        )

        video = video_result.scalars().first()

        if video is None:
            raise FileNotFoundError(
                f"No video is registered for session {session_id}."
            )

        video_path = Path(video.path).expanduser()

        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path

        video_path = video_path.resolve()

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Session video does not exist: {video_path}"
            )

        scan_duration_seconds = int(
            settings.scan_duration_seconds
        )

    video_duration_seconds = int(
        await asyncio.to_thread(
            get_video_duration_seconds,
            video_path,
        )
    )

    windows = build_scan_windows(
        video_duration_seconds=video_duration_seconds,
        scan_count=MANUAL_SCAN_COUNT,
        scan_duration_seconds=scan_duration_seconds,
    )

    async with AsyncSessionLocal() as db:
        scheduler_result = await db.execute(
            select(SchedulerLog)
            .where(
                SchedulerLog.session_id == session_id
            )
            .with_for_update()
        )

        scheduler_log = (
            scheduler_result.scalars().first()
        )

        if scheduler_log is None:
            scheduler_log = SchedulerLog(
                session_id=session_id,
                scan_count=0,
                scan_session_count=MANUAL_SCAN_COUNT,
            )

            db.add(scheduler_log)
            await db.flush()

        else:
            current_scan_count = int(
                scheduler_log.scan_count or 0
            )

            if current_scan_count != 0:
                raise ValueError(
                    "This session already contains attendance scans. "
                    "Use a fresh session for the immediate 3-scan demo."
                )

            scheduler_log.scan_session_count = (
                MANUAL_SCAN_COUNT
            )

        jobs_result = await db.execute(
            select(ScanJob)
            .where(
                ScanJob.session_id == session_id
            )
            .with_for_update()
        )

        existing_jobs = (
            jobs_result.scalars().all()
        )

        if existing_jobs:
            raise ValueError(
                "This session already contains ScanJobs. "
                "Use a fresh session for the immediate 3-scan demo."
            )

        now = _utc_now()

        prepared = []

        for window in windows:
            is_first = window.scan_number == 1
            is_final = (
                window.scan_number
                == MANUAL_SCAN_COUNT
            )

            job = ScanJob(
                session_id=session_id,
                scan_number=window.scan_number,
                segment_start_seconds=window.start_seconds,
                segment_duration_seconds=window.duration_seconds,
                priority=(
                    FINAL_SCAN_PRIORITY
                    if is_final
                    else NORMAL_SCAN_PRIORITY
                ),
                status=(
                    STATUS_PENDING
                    if is_first
                    else STATUS_WAITING
                ),
                attempts=0,
                eligible_at=now,
                queued_at=now if is_first else None,
            )

            db.add(job)

            prepared.append(
                {
                    "scan_number": window.scan_number,
                    "start_seconds": window.start_seconds,
                    "end_seconds": window.end_seconds,
                    "duration_seconds": window.duration_seconds,
                }
            )

        await db.commit()

        return prepared


async def _promote_immediate_job(
    session_id: int,
    scan_number: int,
) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ScanJob)
            .where(
                ScanJob.session_id == session_id,
                ScanJob.scan_number == scan_number,
            )
            .with_for_update()
        )

        job = result.scalar_one_or_none()

        if job is None:
            raise RuntimeError(
                f"ScanJob {scan_number} not found "
                f"for session {session_id}."
            )

        job.status = STATUS_PENDING
        job.queued_at = _utc_now()
        job.error_message = None

        await db.commit()


async def _get_attendance_summary(
    session_id: int,
) -> Dict[str, Any]:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(AttendanceRecord)
            .where(
                AttendanceRecord.session_id == session_id
            )
        )

        records = result.scalars().all()

    total_students = len(records)

    present_count = sum(
        1
        for record in records
        if record.status == AttendanceStatus.PRESENT
    )

    absent_count = sum(
        1
        for record in records
        if record.status == AttendanceStatus.ABSENT
    )

    late_count = sum(
        1
        for record in records
        if record.status == AttendanceStatus.LATE
    )

    def percentage(count: int) -> float:
        if total_students == 0:
            return 0.0

        return round(
            (count / total_students) * 100,
            2,
        )

    attendance_count = (
        present_count + late_count
    )

    return {
        "total_students": total_students,
        "present": {
            "count": present_count,
            "percentage": percentage(
                present_count
            ),
        },
        "absent": {
            "count": absent_count,
            "percentage": percentage(
                absent_count
            ),
        },
        "late": {
            "count": late_count,
            "percentage": percentage(
                late_count
            ),
        },
        "attendance_count": attendance_count,
        "attendance_percentage": percentage(
            attendance_count
        ),
        "absence_percentage": percentage(
            absent_count
        ),
    }


async def run_immediate_three_scan_batch(
    ai,
    session_id: int,
) -> Dict[str, Any]:
    if SCHEDULER_ENABLED:
        raise ValueError(
            "Immediate 3-scan demo requires "
            "SCHEDULER_ENABLED=false."
        )

    if _manual_batch_lock.locked():
        raise RuntimeError(
            "Another immediate attendance batch is already running."
        )

    async with _manual_batch_lock:
        windows = await _prepare_immediate_jobs(
            session_id=session_id
        )

        results = []

        for scan_number in range(
            1,
            MANUAL_SCAN_COUNT + 1,
        ):
            if scan_number > 1:
                await _promote_immediate_job(
                    session_id=session_id,
                    scan_number=scan_number,
                )

            worker_result = await run_one_scan_job(
                ai=ai,
                allowed_session_ids=[
                    session_id
                ],
            )

            scan_result = (
                worker_result.get("scan_result")
                or {}
            )

            compact_result = {
                "scan_number": scan_number,
                "succeeded": worker_result.get(
                    "succeeded",
                    False,
                ),
                "job_id": worker_result.get(
                    "job_id"
                ),
                "saved_logs": scan_result.get(
                    "saved_logs",
                    0,
                ),
                "finalized": scan_result.get(
                    "finalized",
                    False,
                ),
                "records_written": scan_result.get(
                    "records_written",
                    0,
                ),
                "error": worker_result.get(
                    "error"
                ),
            }

            results.append(compact_result)

            if not worker_result.get(
                "succeeded",
                False,
            ):
                return {
                    "completed": False,
                    "session_id": session_id,
                    "scan_count": MANUAL_SCAN_COUNT,
                    "windows": windows,
                    "results": results,
                }
        attendance_summary = await _get_attendance_summary(
            session_id=session_id
        )

        return {
            "completed": True,
            "session_id": session_id,
            "scan_count": MANUAL_SCAN_COUNT,
            "windows": windows,
            "results": results,
            "attendance_summary": attendance_summary,
        }
        # return {
        #     "completed": True,
        #     "session_id": session_id,
        #     "scan_count": MANUAL_SCAN_COUNT,
        #     "windows": windows,
        #     "results": results,
        # }