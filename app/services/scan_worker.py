import asyncio
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select

from app.core.logger import logger
from app.database import AsyncSessionLocal
from app.models.scan_job import ScanJob
from app.models.scheduler_logs import SchedulerLog
from app.models.session import Session
from app.models.videos import Video
from app.services.attendance_scan_service import process_attendance_scan
from app.services.video_segment_service import extract_video_segment
from datetime import (
    date,
    datetime,
    timezone,
)

from app.enums import AttendanceMode

from app.models.system_setting import (
    SystemSetting,
)

STATUS_WAITING = "WAITING"
STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ClaimedScanJob:

    job_id: int
    session_id: int
    attendance_date: date
    scan_number: int

    priority: int

    segment_start_seconds: int
    segment_duration_seconds: int

    attempts: int
    

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def claim_next_scan_job(
    allowed_session_ids=None,
    allowed_attendance_date=None,
):
    async with AsyncSessionLocal() as db:
        try:
            query = (
                select(ScanJob)
                .join(
                    Session,
                    Session.id == ScanJob.session_id,
                )
                .where(
                    ScanJob.status == STATUS_PENDING
                )
            )

            if allowed_session_ids is not None:
                normalized_ids = [
                    int(value)
                    for value in allowed_session_ids
                ]

                if not normalized_ids:
                    await db.rollback()
                    return None

                query = query.where(
                    ScanJob.session_id.in_(
                        normalized_ids
                    )
                )
            if allowed_attendance_date is not None:
                query = query.where(
                    ScanJob.attendance_date
                    == allowed_attendance_date
                )

            query = (
                query
                .order_by(
                    ScanJob.priority.asc(),
                    Session.end_time.asc(),
                    ScanJob.queued_at.asc().nulls_last(),
                    ScanJob.id.asc(),
                )
                .limit(1)
                .with_for_update(
                    skip_locked=True,
                    of=ScanJob,
                )
            )

            result = await db.execute(
                query
            )

            job = result.scalars().first()

            if job is None:
                await db.rollback()
                return None

            job.status = STATUS_RUNNING
            job.started_at = _utc_now()
            job.finished_at = None
            job.error_message = None
            job.attempts = int(
                job.attempts or 0
            ) + 1

            await db.flush()

            claimed = ClaimedScanJob(
                job_id=int(job.id),
                session_id=int(job.session_id),

                attendance_date=
                    job.attendance_date,

                scan_number=int(job.scan_number),
                priority=int(job.priority),

                segment_start_seconds=int(
                    job.segment_start_seconds
                ),

                segment_duration_seconds=int(
                    job.segment_duration_seconds
                ),

                attempts=int(job.attempts),
    )

            await db.commit()

            logger.info(
                "Claimed ScanJob "
                f"id={claimed.job_id}, "
                f"session={claimed.session_id}, "
                f"scan={claimed.scan_number}"
            )

            return claimed

        except Exception:
            await db.rollback()

            logger.exception(
                "Failed to claim ScanJob"
            )

            raise


async def _get_video_path(
    session_id: int,
    attendance_date: date,
):

    async with AsyncSessionLocal() as db:

        result = await db.execute(
            select(Video).where(
                Video.session_id
                == session_id,

                Video.attendance_date
                == attendance_date,
            )
        )

        video = (
            result.scalar_one_or_none()
        )

        if video is None:
            raise FileNotFoundError(
                "No video is registered "
                f"for session {session_id} "
                f"on {attendance_date}."
            )

        source_path = Path(
            video.path
        ).expanduser()

        if not source_path.is_absolute():
            source_path = (
                Path.cwd()
                / source_path
            )

        source_path = (
            source_path.resolve()
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                "Session video does not "
                f"exist: {source_path}"
            )

        return source_path


async def _check_job_sequence(
    job: ClaimedScanJob,
) -> bool:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
           select(SchedulerLog).where(
            SchedulerLog.session_id
            == job.session_id,

            SchedulerLog.attendance_date
            == job.attendance_date,
        )
        )

        scheduler_log = (
            result.scalars().first()
        )

        if scheduler_log is None:
            if job.scan_number == 1:
                return False

            raise RuntimeError(
                f"SchedulerLog missing for session "
                f"{job.session_id}, but job requires "
                f"scan {job.scan_number}."
            )

        current_scan_count = int(
            scheduler_log.scan_count or 0
        )

        if current_scan_count >= job.scan_number:
            return True

        expected_next = (
            current_scan_count + 1
        )

        if expected_next != job.scan_number:
            raise RuntimeError(
                f"ScanJob is out of sequence for session "
                f"{job.session_id}: "
                f"job={job.scan_number}, "
                f"expected={expected_next}."
            )

        return False


async def _mark_job_succeeded(
    job_id: int,
) -> None:
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(ScanJob)
                .where(
                    ScanJob.id == job_id
                )
                .with_for_update()
            )

            job = result.scalar_one_or_none()

            if job is None:
                raise RuntimeError(
                    f"ScanJob {job_id} not found."
                )

            job.status = STATUS_SUCCEEDED
            job.finished_at = _utc_now()
            job.error_message = None

            await db.commit()

        except Exception:
            await db.rollback()
            raise


async def _mark_job_failed(
    job_id: int,
    error_message: str,
) -> None:
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(ScanJob)
                .where(
                    ScanJob.id == job_id
                )
                .with_for_update()
            )

            job = result.scalar_one_or_none()

            if job is None:
                await db.rollback()
                return

            job.status = STATUS_FAILED
            job.finished_at = _utc_now()

            job.error_message = (
                error_message[:4000]
            )

            await db.commit()

        except Exception:
            await db.rollback()

            logger.exception(
                f"Failed to mark ScanJob {job_id} as FAILED"
            )


async def execute_claimed_scan_job(
    ai,
    job: ClaimedScanJob,
) -> Dict[str, Any]:
    try:
        already_completed = (
            await _check_job_sequence(
                job
            )
        )

        if already_completed:
            await _mark_job_succeeded(
                job.job_id
            )

            return {
                "found": True,
                "job_id": job.job_id,
                "session_id": job.session_id,
                "scan_number": job.scan_number,
                "succeeded": True,
                "reconciled": True,
                "message": (
                    "Scan was already committed in "
                    "SchedulerLog. Job reconciled."
                ),
            }

        source_path = await _get_video_path(
            job.session_id,
            job.attendance_date
        )

        with tempfile.TemporaryDirectory(
            prefix=(
                f"attendance_scan_"
                f"{job.session_id}_"
                f"{job.scan_number}_"
            )
        ) as temp_dir:
            segment_path = (
                Path(temp_dir)
                / (
                    f"session_{job.session_id}_"
                    f"scan_{job.scan_number}.mp4"
                )
            )

            segment_info = await asyncio.to_thread(
                extract_video_segment,
                source_path,
                segment_path,
                job.segment_start_seconds,
                job.segment_duration_seconds,
            )

            logger.info(
                "Prepared scan segment: "
                f"session={job.session_id}, "
                f"scan={job.scan_number}, "
                f"start={job.segment_start_seconds}s, "
                f"duration={job.segment_duration_seconds}s, "
                f"frames={segment_info.frames_written}"
            )

            async with AsyncSessionLocal() as attendance_db:
                scan_result = (
                    await process_attendance_scan(
                        db=attendance_db,
                        ai=ai,
                        session_id=job.session_id,
                        video_path=str(
                            segment_path
                        ),
                        max_frames=0,
                        expected_scan_number=(
                            job.scan_number
                        ),
                        attendance_date=job.attendance_date,
                    )
                )

        # process_attendance_scan committed successfully here.
        # From this point onward we MUST NOT mark the scan failed
        # merely because synchronizing ScanJob status fails.
        try:
            await _mark_job_succeeded(
                job.job_id
            )

        except Exception:
            logger.exception(
                "Attendance scan committed successfully, "
                f"but ScanJob {job.job_id} could not be "
                "marked SUCCEEDED. Startup recovery will "
                "reconcile it."
            )

            return {
                "found": True,
                "job_id": job.job_id,
                "session_id": job.session_id,
                "scan_number": job.scan_number,
                "succeeded": True,
                "job_status_sync_error": True,
                "scan_result": scan_result,
            }

        return {
            "found": True,
            "job_id": job.job_id,
            "session_id": job.session_id,
            "scan_number": job.scan_number,
            "succeeded": True,
            "reconciled": False,
            "scan_result": scan_result,
        }

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        logger.exception(
            "ScanJob failed: "
            f"job={job.job_id}, "
            f"session={job.session_id}, "
            f"scan={job.scan_number}"
        )

        await _mark_job_failed(
            job_id=job.job_id,
            error_message=error_message,
        )

        return {
            "found": True,
            "job_id": job.job_id,
            "session_id": job.session_id,
            "scan_number": job.scan_number,
            "succeeded": False,
            "error": error_message,
        }


async def _automatic_worker_enabled():
    async with AsyncSessionLocal() as db:

        result = await db.execute(
            select(SystemSetting)
        )

        setting = (
            result.scalars().first()
        )

        return bool(
            setting
            and setting.is_active
            and (
                setting.attendance_mode
                == AttendanceMode.AUTOMATIC
            )
        )



async def run_one_scan_job(
    ai,
    allowed_session_ids=None,
    allowed_attendance_date=None,
):

    job = await claim_next_scan_job(
        allowed_session_ids=
            allowed_session_ids,

        allowed_attendance_date=
            allowed_attendance_date,
    )

    if job is None:
        return {
            "found": False,
            "succeeded": False,
            "message":
                "No pending scan jobs.",
        }

    return await execute_claimed_scan_job(
        ai=ai,
        job=job,
    )


# async def run_one_scan_job(
#     ai,
#     allowed_session_ids: Optional[Sequence[int]] = None,
# ) -> Dict[str, Any]:
#     job = await claim_next_scan_job(
#         allowed_session_ids=allowed_session_ids
#     )

#     if job is None:
#         return {
#             "found": False,
#             "succeeded": False,
#             "message": "No pending scan jobs.",
#         }

#     return await execute_claimed_scan_job(
#         ai=ai,
#         job=job,
#     )


async def recover_running_scan_jobs(
    max_attempts: int = 3,
) -> Dict[str, int]:
    if max_attempts < 1:
        raise ValueError(
            "max_attempts must be greater than zero."
        )

    recovered_pending = 0
    recovered_succeeded = 0
    recovered_failed = 0

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(ScanJob)
                .where(
                    ScanJob.status
                    == STATUS_RUNNING
                )
                .with_for_update(
                    skip_locked=True
                )
            )


            jobs = result.scalars().all()

            now = _utc_now()

            for job in jobs:
                scheduler_result = await db.execute(
                    select(SchedulerLog)
                    .where(
                        SchedulerLog.session_id
                        == job.session_id ,
                        SchedulerLog.attendance_date
                            == job.attendance_date
                    )
                )

                scheduler_log = (
                    scheduler_result.scalars().first()
                )

                current_scan_count = (
                    int(
                        scheduler_log.scan_count
                        or 0
                    )
                    if scheduler_log is not None
                    else 0
                )

                if (
                    current_scan_count
                    >= int(job.scan_number)
                ):
                    job.status = STATUS_SUCCEEDED
                    job.finished_at = now
                    job.error_message = None

                    recovered_succeeded += 1
                    continue

                if int(job.attempts or 0) >= max_attempts:
                    job.status = STATUS_FAILED
                    job.finished_at = now
                    job.error_message = (
                        "Recovered after application restart, "
                        "but maximum attempts were reached."
                    )

                    recovered_failed += 1
                    continue

                job.status = STATUS_PENDING
                job.queued_at = now
                job.started_at = None
                job.finished_at = None
                job.error_message = (
                    "Recovered after application restart."
                )

                recovered_pending += 1

            await db.commit()

            return {
                "pending": recovered_pending,
                "succeeded": recovered_succeeded,
                "failed": recovered_failed,
            }

        except Exception:
            await db.rollback()

            logger.exception(
                "Failed to recover running ScanJobs"
            )

            raise

async def scan_worker_loop(
    ai,
    stop_event: asyncio.Event,
    poll_seconds: float = 2.0,
) -> None:

    if poll_seconds <= 0:
        raise ValueError(
            "poll_seconds must be greater than zero."
        )

    logger.info(
        "Attendance scan worker started."
    )

    try:

        while not stop_event.is_set():

            try:

                # ==========================================
                # Background Worker = AUTOMATIC only
                # ==========================================

                automatic_enabled = (
                    await _automatic_worker_enabled()
                )

                if automatic_enabled:

                    result = (
                        await run_one_scan_job(
                            ai=ai
                        )
                    )

                    if result.get("found"):
                        continue

            except Exception:

                logger.exception(
                    "Unexpected error in "
                    "scan worker loop"
                )

            try:

                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=poll_seconds,
                )

            except asyncio.TimeoutError:
                pass

    finally:

        logger.info(
            "Attendance scan worker stopped."
        )
# async def scan_worker_loop(
#     ai,
#     stop_event: asyncio.Event,
#     poll_seconds: float = 2.0,
# ) -> None:
#     if poll_seconds <= 0:
#         raise ValueError(
#             "poll_seconds must be greater than zero."
#         )

#     logger.info(
#         "Attendance scan worker started."
#     )

#     try:
#         while not stop_event.is_set():
#             try:
#                 result = await run_one_scan_job(
#                     ai=ai
#                 )

#                 if result.get("found"):
#                     # Queue may contain another job.
#                     # Continue immediately.
#                     continue

#             except Exception:
#                 logger.exception(
#                     "Unexpected error in scan worker loop"
#                 )

#             try:
#                 await asyncio.wait_for(
#                     stop_event.wait(),
#                     timeout=poll_seconds,
#                 )

#             except asyncio.TimeoutError:
#                 pass

#     finally:
#         logger.info(
#             "Attendance scan worker stopped."
#         )