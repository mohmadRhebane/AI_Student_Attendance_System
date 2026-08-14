from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.scheduler_logs import SchedulerLog
from app.models.system_setting import SystemSetting

from app.services.attendance_ai import run_attendance_video
from app.services.attendance_persistence import (
    save_scan_logs,
    finalize_attendance,
)
from typing import Optional

async def process_attendance_scan(
    db: AsyncSession,
    ai,
    session_id: int,
    video_path: str,
    max_frames: int = 0,
    expected_scan_number: Optional[int] = None,
):
    """
    Execute exactly ONE attendance scan.

    Flow:
    1. Read times_per_session.
    2. Determine the next scan number.
    3. Run AI.
    4. Store positive attendance evidence in attendance_logs.
    5. Increment scan_count only after successful AI + log persistence.
    6. On the final scan, build attendance_records.
    7. Commit the whole operation atomically.
    """

    settings_result = await db.execute(
        select(SystemSetting)
    )

    settings = settings_result.scalars().first()

    if not settings:
        raise ValueError("System settings not found.")

    configured_max_scans = int(settings.times_per_session)

    if configured_max_scans < 1:
        raise ValueError(
            "times_per_session must be greater than zero."
        )

    scheduler_result = await db.execute(
        select(SchedulerLog)
        .where(
            SchedulerLog.session_id == session_id
        )
        .with_for_update()
    )

    scheduler_log = scheduler_result.scalars().first()

    if scheduler_log is None:
        scheduler_log = SchedulerLog(
            session_id=session_id,
            scan_count=0,
            scan_session_count=configured_max_scans,
        )

        db.add(scheduler_log)
        await db.flush()

    max_scans = int(
        scheduler_log.scan_session_count
    )

    if scheduler_log.scan_count >= max_scans:
        raise ValueError(
            f"Session {session_id} already completed "
            f"all {max_scans} scans."
        )

    scan_number = int(scheduler_log.scan_count) + 1
    if (
    expected_scan_number is not None
    and scan_number != expected_scan_number
    ):
        await db.rollback()

        raise ValueError(
            f"Scan sequence mismatch for session {session_id}: "
            f"expected {expected_scan_number}, "
            f"but next scan is {scan_number}."
        )
    
    try:
        # -----------------------------
        # 1. Run AI
        # -----------------------------
        ai_result = await run_attendance_video(
            db=db,
            ai=ai,
            session_id=session_id,
            video_path=video_path,
            max_frames=max_frames,
        )

        if not ai_result.succeeded:
            raise RuntimeError(
                f"AI scan failed: {ai_result.error_message}"
            )

        # -----------------------------
        # 2. Save this scan
        # -----------------------------
        saved_logs = await save_scan_logs(
            db=db,
            session_id=session_id,
            scan_number=scan_number,
            ai_result=ai_result,
        )

        # -----------------------------
        # 3. Increment only after success
        # -----------------------------
        scheduler_log.scan_count = scan_number

        # -----------------------------
        # 4. Finalize only on last scan
        # -----------------------------
        finalized = False
        records_written = 0

        if scan_number >= max_scans:
            records_written = await finalize_attendance(
                db=db,
                session_id=session_id,
            )

            finalized = True

        # -----------------------------
        # 5. One atomic transaction
        # -----------------------------
        await db.commit()

        return {
            "session_id": session_id,
            "scan_number": scan_number,
            "max_scans": max_scans,
            "saved_logs": saved_logs,
            "finalized": finalized,
            "records_written": records_written,
            "ai_result": ai_result,
        }

    except Exception:
        await db.rollback()
        raise