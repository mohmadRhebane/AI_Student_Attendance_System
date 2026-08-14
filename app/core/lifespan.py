from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logger import logger
from app.database import AsyncSessionLocal, engine
from app.models.system_setting import SystemSetting
from app.enums import AttendanceMode
# from app.core.scan_scheduler import scheduler, schedule_daily_sessions

from app.config import AI_PROJECT_ROOT
from app.services.ai_runtime import create_ai
import asyncio

from app.config import (
    SCAN_JOB_MAX_ATTEMPTS,
    SCAN_PLANNER_INTERVAL_MINUTES,
    SCAN_WORKER_POLL_SECONDS,
    SCHEDULER_ENABLED,
)
from app.core.scan_scheduler import (
    start_scan_scheduler,
    stop_scan_scheduler,
)
from app.services.scan_planner import run_scan_planner
from app.services.scan_worker import (
    recover_running_scan_jobs,
    scan_worker_loop,
)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================================
    # 1. مرحلة الإقلاع (Startup): إعداد قاعدة البيانات
    # ==========================================
    async with AsyncSessionLocal() as session:
        query = await session.execute(select(SystemSetting))
        existing_setting = query.scalars().first()

        if not existing_setting:
            default_setting = SystemSetting(
                attendance_mode=AttendanceMode.AUTOMATIC,
                times_per_session=2,
                is_active=True,
                min_confidence_score=0.50
            )
            session.add(default_setting)
            await session.commit()
            await session.refresh(default_setting)
            logger.warning(f"Default setting: {default_setting.id} System Settings initialized successfully.")
        else:
            logger.warning("System Settings already exist. Skipping initialization.")

    # ==========================================
    # 2. مرحلة الإقلاع (Startup): تشغيل الجدولة
    # ==========================================

    # scheduler.add_job(schedule_daily_sessions, 'cron', hour=6, minute=0)

    # scheduler.start()
    # logger.warning("APScheduler started successfully.")

    # await schedule_daily_sessions()

    app.state.ai = create_ai(AI_PROJECT_ROOT)
    logger.warning("Smart Attendance AI initialized successfully.")
    app.state.scan_worker_stop_event = None
    app.state.scan_worker_task = None

    if SCHEDULER_ENABLED:
        recovery = await recover_running_scan_jobs(
            max_attempts=SCAN_JOB_MAX_ATTEMPTS
        )

        logger.info(
            f"ScanJob recovery completed: {recovery}"
        )

        planner_report = await run_scan_planner()

        logger.info(
            "Initial scan planner completed: "
            f"created={planner_report.get('created_jobs_count', 0)}, "
            f"queued={planner_report.get('queued_jobs_count', 0)}"
        )

        scan_worker_stop_event = asyncio.Event()

        app.state.scan_worker_stop_event = (
            scan_worker_stop_event
        )

        app.state.scan_worker_task = (
            asyncio.create_task(
                scan_worker_loop(
                    ai=app.state.ai,
                    stop_event=scan_worker_stop_event,
                    poll_seconds=SCAN_WORKER_POLL_SECONDS,
                ),
                name="attendance-scan-worker",
            )
        )

        start_scan_scheduler(
            interval_minutes=(
                SCAN_PLANNER_INTERVAL_MINUTES
            )
        )

        logger.info(
            "Automatic attendance scheduler enabled."
        )

    else:
        logger.info(
            "Automatic attendance scheduler disabled."
        )
    # ==========================================
    # 3. مرحلة التشغيل (Yield)
    # ==========================================
    yield
    if SCHEDULER_ENABLED:
        stop_scan_scheduler()

        stop_event = getattr(
            app.state,
            "scan_worker_stop_event",
            None,
        )

        worker_task = getattr(
            app.state,
            "scan_worker_task",
            None,
        )

        if stop_event is not None:
            stop_event.set()

        if worker_task is not None:
            await worker_task

        logger.info(
            "Attendance scan worker shut down safely."
        )   

    app.state.ai.shutdown()
    logger.warning("Smart Attendance AI shut down successfully.")
  
    # ==========================================
    # 4. مرحلة الإغلاق (Shutdown): تنظيف وإيقاف
    # ==========================================
    # scheduler.shutdown()
    # logger.warning("APScheduler shut down successfully.")

    logger.warning("Server shutting down... Data is safe.")

  # async with AsyncSessionLocal() as session:
    #    await session.execute(delete(SystemSetting))
    #    await session.commit()
    #    logger.warning("System Settings deleted on shutdown.")