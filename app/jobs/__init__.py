from sqlalchemy import select
from datetime import datetime, time
from app.core.logger import logger
from app.database import AsyncSessionLocal
from app.models.scheduler_logs import SchedulerLog
from app.models.session import Session
from app.models.system_setting import SystemSetting
from app.enums import AttendanceMode

DAYS_OF_WEEK = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


async def check_attendance_automatically():
    logger.warning(f"[{datetime.now()}] Running background attendance check...")

    async with AsyncSessionLocal() as db_session:
        try:
            setting_query = select(SystemSetting)
            setting_result = await db_session.execute(setting_query)
            system_setting = setting_result.scalars().first()

            if not system_setting or not system_setting.is_active:
                logger.warning("System settings are inactive. Skipping background check.")
                return

            if system_setting.attendance_mode != AttendanceMode.AUTOMATIC:
                logger.info("Attendance mode is not AUTOMATIC. Skipping background check.")
                return
            max_scans = system_setting.times_per_session

            current_time = datetime.now().time()
            current_date = datetime.now().date()
            day_index = current_date.weekday()
            current_day_str = DAYS_OF_WEEK[day_index]

            query_session = select(Session).where(
                Session.date == current_day_str,
                Session.start_time <= current_time,
                Session.end_time > current_time
            )

            result_session = await db_session.execute(query_session)
            active_sessions = result_session.scalars().all()
            if not active_sessions:
                logger.info("No active classes right now.")
                return

            active_session_ids = [s.id for s in active_sessions]

            query_schedule = select(SchedulerLog).where(SchedulerLog.session_id.in_(active_session_ids))
            result_schedule = await db_session.execute(query_schedule)
            existing_logs = result_schedule.scalars().all()

            logs_map = {log.session_id: log for log in existing_logs}

            for active_session in active_sessions:
                sched_log = logs_map.get(active_session.id)

                if not sched_log:
                    logger.warning(f"Running AI Scan [1/{max_scans}] for Classroom ID: {active_session.classroom_id}")

                    # await run_ai_face_recognition(session_id=active_session.id)

                    new_log = SchedulerLog(
                        session_id=active_session.id,
                        scan_count=1,
                        scan_session_count=1
                    )
                    db_session.add(new_log)

                elif sched_log.scan_count < max_scans:
                    next_scan_num = sched_log.scan_count + 1
                    logger.warning(f"Running AI Scan [{next_scan_num}/{max_scans}] for Classroom ID: {active_session.classroom_id}")

                    # await run_ai_face_recognition(session_id=active_session.id)

                    sched_log.scan_count += 1
                    sched_log.scan_session_count += 1

                else:
                    logger.info(f"Classroom {active_session.classroom_id} already reached maximum scans ({max_scans}/{max_scans}). Skipping.")

            await db_session.commit()
            logger.warning(f"[{datetime.now()}] Background attendance check completed successfully.")

        except Exception as e:
            logger.warning(f"Error in background task: {e}")
            await db_session.rollback()

