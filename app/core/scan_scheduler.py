
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.logger import logger
from app.services.scan_planner import run_scan_planner


scheduler = AsyncIOScheduler()


def start_scan_scheduler(
    interval_minutes: int = 5,
) -> None:
    if interval_minutes < 1:
        raise ValueError(
            "interval_minutes must be at least 1."
        )

    scheduler.add_job(
        run_scan_planner,
        trigger="interval",
        minutes=interval_minutes,
        id="attendance-scan-planner",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=60,
    )

    if not scheduler.running:
        scheduler.start()

    logger.info(
        "Attendance scan planner scheduler started: "
        f"interval={interval_minutes} minutes."
    )


def stop_scan_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(
            wait=False
        )

        logger.info(
            "Attendance scan planner scheduler stopped."
        )

# from datetime import datetime, timedelta
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from sqlalchemy import select
# from app.core.logger import logger
# from app.database import AsyncSessionLocal
# from app.models.scheduler_logs import SchedulerLog
# from app.models.session import Session
# from app.models.system_setting import SystemSetting
# from app.enums import AttendanceMode

# scheduler = AsyncIOScheduler()
# DAYS_OF_WEEK = [
#     "MON",
#     "TUE",
#     "WED",
#     "THU",
#     "FRI",
#     "SAT",
#     "SUN",
# ]

# # ---------------------------------------------------------
# # 1. دالة تنفيذ المسح (تعمل في الوقت الدقيق الذي حدده المخطط)
# # ---------------------------------------------------------
# async def execute_ai_scan(session_id: int):
#     logger.warning(f"[{datetime.now()}] Triggering scheduled AI Scan for Session ID: {session_id}")

#     async with AsyncSessionLocal() as db_session:
#         try:
#             # 1. تحديث عداد المسحات في SchedulerLog
#             query_log = select(SchedulerLog).where(SchedulerLog.session_id == session_id)
#             sched_log = (await db_session.execute(query_log)).scalars().first()

#             if sched_log:
#                 sched_log.scan_count += 1

#             # 2. تشغيل الكاميرا والذكاء الاصطناعي (أضف دالتك هنا)
#             # await run_ai_face_recognition(session_id=session_id)

#             # 3. إدراج النتائج في AttendanceLogs (يتم داخل دالة الذكاء الاصطناعي أو هنا)

#             await db_session.commit()
#             logger.info(
#                 f" Scan successful for Session {session_id}. Count: {sched_log.scan_count}/{sched_log.scan_session_count}")

#         except Exception as e:
#             await db_session.rollback()
#             logger.error(f" Error executing scan for session {session_id}: {e}")


# # ---------------------------------------------------------
# # 2. خافرة الصباح (تعمل مرة واحدة لتوزيع أوقات اليوم بالكامل)
# # ---------------------------------------------------------
# async def schedule_daily_sessions():
#     logger.warning(f"[{datetime.now()}] Running morning planner to schedule today's scans...")

#     async with AsyncSessionLocal() as db_session:
#         try:
#             # 1. فحص إعدادات النظام (من الكود الخاص بك)
#             setting_result = await db_session.execute(select(SystemSetting))
#             system_setting = setting_result.scalars().first()

#             if not system_setting or not system_setting.is_active:
#                 logger.warning("System settings are inactive. Skipping scheduling.")
#                 return

#             if system_setting.attendance_mode != AttendanceMode.AUTOMATIC:
#                 logger.info("Attendance mode is not AUTOMATIC. Skipping scheduling.")
#                 return

#             times_per_session = system_setting.times_per_session

#             # 2. تحديد اليوم الحالي (من الكود الخاص بك)
#             current_date = datetime.now().date()
#             day_index = current_date.weekday()
#             current_day_str = DAYS_OF_WEEK[day_index]

#             # 3. جلب جميع حصص اليوم
#             query_session = select(Session).where(Session.date == current_day_str)
#             active_sessions = (await db_session.execute(query_session)).scalars().all()

#             if not active_sessions:
#                 logger.info(f"No active classes scheduled for today ({current_day_str}).")
#                 return

#             # 4. حساب الأوقات وجدولة المهام لكل حصة
#             for cls_session in active_sessions:
#                 # تصفير أو إنشاء سجل الجدولة لليوم الجديد
#                 query_log = select(SchedulerLog).where(SchedulerLog.session_id == cls_session.id)
#                 sched_log = (await db_session.execute(query_log)).scalars().first()

#                 if not sched_log:
#                     sched_log = SchedulerLog(session_id=cls_session.id, scan_count=0,
#                                              scan_session_count=times_per_session)
#                     db_session.add(sched_log)
#                 else:
#                     sched_log.scan_count = 0
#                     sched_log.scan_session_count = times_per_session

#                 # حساب أوقات المسح الدقيقة
#                 # دمج تاريخ اليوم مع وقت الحصة لنحصل على datetime كامل
#                 start_dt = datetime.combine(current_date, cls_session.start_time)
#                 end_dt = datetime.combine(current_date, cls_session.end_time)

#                 duration_sec = (end_dt - start_dt).total_seconds()
#                 buffer_sec = 5 * 60  # اقتطاع 5 دقائق من البداية والنهاية
#                 effective_duration = duration_sec - (2 * buffer_sec)

#                 # تخطي الحصص القصيرة جداً أو المعطوبة
#                 if effective_duration <= 0 or times_per_session <= 0:
#                     continue

#                 # تحديد الفاصل الزمني بين المسحات
#                 if times_per_session == 1:
#                     interval_sec = effective_duration / 2
#                 else:
#                     interval_sec = effective_duration / (times_per_session - 1)

#                 # إنشاء المهام
#                 for i in range(times_per_session):
#                     if times_per_session == 1:
#                         offset = buffer_sec + interval_sec
#                     else:
#                         offset = buffer_sec + (i * interval_sec)

#                     run_time = start_dt + timedelta(seconds=offset)

#                     # نجدول المهمة فقط إذا كان وقتها في المستقبل (تحسباً لتشغيل السيرفر متأخراً)
#                     if run_time > datetime.now():
#                         scheduler.add_job(
#                             execute_ai_scan,
#                             'date',
#                             run_date=run_time,
#                             args=[cls_session.id]
#                         )
#                         logger.info(
#                             f" Scheduled scan {i + 1}/{times_per_session} for Session {cls_session.id} at {run_time.strftime('%H:%M:%S')}")

#             await db_session.commit()
#             logger.warning(" Morning planner completed successfully.")

#         except Exception as e:
#             logger.error(f"Error in morning planner task: {e}")
#             await db_session.rollback()