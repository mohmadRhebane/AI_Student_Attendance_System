from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.enums import AttendanceStatus
from app.models.attendance_logs import AttendanceLogs
from app.models.attendance_record import AttendanceRecord
from app.models.classroom import Classroom
from app.models.session import Session
from app.models.student import Student


from datetime import date


async def save_scan_logs(
    db: AsyncSession,
    session_id: int,
    attendance_date: date,
    scan_number: int,
    ai_result,
    min_confidence_score: float = 0.0,
) -> int:

    if scan_number < 1:
        raise ValueError(
            "scan_number must be >= 1."
        )

    if not ai_result.succeeded:
        raise RuntimeError(
            f"AI scan failed: "
            f"{ai_result.error_message}"
        )

    db_session = await db.get(
        Session,
        session_id,
    )

    if not db_session:
        raise ValueError(
            "Session not found."
        )

    roster_result = await db.execute(
        select(Student.id).where(
            Student.classroom_id
            == db_session.classroom_id,

            Student.is_active.is_(True),
        )
    )

    roster_ids = {
        int(student_id)
        for student_id
        in roster_result.scalars().all()
    }

    # إعادة Scan نفسها لا تصنع duplicate
    await db.execute(
        delete(AttendanceLogs).where(
            AttendanceLogs.session_id
            == session_id,

            AttendanceLogs.attendance_date
            == attendance_date,

            AttendanceLogs.scan_number
            == scan_number,
        )
    )

    inserted = 0

    for item in ai_result.attendance:

        status_value = getattr(
            item.status,
            "value",
            item.status,
        )

        if (
            str(status_value).upper()
            != "PRESENT"
        ):
            continue

        student_id = int(
            item.student_id
        )

        if student_id not in roster_ids:
            continue

        confidence_score = (
            float(item.best_score)
            if item.best_score is not None
            else None
        )

        # ======================================
        # Database confidence threshold
        # ======================================

        if confidence_score is None:
            continue

        if (
            confidence_score
            < float(min_confidence_score)
        ):
            continue

        detection_count = max(
            1,
            int(
                item.observation_count
                or 1
            ),
        )

        log = AttendanceLogs(
            session_id=session_id,
            attendance_date=attendance_date,
            student_id=student_id,
            scan_number=scan_number,
            confidence_score=confidence_score,
            detection_count=detection_count,
        )

        db.add(log)

        inserted += 1

    await db.flush()

    return inserted



async def finalize_attendance(
    db: AsyncSession,
    session_id: int,
    attendance_date: date,
) -> int:
    """
    Build the FINAL attendance_records from all successful scan logs.

    Call this only AFTER the last scan of the session.
    """

    db_session = await db.get(Session, session_id)

    if not db_session:
        raise ValueError("Session not found.")

    classroom = await db.get(
        Classroom,
        db_session.classroom_id,
    )

    if not classroom:
        raise ValueError("Classroom not found.")

    students_result = await db.execute(
        select(Student).where(
            Student.classroom_id == db_session.classroom_id,
            Student.is_active.is_(True),
        )
    )

    students = students_result.scalars().all()
    aggregate_result = await db.execute(
    select(
        AttendanceLogs.student_id,

        func.max(
            AttendanceLogs.confidence_score
        ).label("best_confidence"),

        func.sum(
            AttendanceLogs.detection_count
        ).label("total_detections"),

        func.count(
            func.distinct(
                AttendanceLogs.scan_number
            )
        ).label("present_scans"),
    )
    .where(
        AttendanceLogs.session_id
        == session_id,

        AttendanceLogs.attendance_date
        == attendance_date,
    )
    .group_by(
        AttendanceLogs.student_id
    )
)
    evidence = {
        int(row.student_id): row
        for row in aggregate_result.all()
    }

    existing_result = await db.execute(
        select(AttendanceRecord).where(
            AttendanceRecord.session_id
            == session_id,

            AttendanceRecord.attendance_date
            == attendance_date,
        )
    )

    existing_records = {
        int(record.student_id): record
        for record in existing_result.scalars().all()
    }

    written = 0

    for student in students:
        student_evidence = evidence.get(student.id)

        if student_evidence is not None:
            final_status = AttendanceStatus.PRESENT

            final_confidence = (
                float(student_evidence.best_confidence)
                if student_evidence.best_confidence is not None
                else None
            )

            notes = (
                f"AI attendance: detected in "
                f"{student_evidence.present_scans} scan(s), "
                f"total observations="
                f"{student_evidence.total_detections}"
            )

        else:
            final_status = AttendanceStatus.ABSENT
            final_confidence = None

            notes = (
                "AI attendance: no valid presence "
                "detected in any successful scan."
            )

        record = existing_records.get(student.id)

        if record is not None:
            # تعديل المشرف اليدوي له أولوية على AI.
            if record.manually_modified:
                continue

            record.status = final_status
            record.confidence_score = final_confidence
            record.notes = notes

        else:
            record = AttendanceRecord(
                session_id=session_id,
                attendance_date=attendance_date,
                student_id=student.id,
                supervisor_id=classroom.supervisor_id,
                status=final_status,
                confidence_score=final_confidence,
                manually_modified=False,
                notes=notes,
)

            db.add(record)

        written += 1

    await db.flush()

    return written