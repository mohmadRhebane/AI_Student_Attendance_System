import asyncio
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Student
from app.models.session import Session

from smart_attendance_ai.ai_contracts import AttendanceSessionAIRequest


async def run_attendance_video(
    db: AsyncSession,
    ai,
    session_id: int,
    video_path: str,
    max_frames: int = 300,
):
    db_session = await db.get(Session, session_id)

    if not db_session:
        raise ValueError("Session not found.")

    path = Path(video_path).resolve()

    if not path.is_file():
        raise ValueError(f"Video not found: {path}")

    result = await db.execute(
        select(Student).where(
            Student.classroom_id == db_session.classroom_id,
            Student.is_active.is_(True),
        )
    )

    students = result.scalars().all()

    if not students:
        raise ValueError("No active students in this classroom.")

    ai_session_id = f"DB_{db_session.id}_{uuid.uuid4().hex[:8]}"

    ai_request = AttendanceSessionAIRequest(
        session_id=ai_session_id,
        class_id=str(db_session.classroom_id),
        source=str(path),
        roster_student_ids=[str(student.id) for student in students],
        max_frames=max_frames,
        metadata={
            "db_session_id": db_session.id,
        },
    )

    snapshot = await asyncio.to_thread(
        ai.start_attendance,
        ai_request,
    )

    await asyncio.to_thread(
        ai.wait_for_attendance,
        snapshot.session_id,
    )

    attendance_result = await asyncio.to_thread(
        ai.get_attendance_result,
        snapshot.session_id,
    )

    if attendance_result is None:
        raise RuntimeError("AI attendance result is unavailable.")

    return attendance_result