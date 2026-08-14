from fastapi import APIRouter, Depends, Request, Form
from app.services.attendance_ai import run_attendance_video
from app.services.immediate_scan_service import run_immediate_three_scan_batch
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select, and_,delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from app.core.exceptions import AppException
from app.core.security import get_current_supervisor
from app.database import get_session
from app.models import Supervisor, Classroom, Session
from app.schemas.session import SessionCreate, SessionUpdate
from app.utils.helpers import success_response
from app.services.attendance_scan_service import process_attendance_scan

from app.models.attendance_record import AttendanceRecord
from app.models.scheduler_logs import SchedulerLog
from app.models.student import Student
from app.schemas.attendance_record import ManualAttendanceUpdate

from app.models.classroom import Classroom
from app.services.scan_worker import run_one_scan_job
from app.models.scan_job import ScanJob
import asyncio
import shutil
from pathlib import Path

from fastapi import File, UploadFile

from app.models.videos import Video
router = APIRouter(prefix="/supervisor/manage/session", tags=["Supervisors Manage Session"])


@router.post("/create-session")
async def create_session(
        session_data: SessionCreate,
        db_session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    if session_data.end_time <= session_data.start_time:
        raise AppException(status_code=400, message="End time must be after start time.")

    classroom_query = select(Classroom).where(Classroom.id == session_data.classroom_id)
    classroom_result = await db_session.execute(classroom_query)
    classroom = classroom_result.scalar_one_or_none()

    if not classroom:
        raise AppException(status_code=404, message="Classroom not found.")

    if classroom.supervisor_id != current_supervisor.id:
        raise AppException(
            status_code=403,
            message="You are not authorized to create a session for this classroom."
        )

    overlap_query = select(Session).where(
        and_(
            Session.classroom_id == session_data.classroom_id,
            Session.date == session_data.date,
            Session.name == session_data.name,
            Session.start_time < session_data.end_time,
            Session.end_time > session_data.start_time
        )
    )
    overlap_result = await db_session.execute(overlap_query)
    existing_session = overlap_result.scalar_one_or_none()

    if existing_session:
        raise AppException(
            status_code=409,
            message="A session already exists in this classroom during the specified time."
        )

    new_session = Session(**session_data.model_dump(exclude_unset=True))
    db_session.add(new_session)
    await db_session.commit()
    await db_session.refresh(new_session)

    return success_response(message="Session created successfully", data=new_session)


@router.patch("/update-session/{session_id}")
async def update_session(
        session_id: int,
        session_data: SessionUpdate,
        db_session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    query = (
        select(Session)
        .options(selectinload(Session.classroom))
        .where(Session.id == session_id)
    )
    result = await db_session.execute(query)
    session = result.scalar_one_or_none()

    if not session:
        raise AppException(status_code=404, message="Session not found.")

    if current_supervisor.id != session.classroom.supervisor_id:
        raise AppException(status_code=403, message="You are not authorized to update this classroom.")

    update_class = session_data.model_dump(exclude_unset=True)
    for key, value in update_class.items():
        setattr(session, key, value)

    await db_session.commit()
    await db_session.refresh(session)
    return success_response(message="Session updated successfully", data=jsonable_encoder(session, exclude={"classroom"}))


@router.delete("/delete-session/{session_id}")
async def delete_session(
        session_id: int,
        db_session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    query = select(Session).where(Session.id == session_id)
    result = await db_session.execute(query)
    session = result.scalar_one_or_none()
    if not session:
        raise AppException(status_code=404, message="Session not found.")
    await db_session.delete(session)
    await db_session.commit()
    return success_response(message="Session deleted successfully", data=None)




@router.post("/test-ai-attendance/{session_id}")
async def test_ai_attendance(
        session_id: int,
        request: Request,
        video_path: str = Form(...),
        max_frames: int = Form(300),
        db_session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    try:
        result = await run_attendance_video(
            db=db_session,
            ai=request.app.state.ai,
            session_id=session_id,
            video_path=video_path,
            max_frames=max_frames,
        )

        return success_response(
            message="AI attendance test completed successfully",
            data=result.to_dict(),
        )

    except Exception as e:
        raise AppException(
            status_code=500,
            message=str(e),
        )


@router.post("/test-ai-scan/{session_id}")
async def test_ai_scan(
        session_id: int,
        request: Request,
        video_path: str = Form(...),
        max_frames: int = Form(0),
        db_session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    try:
        result = await process_attendance_scan(
            db=db_session,
            ai=request.app.state.ai,
            session_id=session_id,
            video_path=video_path,
            max_frames=max_frames,
        )

        return success_response(
            message="AI attendance scan completed successfully",
            data={
                "session_id": result["session_id"],
                "scan_number": result["scan_number"],
                "max_scans": result["max_scans"],
                "saved_logs": result["saved_logs"],
                "finalized": result["finalized"],
                "records_written": result["records_written"],
                "ai_result": result["ai_result"].to_dict(),
            },
        )

    except ValueError as e:
        raise AppException(
            status_code=400,
            message=str(e),
        )

    except Exception as e:
        raise AppException(
            status_code=500,
            message=str(e),
        )


@router.post("/run-ai-attendance/{session_id}")
async def run_ai_attendance(
        session_id: int,
        request: Request,
        video_path: str = Form(...),
        max_frames: int = Form(0),
        db_session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    session_result = await db_session.execute(
        select(Session)
        .join(
            Classroom,
            Classroom.id == Session.classroom_id,
        )
        .where(
            Session.id == session_id,
            Classroom.supervisor_id == current_supervisor.id,
        )
    )

    db_session_obj = session_result.scalar_one_or_none()

    if db_session_obj is None:
        raise AppException(
            status_code=404,
            message="Session not found or you are not authorized to access it.",
        )

    try:
        result = await process_attendance_scan(
            db=db_session,
            ai=request.app.state.ai,
            session_id=session_id,
            video_path=video_path,
            max_frames=max_frames,
        )

        ai_data = result["ai_result"].to_dict()

        return success_response(
            message="AI attendance scan completed successfully",
            data={
                "session_id": result["session_id"],
                "scan_number": result["scan_number"],
                "max_scans": result["max_scans"],
                "saved_logs": result["saved_logs"],
                "finalized": result["finalized"],
                "records_written": result["records_written"],
                "present_count": ai_data["present_count"],
                "absent_count": ai_data["absent_count"],
                "present_student_ids": ai_data["present_student_ids"],
                "absent_student_ids": ai_data["absent_student_ids"],
            },
        )

    except ValueError as e:
        raise AppException(
            status_code=400,
            message=str(e),
        )

    except Exception as e:
        raise AppException(
            status_code=500,
            message=str(e),
        )


@router.get("/{session_id}/attendance")
async def get_session_attendance(
        session_id: int,
        db_session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    # ---------------------------------------------------------
    # 1. Verify that the session belongs to this supervisor
    # ---------------------------------------------------------
    session_result = await db_session.execute(
        select(Session)
        .join(
            Classroom,
            Classroom.id == Session.classroom_id,
        )
        .where(
            Session.id == session_id,
            Classroom.supervisor_id == current_supervisor.id,
        )
    )

    session_obj = session_result.scalar_one_or_none()

    if session_obj is None:
        raise AppException(
            status_code=404,
            message="Session not found or you are not authorized to access it.",
        )

    # ---------------------------------------------------------
    # 2. Read scan progress
    # ---------------------------------------------------------
    scheduler_result = await db_session.execute(
        select(SchedulerLog).where(
            SchedulerLog.session_id == session_id
        )
    )

    scheduler_log = scheduler_result.scalars().first()

    if scheduler_log is not None:
        scan_count = int(scheduler_log.scan_count or 0)
        max_scans = int(scheduler_log.scan_session_count or 0)
    else:
        scan_count = 0
        max_scans = 0

    finalized = (
        max_scans > 0
        and scan_count >= max_scans
    )

    # ---------------------------------------------------------
    # 3. Read final attendance records
    # ---------------------------------------------------------
    attendance_result = await db_session.execute(
        select(
            AttendanceRecord,
            Student,
        )
        .join(
            Student,
            Student.id == AttendanceRecord.student_id,
        )
        .where(
            AttendanceRecord.session_id == session_id
        )
        .order_by(Student.id)
    )

    rows = attendance_result.all()

    students = []

    for record, student in rows:
        status_value = getattr(
            record.status,
            "value",
            record.status,
        )

        students.append(
            {
                "student_id": int(student.id),
                "full_name": f"{student.first_name} {student.last_name}",
                "status": str(status_value).lower(),
                "confidence_score": (
                    float(record.confidence_score)
                    if record.confidence_score is not None
                    else None
                ),
                "manually_modified": bool(
                    record.manually_modified
                ),
                "notes": record.notes,
            }
        )

    return success_response(
        message="Session attendance retrieved successfully",
        data={
            "session_id": session_id,
            "scan_count": scan_count,
            "max_scans": max_scans,
            "finalized": finalized,
            "students_count": len(students),
            "students": students,
        },
    )





@router.patch("/{session_id}/attendance/{student_id}")
async def update_attendance_manually(
        session_id: int,
        student_id: int,
        attendance_data: ManualAttendanceUpdate,
        db_session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    # ---------------------------------------------------------
    # 1. Verify session ownership
    # ---------------------------------------------------------
    session_result = await db_session.execute(
        select(Session)
        .join(
            Classroom,
            Classroom.id == Session.classroom_id,
        )
        .where(
            Session.id == session_id,
            Classroom.supervisor_id == current_supervisor.id,
        )
    )

    session_obj = session_result.scalar_one_or_none()

    if session_obj is None:
        raise AppException(
            status_code=404,
            message="Session not found or you are not authorized to access it.",
        )

    # ---------------------------------------------------------
    # 2. Verify student belongs to this session's classroom
    # ---------------------------------------------------------
    student_result = await db_session.execute(
        select(Student).where(
            Student.id == student_id,
            Student.classroom_id == session_obj.classroom_id,
        )
    )

    student = student_result.scalar_one_or_none()

    if student is None:
        raise AppException(
            status_code=404,
            message="Student not found in this session's classroom.",
        )

    # ---------------------------------------------------------
    # 3. Find final attendance record
    # ---------------------------------------------------------
    record_result = await db_session.execute(
        select(AttendanceRecord).where(
            AttendanceRecord.session_id == session_id,
            AttendanceRecord.student_id == student_id,
        )
    )

    record = record_result.scalar_one_or_none()

    try:
        # -----------------------------------------------------
        # 4. Update existing record or create manual record
        # -----------------------------------------------------
        if record is not None:
            record.status = attendance_data.status
            record.manually_modified = True
            record.notes = attendance_data.notes

        else:
            record = AttendanceRecord(
                session_id=session_id,
                student_id=student_id,
                supervisor_id=current_supervisor.id,
                status=attendance_data.status,
                confidence_score=None,
                manually_modified=True,
                notes=attendance_data.notes,
            )

            db_session.add(record)

        await db_session.commit()
        await db_session.refresh(record)

        status_value = getattr(
            record.status,
            "value",
            record.status,
        )

        return success_response(
            message="Attendance manually updated successfully",
            data={
                "session_id": session_id,
                "student_id": student_id,
                "full_name": f"{student.first_name} {student.last_name}",
                "status": str(status_value).lower(),
                "confidence_score": (
                    float(record.confidence_score)
                    if record.confidence_score is not None
                    else None
                ),
                "manually_modified": bool(record.manually_modified),
                "notes": record.notes,
            },
        )


    except Exception as e:
        await db_session.rollback()

        raise AppException(
            status_code=500,
            message=f"Failed to update attendance manually: {str(e)}",
        )

@router.post("/scheduler/run-next-scan")
async def run_next_queued_scan(
    request: Request,
    session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    allowed_result = await session.execute(
        select(Session.id)
        .join(
            Classroom,
            Classroom.id == Session.classroom_id,
        )
        .where(
            Classroom.supervisor_id
            == current_supervisor.id
        )
    )

    allowed_session_ids = list(
        allowed_result.scalars().all()
    )

    result = await run_one_scan_job(
        ai=request.app.state.ai,
        allowed_session_ids=allowed_session_ids,
    )

    return success_response(
        message="Queued scan worker executed",
        data=result,
    )

@router.get("/scheduler/scan-queue")
async def get_scan_queue(
    session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    result = await session.execute(
        select(
            ScanJob,
            Session,
        )
        .join(
            Session,
            Session.id == ScanJob.session_id,
        )
        .join(
            Classroom,
            Classroom.id == Session.classroom_id,
        )
        .where(
            Classroom.supervisor_id
            == current_supervisor.id
        )
        .order_by(
            ScanJob.priority.asc(),
            Session.end_time.asc(),
            ScanJob.queued_at.asc().nulls_last(),
            ScanJob.id.asc(),
        )
    )

    rows = result.all()

    jobs = []

    for job, session_obj in rows:
        jobs.append(
            {
                "job_id": job.id,
                "session_id": job.session_id,
                "session_name": session_obj.name,
                "scan_number": job.scan_number,
                "priority": job.priority,
                "status": job.status,
                "attempts": job.attempts,
                "segment_start_seconds": (
                    job.segment_start_seconds
                ),
                "segment_duration_seconds": (
                    job.segment_duration_seconds
                ),
                "eligible_at": job.eligible_at,
                "queued_at": job.queued_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "error_message": job.error_message,
            }
        )

    return success_response(
        message="Scan queue available",
        data=jobs,
    )



@router.post("/{session_id}/video")
async def upload_session_video(
    session_id: int,
    video_file: UploadFile = File(...),
    db_session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    session_result = await db_session.execute(
        select(Session)
        .join(
            Classroom,
            Classroom.id == Session.classroom_id,
        )
        .where(
            Session.id == session_id,
            Classroom.supervisor_id == current_supervisor.id,
        )
    )

    session_obj = session_result.scalar_one_or_none()

    if session_obj is None:
        raise AppException(
            status_code=404,
            message="Session not found or not authorized.",
        )

    original_name = video_file.filename or ""
    suffix = Path(original_name).suffix.lower()

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }

    if suffix not in allowed_extensions:
        raise AppException(
            status_code=400,
            message="Unsupported video format.",
        )

    storage_dir = (
        Path.cwd()
        / "data"
        / "managed"
        / "session_videos"
    ).resolve()

    storage_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_path = (
        storage_dir
        / f"session_{session_id}{suffix}"
    )

    temp_path = (
        storage_dir
        / f".session_{session_id}{suffix}.uploading"
    )

    try:
        await video_file.seek(0)

        def copy_video():
            with temp_path.open("wb") as destination:
                shutil.copyfileobj(
                    video_file.file,
                    destination,
                    length=1024 * 1024,
                )

            temp_path.replace(target_path)

        await asyncio.to_thread(
            copy_video
        )

        if (
            not target_path.is_file()
            or target_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Uploaded video is empty."
            )

        video_result = await db_session.execute(
            select(Video)
            .where(
                Video.session_id == session_id
            )
            .with_for_update()
        )

        existing_video = (
            video_result.scalars().first()
        )

        if existing_video is None:
            existing_video = Video(
                session_id=session_id,
                path=str(target_path),
            )

            db_session.add(
                existing_video
            )

        else:
            existing_video.path = str(
                target_path
            )

        await db_session.commit()
        await db_session.refresh(
            existing_video
        )

        return success_response(
            message="Session video uploaded successfully",
            data={
                "video_id": existing_video.id,
                "session_id": session_id,
                "filename": target_path.name,
                "size_bytes": target_path.stat().st_size,
            },
        )

    except AppException:
        await db_session.rollback()
        raise

    except Exception as exc:
        await db_session.rollback()

        if temp_path.exists():
            temp_path.unlink()

        raise AppException(
            status_code=500,
            message=f"Video upload failed: {type(exc).__name__}: {exc}",
        )

    finally:
        await video_file.close()



@router.get("/{session_id}/video")
async def get_session_video(
    session_id: int,
    db_session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    result = await db_session.execute(
        select(
            Video,
            Session,
        )
        .join(
            Session,
            Session.id == Video.session_id,
        )
        .join(
            Classroom,
            Classroom.id == Session.classroom_id,
        )
        .where(
            Session.id == session_id,
            Classroom.supervisor_id == current_supervisor.id,
        )
    )

    row = result.first()

    if row is None:
        raise AppException(
            status_code=404,
            message="No video is registered for this session.",
        )

    video, session_obj = row

    path = Path(video.path)

    return success_response(
        message="Session video available",
        data={
            "video_id": video.id,
            "session_id": session_obj.id,
            "session_name": session_obj.name,
            "filename": path.name,
            "file_exists": path.is_file(),
            "size_bytes": (
                path.stat().st_size
                if path.is_file()
                else None
            ),
        },
    )


@router.post("/{session_id}/run-3-scans-now")
async def run_three_scans_now(
    session_id: int,
    request: Request,
    db_session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    allowed_result = await db_session.execute(
        select(Session.id)
        .join(
            Classroom,
            Classroom.id == Session.classroom_id,
        )
        .where(
            Session.id == session_id,
            Classroom.supervisor_id
            == current_supervisor.id,
        )
    )

    allowed_session_id = (
        allowed_result.scalars().first()
    )

    if allowed_session_id is None:
        raise AppException(
            status_code=404,
            message="Session not found or not authorized.",
        )

    try:
        result = await run_immediate_three_scan_batch(
            ai=request.app.state.ai,
            session_id=session_id,
        )

    except FileNotFoundError as exc:
        raise AppException(
            status_code=404,
            message=str(exc),
        )

    except ValueError as exc:
        raise AppException(
            status_code=409,
            message=str(exc),
        )

    except RuntimeError as exc:
        raise AppException(
            status_code=409,
            message=str(exc),
        )

    return success_response(
        message="Immediate 3-scan attendance completed",
        data=result,
    )