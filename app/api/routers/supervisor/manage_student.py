import asyncio
import re
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import ValidationError
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppException
from app.core.logger import logger
from app.core.security import get_current_supervisor
from app.database import get_session
from app.models import Classroom, FaceEmbedding, Student, Supervisor
from app.schemas.student import StudentCreate, StudentResponse, StudentUpdate
from app.services.gallery_sync import build_gallery_from_db
from app.utils.helpers import success_response


router = APIRouter(
    prefix="/supervisor/manage/student",
    tags=["Students Management"],
)


@router.get("/get-students",response_model=StudentResponse)
async def get_students(
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):
    result = await session.execute(select(Student).offset(0).limit(100))
    students = result.scalars().all()
    if not students:
        raise AppException(status_code=404,message="There are no students")
    return success_response(message="Students retrieved successfully", data=students)

@router.get("/get-classroom-students/{classroom_id}")
async def get_classroom_students(
        classroom_id: int,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):
    classroom_query = select(Classroom).where(Classroom.id == classroom_id)
    classroom_result = await session.execute(classroom_query)
    classroom = classroom_result.scalar_one_or_none()

    if not classroom:
        raise AppException(status_code=404, message="Classroom not found.")

    if classroom.supervisor_id != current_supervisor.id:
        raise AppException(
            status_code=403,
            message="You are not authorized to view students in this classroom."
        )

    students_query = select(Student).where(Student.classroom_id == classroom_id)
    students_result = await session.execute(students_query)
    students = students_result.scalars().all()
    if not students:
        raise AppException(status_code=404,message="There are no students")
    return success_response(message="Students retrieved successfully", data=students)

@router.get("/get-student/{classroom_id}")
async def get_student(
        classroom_id: int,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):
    query = select(Student).where(Student.classroom_id == classroom_id)
    result = await session.execute(query)
    student = result.scalar_one_or_none()
    if not student:
        raise AppException(status_code=404, message="Student not found.")
    return success_response(message="Student retrieved successfully", data=student)


@router.get("/search-for-student/{name}")
async def search_for_student(
        name: str,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    search_term = prepare_arabic_search(name)
    query = select(Student).where(
        or_(
            Student.first_name.ilike(search_term),
            Student.last_name.ilike(search_term)
        )
    )
    result = await session.execute(query)
    students = result.scalars().all()
    if not students:
        raise AppException(status_code=404, message="There are no students")
    return success_response(message="Students retrieved successfully", data=students)

def prepare_arabic_search(text: str) -> str:
    text = re.sub(r'[أإآا]', '_', text)
    text = re.sub(r'[ةه]', '_', text)
    return f"%{text}%"

import shutil
import uuid
from pathlib import Path


@router.post("/create-student")
async def create_student(
        request: Request,
        first_name: str = Form(...),
        last_name: str = Form(...),
        father_name: str = Form(...),
        classroom_id: int = Form(...),
        grade: str = Form(...),
        section: str = Form(...),
        student_images: List[UploadFile] = File(...),
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    try:
        validated_data = StudentCreate(
            first_name=first_name,
            last_name=last_name,
            father_name=father_name,
            classroom_id=classroom_id,
            grade=grade,
            section=section,
        )

        classroom = (
            await session.execute(
                select(Classroom).where(Classroom.id == classroom_id)
            )
        ).scalar_one_or_none()

        if not classroom:
            raise AppException(status_code=404, message="Classroom not found.")

        if classroom.supervisor_id != current_supervisor.id:
            raise AppException(
                status_code=403,
                message="You are not authorized to add students to this classroom."
            )

        duplicate_query = select(Student).where(
            and_(
                Student.first_name == first_name,
                Student.last_name == last_name,
                Student.father_name == father_name,
                Student.classroom_id == classroom_id,
                Student.grade == grade,
                Student.section == section,
            )
        )

        existing = (
            await session.execute(duplicate_query)
        ).scalar_one_or_none()

        if existing:
            raise AppException(
                status_code=409,
                message="Student already exists in this classroom"
            )

        student_dict = validated_data.model_dump(exclude_unset=True)
        student_dict.pop("face_embeddings", None)

        new_student = Student(**student_dict)
        session.add(new_student)

        await session.flush()

        ai = request.app.state.ai

        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []

            for image_file in student_images:
                suffix = Path(image_file.filename or "image.jpg").suffix or ".jpg"
                path = Path(temp_dir) / f"{uuid.uuid4().hex}{suffix}"

                contents = await image_file.read()
                path.write_bytes(contents)
                image_paths.append(path)

            enrollment = await asyncio.to_thread(
                ai.prepare_student_enrollment,
                student_id=str(new_student.id),
                full_name=f"{first_name} {last_name}".strip(),
                image_paths=image_paths,
                mode="NEW",
            )

        if not enrollment.ready:
            raise AppException(
                status_code=422,
                message=f"AI enrollment rejected: {enrollment.status}"
            )

        for embedding in enrollment.embeddings:
            session.add(
                FaceEmbedding(
                    student_id=new_student.id,
                    embedding=embedding.tolist(),
                )
            )

        await session.flush()

        gallery = await build_gallery_from_db(
            session=session,
            ai=ai,
            activate_after_build=False,
        )

        await session.commit()

        await asyncio.to_thread(
            ai.activate_gallery_version,
            gallery.version_id,
        )

        await session.refresh(new_student)

        return success_response(
            message="Student enrolled successfully",
            data=new_student,
        )

    except AppException:
        await session.rollback()
        raise

    except ValidationError as e:
        await session.rollback()
        raise AppException(status_code=422, message=e.errors())
    except Exception as e:
        await session.rollback()
        logger.exception("Error creating student")
        raise AppException(
            status_code=500,
            message=f"Student enrollment failed: {type(e).__name__}: {str(e)}"
        )
    # except Exception as e:
    #     await session.rollback()
    #     logger.warning(f"Error creating student: {e}")
    #     raise AppException(
    #         status_code=500,
    #         message="Student enrollment failed."
    #     )
    



@router.patch("/update-student/{student_id}")
async def update_student(
        student_id: int,
        student_data: StudentUpdate,
        request: Request,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    student_query = (
        select(Student)
        .join(
            Classroom,
            Classroom.id == Student.classroom_id,
        )
        .where(
            Student.id == student_id,
            Classroom.supervisor_id == current_supervisor.id,
        )
    )

    student_db = (
        await session.execute(student_query)
    ).scalar_one_or_none()

    if student_db is None:
        raise AppException(
            status_code=404,
            message=(
                "Student not found or you are not authorized "
                "to modify this student."
            ),
        )

    update_dict = student_data.model_dump(
        exclude_unset=True
    )

    update_dict.pop(
        "face_embeddings",
        None,
    )

    if not update_dict:
        raise AppException(
            status_code=400,
            message="No fields were provided for update.",
        )

    # -----------------------------------------
    # إذا تغير الصف، تحقق أنه تابع للمشرف نفسه.
    # -----------------------------------------
    new_classroom_id = update_dict.get(
        "classroom_id"
    )

    if (
        new_classroom_id is not None
        and new_classroom_id != student_db.classroom_id
    ):
        classroom_result = await session.execute(
            select(Classroom).where(
                Classroom.id == new_classroom_id,
                Classroom.supervisor_id == current_supervisor.id,
            )
        )

        new_classroom = (
            classroom_result.scalar_one_or_none()
        )

        if new_classroom is None:
            raise AppException(
                status_code=403,
                message=(
                    "You are not authorized to move "
                    "the student to this classroom."
                ),
            )

    # -----------------------------------------
    # منع إنشاء طالب مكرر بسبب عملية Update.
    # نستخدم القيم الجديدة إن وجدت وإلا الحالية.
    # -----------------------------------------
    effective_first_name = update_dict.get(
        "first_name",
        student_db.first_name,
    )

    effective_last_name = update_dict.get(
        "last_name",
        student_db.last_name,
    )

    effective_father_name = update_dict.get(
        "father_name",
        student_db.father_name,
    )

    effective_classroom_id = update_dict.get(
        "classroom_id",
        student_db.classroom_id,
    )

    effective_grade = update_dict.get(
        "grade",
        student_db.grade,
    )

    effective_section = update_dict.get(
        "section",
        student_db.section,
    )

    duplicate_result = await session.execute(
        select(Student).where(
            Student.id != student_id,
            Student.first_name == effective_first_name,
            Student.last_name == effective_last_name,
            Student.father_name == effective_father_name,
            Student.classroom_id == effective_classroom_id,
            Student.grade == effective_grade,
            Student.section == effective_section,
        )
    )

    if duplicate_result.scalar_one_or_none() is not None:
        raise AppException(
            status_code=409,
            message="Another student with the same data already exists.",
        )

    # -----------------------------------------
    # هذه الحقول تؤثر على Gallery.
    #
    # مهم:
    # نستخدم "هل تم إرسال الحقل؟"
    # وليس فقط "هل تغيرت القيمة؟"
    #
    # هذا يسمح لنا بإصلاح Gallery قديمة حتى
    # لو كان الاسم في PostgreSQL صحيحًا أصلًا.
    # -----------------------------------------
    gallery_identity_fields = {
        "first_name",
        "last_name",
        "is_active",
    }

    gallery_rebuild_required = any(
        field in update_dict
        for field in gallery_identity_fields
    )

    try:
        for key, value in update_dict.items():
            setattr(
                student_db,
                key,
                value,
            )

        # نجعل التغييرات مرئية لنفس transaction
        # قبل إعادة بناء Gallery.
        await session.flush()

        gallery_result = None

        if gallery_rebuild_required:
            gallery_result = await build_gallery_from_db(
                session=session,
                ai=request.app.state.ai,
                activate_after_build=False,
            )

        # PostgreSQL هو Source of Truth.
        await session.commit()

        # فعّل Gallery الجديدة بعد نجاح DB commit.
        if gallery_result is not None:
            await asyncio.to_thread(
                request.app.state.ai.activate_gallery_version,
                gallery_result.version_id,
            )

        await session.refresh(
            student_db
        )

        return success_response(
            message="Student updated successfully",
            data={
                "id": student_db.id,
                "first_name": student_db.first_name,
                "last_name": student_db.last_name,
                "father_name": student_db.father_name,
                "classroom_id": student_db.classroom_id,
                "grade": student_db.grade,
                "section": student_db.section,
                "is_active": student_db.is_active,
                "gallery_rebuilt": (
                    gallery_result is not None
                ),
                "gallery_version_id": (
                    gallery_result.version_id
                    if gallery_result is not None
                    else None
                ),
            },
        )

    except AppException:
        await session.rollback()
        raise

    except Exception as e:
        await session.rollback()

        logger.error(
            f"Error updating student {student_id}: {e}"
        )

        raise AppException(
            status_code=500,
            message=f"Failed to update student: {str(e)}",
        )


@router.delete("/delete-student/{student_id}")
async def delete_student(
        student_id: int,
        request: Request,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    # ---------------------------------------------------------
    # 1. Get student and verify supervisor ownership
    # ---------------------------------------------------------
    student_query = (
        select(Student)
        .join(
            Classroom,
            Classroom.id == Student.classroom_id,
        )
        .where(
            Student.id == student_id,
            Classroom.supervisor_id == current_supervisor.id,
        )
    )

    student = (
        await session.execute(student_query)
    ).scalar_one_or_none()

    if student is None:
        raise AppException(
            status_code=404,
            message=(
                "Student not found or you are not authorized "
                "to deactivate this student."
            ),
        )

    # ---------------------------------------------------------
    # 2. Idempotency:
    # already inactive = do nothing
    # ---------------------------------------------------------
    if not student.is_active:
        return success_response(
            message="Student is already inactive",
            data={
                "student_id": student.id,
                "is_active": False,
                "gallery_rebuilt": False,
                "gallery_version_id": None,
            },
        )

    try:
        # -----------------------------------------------------
        # 3. Soft delete
        # -----------------------------------------------------
        student.is_active = False

        # Make the new state visible to this DB transaction.
        await session.flush()

        # -----------------------------------------------------
        # 4. Rebuild Gallery from ACTIVE DB students only
        # -----------------------------------------------------
        gallery = await build_gallery_from_db(
            session=session,
            ai=request.app.state.ai,
            activate_after_build=False,
        )

        # -----------------------------------------------------
        # 5. PostgreSQL is the source of truth
        # -----------------------------------------------------
        await session.commit()

        # -----------------------------------------------------
        # 6. Activate the newly built Gallery
        # -----------------------------------------------------
        await asyncio.to_thread(
            request.app.state.ai.activate_gallery_version,
            gallery.version_id,
        )

        await session.refresh(student)

        return success_response(
            message="Student deactivated successfully",
            data={
                "student_id": student.id,
                "full_name": (
                    f"{student.first_name} "
                    f"{student.last_name}"
                ).strip(),
                "is_active": student.is_active,
                "gallery_rebuilt": True,
                "gallery_version_id": gallery.version_id,
            },
        )

    except AppException:
        await session.rollback()
        raise

    except Exception as e:
        await session.rollback()

        logger.error(
            f"Error deactivating student {student_id}: {e}"
        )

        raise AppException(
            status_code=500,
            message=f"Failed to deactivate student: {str(e)}",
        )

@router.post("/add-embeddings/{student_id}")
async def add_embeddings(
        student_id: int,
        request: Request,
        student_images: List[UploadFile] = File(...),
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    # ---------------------------------------------------------
    # 1. Get student and verify supervisor ownership
    # ---------------------------------------------------------
    student_query = (
        select(Student)
        .join(
            Classroom,
            Classroom.id == Student.classroom_id,
        )
        .where(
            Student.id == student_id,
            Classroom.supervisor_id == current_supervisor.id,
        )
    )

    student = (
        await session.execute(student_query)
    ).scalar_one_or_none()

    if student is None:
        raise AppException(
            status_code=404,
            message=(
                "Student not found or you are not authorized "
                "to modify this student."
            ),
        )

    # ---------------------------------------------------------
    # 2. UPDATE enrollment requires an active Gallery student
    # ---------------------------------------------------------
    if not student.is_active:
        raise AppException(
            status_code=409,
            message=(
                "Cannot add face embeddings to an inactive student. "
                "Reactivate the student first."
            ),
        )

    if not student_images:
        raise AppException(
            status_code=422,
            message="At least one student image is required.",
        )

    ai = request.app.state.ai

    full_name = (
        f"{student.first_name} {student.last_name}"
    ).strip()

    try:
        # -----------------------------------------------------
        # 3. Store uploaded images temporarily
        # -----------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []

            for image_file in student_images:
                suffix = (
                    Path(
                        image_file.filename or "image.jpg"
                    ).suffix
                    or ".jpg"
                )

                image_path = (
                    Path(temp_dir)
                    / f"{uuid.uuid4().hex}{suffix}"
                )

                contents = await image_file.read()

                if not contents:
                    raise AppException(
                        status_code=422,
                        message=(
                            f"Uploaded image is empty: "
                            f"{image_file.filename}"
                        ),
                    )

                image_path.write_bytes(contents)
                image_paths.append(image_path)

            # -------------------------------------------------
            # 4. Real ArcFace enrollment in UPDATE mode
            # -------------------------------------------------
            enrollment = await asyncio.to_thread(
                ai.prepare_student_enrollment,
                student_id=str(student.id),
                full_name=full_name,
                image_paths=image_paths,
                mode="UPDATE",
                minimum_accepted_images=1,
            )

        # -----------------------------------------------------
        # 5. Normalize enrollment status
        # -----------------------------------------------------
        enrollment_status = str(
            getattr(
                enrollment.status,
                "value",
                enrollment.status,
            )
        ).upper()

        # -----------------------------------------------------
        # 6. Do NOT automatically accept review cases
        # -----------------------------------------------------
        if enrollment_status == "READY_FOR_REVIEW":
            identity_check = enrollment.identity_check

            if hasattr(identity_check, "to_dict"):
                identity_data = identity_check.to_dict()
            elif isinstance(identity_check, dict):
                identity_data = identity_check
            else:
                identity_data = {}

            raise AppException(
                status_code=409,
                message={
                    "message": (
                        "The submitted images require manual review "
                        "and were not added."
                    ),
                    "status": enrollment_status,
                    "identity_check": identity_data,
                    "warnings": list(enrollment.warnings or []),
                },
            )

        if enrollment_status != "READY":
            raise AppException(
                status_code=422,
                message=(
                    f"AI enrollment was not accepted. "
                    f"Status: {enrollment_status}"
                ),
            )

        if int(enrollment.accepted_images) < 1:
            raise AppException(
                status_code=422,
                message="No valid face image was accepted.",
            )

        # -----------------------------------------------------
        # 7. Save REAL embeddings in PostgreSQL
        # -----------------------------------------------------
        added_embeddings = 0

        for embedding in enrollment.embeddings:
            embedding_list = embedding.tolist()

            if len(embedding_list) != 512:
                raise RuntimeError(
                    "AI returned an invalid embedding dimension: "
                    f"{len(embedding_list)}"
                )

            session.add(
                FaceEmbedding(
                    student_id=student.id,
                    embedding=embedding_list,
                )
            )

            added_embeddings += 1

        if added_embeddings == 0:
            raise AppException(
                status_code=422,
                message="AI did not return any usable embeddings.",
            )

        await session.flush()

        # -----------------------------------------------------
        # 8. Rebuild Gallery from PostgreSQL
        # -----------------------------------------------------
        gallery = await build_gallery_from_db(
            session=session,
            ai=ai,
            activate_after_build=False,
        )

        # -----------------------------------------------------
        # 9. PostgreSQL remains Source of Truth
        # -----------------------------------------------------
        await session.commit()

        # -----------------------------------------------------
        # 10. Activate new Gallery
        # -----------------------------------------------------
        await asyncio.to_thread(
            ai.activate_gallery_version,
            gallery.version_id,
        )

        # -----------------------------------------------------
        # 11. Prepare useful image results for Frontend
        # -----------------------------------------------------
        image_results = []

        for item in enrollment.image_results:
            image_results.append(
                {
                    "filename": item.original_filename,
                    "accepted": bool(item.accepted),
                    "reason": item.reason,
                    "detection_score": item.detection_score,
                    "blur_score": item.blur_score,
                    "brightness": item.brightness,
                    "contrast": item.contrast,
                }
            )

        identity_check = enrollment.identity_check

        if hasattr(identity_check, "to_dict"):
            identity_data = identity_check.to_dict()
        elif isinstance(identity_check, dict):
            identity_data = identity_check
        else:
            identity_data = {}

        return success_response(
            message="Student face embeddings added successfully",
            data={
                "student_id": student.id,
                "full_name": full_name,
                "enrollment_status": enrollment_status,
                "submitted_images": int(
                    enrollment.submitted_images
                ),
                "accepted_images": int(
                    enrollment.accepted_images
                ),
                "rejected_images": int(
                    enrollment.rejected_images
                ),
                "added_embeddings": added_embeddings,
                "embedding_dimension": 512,
                "identity_check": identity_data,
                "warnings": list(
                    enrollment.warnings or []
                ),
                "images": image_results,
                "gallery_rebuilt": True,
                "gallery_version_id": gallery.version_id,
            },
        )

    except AppException:
        await session.rollback()
        raise

    except Exception as e:
        await session.rollback()

        logger.exception(
            f"Error adding embeddings for student {student_id}"
        )

        raise AppException(
            status_code=500,
            message=(
                "Failed to add student embeddings: "
                f"{type(e).__name__}: {str(e)}"
            ),
        )
    

# @router.post("/attendance-student-manually/{student_id}")
# async def attendance_student_manually(
#         student_id: int,
#         attendance_data: AttendanceRecordCreate,
#         session: AsyncSession = Depends(get_session),
#         current_supervisor: Supervisor = Depends(get_current_supervisor),
# ):
#     today = date.today()
#     start_of_day = datetime.combine(today, time.min)  # 00:00:00
#     end_of_day = datetime.combine(today, time.max)  # 23:59:59

#     query = select(Student).join(Classroom).where(
#         Student.id == student_id,
#         Classroom.supervisor_id == current_supervisor.id
#     )
#     student_result = await session.execute(query)
#     if not student_result.scalar_one_or_none():
#         raise AppException(status_code=403, message="You are not authorized to view/edit students in this classroom.")

#     query_attendance = select(AttendanceRecord).where(
#         and_(
#             AttendanceRecord.student_id == student_id,
#             AttendanceRecord.created_at >= start_of_day,
#             AttendanceRecord.created_at <= end_of_day
#         )
#     )
#     attendance_result = await session.execute(query_attendance)
#     attendance_record = attendance_result.scalar_one_or_none()

#     message_attendance = ""
#     try:
#         if not attendance_record:
#             new_attendance = AttendanceRecord(
#                 student_id=student_id,
#                 supervisor_id=current_supervisor.id,
#                 status=attendance_data.status,
#                 manually_modified=True,
#                 notes=attendance_data.notes
#             )
#             session.add(new_attendance)
#             message_attendance = "Student attendance recorded successfully (New Record)."
#         else:
#             attendance_record.status = str(attendance_data.status)
#             attendance_record.manually_modified = True
#             attendance_record.notes = attendance_data.notes
#             message_attendance = "Student attendance updated successfully (Corrected)."

#         await session.commit()
#         return success_response(message=message_attendance, data=None)

#     except Exception as e:
#         await session.rollback()
#         logger.error(f"Error saving attendance manually: {e}")
#         raise AppException(status_code=500, message="An error occurred while saving the attendance. Transaction rolled back.")

