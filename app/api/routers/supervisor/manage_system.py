from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from sqlalchemy.orm import joinedload

from app.core.exceptions import AppException
from app.core.logger import logger

from app.core.security import get_current_supervisor
from app.database import get_session
from app.models.classroom import Classroom
from app.models.supervisor import Supervisor
from app.models.system_setting import SystemSetting
from app.schemas.classroom import ClassroomCreate, ClassroomUpdate
from app.schemas.system_settings import SystemSettingUpdate
from app.utils.helpers import success_response

from app.models.student import Student
from app.models.session import Session

from app.enums import AttendanceMode

router = APIRouter(prefix="/supervisor/manage/system", tags=["Supervisors Management"])

@router.get(
    "/get-system-settings"
)
async def get_system_settings(
    session: AsyncSession = Depends(
        get_session
    ),
    current_supervisor: Supervisor = Depends(
        get_current_supervisor
    ),
):
    # ==========================================
    # البحث عن Settings الحالية
    # ==========================================

    result = await session.execute(
        select(SystemSetting)
    )

    setting = (
        result.scalars().first()
    )

    # ==========================================
    # إذا لم توجد، ننشئ Default
    # ==========================================

    if setting is None:

        setting = SystemSetting(
            attendance_mode=(
                AttendanceMode.AUTOMATIC
            ),
            times_per_session=2,
            min_confidence_score=0.50,
            is_active=True,
            scan_duration_seconds=120,
        )

        session.add(setting)

        await session.commit()
        await session.refresh(setting)

    # ==========================================
    # Response
    # ==========================================

    return success_response(
        message="System settings retrieved successfully",
        data=setting,
    )


@router.patch(
    "/update-system-settings"
)
async def update_system_settings(
    system_setting: SystemSettingUpdate,

    session: AsyncSession = Depends(
        get_session
    ),

    current_supervisor: Supervisor = Depends(
        get_current_supervisor
    ),
):
    # ==========================================
    # جلب Settings
    # ==========================================

    result = await session.execute(
        select(SystemSetting)
    )

    setting = (
        result.scalars().first()
    )

    # ==========================================
    # حماية إضافية
    # ==========================================

    if setting is None:

        setting = SystemSetting(
            attendance_mode=(
                AttendanceMode.AUTOMATIC
            ),
            times_per_session=2,
            min_confidence_score=0.50,
            is_active=True,
            scan_duration_seconds=120,
        )

        session.add(setting)

        await session.flush()

    # ==========================================
    # الحقول المرسلة فقط
    # ==========================================

    update_data = (
        system_setting.model_dump(
            exclude_unset=True
        )
    )

    if not update_data:

        return success_response(
            message=(
                "No system settings "
                "were changed"
            ),
            data=setting,
        )

    # ==========================================
    # Update
    # ==========================================

    for key, value in (
        update_data.items()
    ):
        setattr(
            setting,
            key,
            value,
        )

    try:

        await session.commit()
        await session.refresh(setting)

    except Exception as exc:

        await session.rollback()

        raise AppException(
            status_code=500,
            message=(
                "Failed to update system "
                f"settings: {str(exc)}"
            ),
        )

    return success_response(
        message=(
            "System settings updated "
            "successfully"
        ),
        data=setting,
    )

@router.post("/create-classroom")
async def create_classroom(
    classroom: ClassroomCreate,
    session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    # ==========================================
    # 1. تنظيف اسم الصف
    # ==========================================

    classroom_name = classroom.name.strip()

    if not classroom_name:
        raise AppException(
            status_code=422,
            message="Classroom name is required.",
        )

    # ==========================================
    # 2. منع التكرار للمشرف نفسه فقط
    # ==========================================

    query = select(Classroom).where(
        Classroom.supervisor_id == current_supervisor.id,
        Classroom.name == classroom_name,
    )

    result = await session.execute(query)

    existing_classroom = result.scalar_one_or_none()

    if existing_classroom:
        raise AppException(
            status_code=409,
            message="Classroom already exists.",
        )

    # ==========================================
    # 3. إنشاء الصف
    # ==========================================

    new_classroom = Classroom(
        name=classroom_name,
        supervisor_id=current_supervisor.id,
    )

    session.add(new_classroom)

    try:
        await session.commit()
        await session.refresh(new_classroom)

    except Exception as exc:
        await session.rollback()

        raise AppException(
            status_code=500,
            message=f"Failed to create classroom: {str(exc)}",
        )

    # ==========================================
    # 4. Response
    # ==========================================

    return success_response(
        message="Classroom created successfully",
        data=new_classroom,
    )

@router.patch("/update-classroom/{classroom_id}")
async def update_classroom(
    classroom_id: int,
    classroom: ClassroomUpdate,
    session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    # ==========================================
    # 1. جلب الصف
    # ==========================================

    result = await session.execute(
        select(Classroom).where(
            Classroom.id == classroom_id
        )
    )

    db_classroom = result.scalar_one_or_none()

    if db_classroom is None:
        raise AppException(
            status_code=404,
            message="Classroom not found.",
        )

    # ==========================================
    # 2. التحقق من الملكية
    # ==========================================

    if db_classroom.supervisor_id != current_supervisor.id:
        raise AppException(
            status_code=403,
            message=(
                "You are not authorized "
                "to update this classroom."
            ),
        )

    # ==========================================
    # 3. البيانات المطلوب تعديلها
    # ==========================================

    update_data = classroom.model_dump(
        exclude_unset=True
    )

    # لا نريد نقل ملكية الصف من هذه الواجهة
    requested_supervisor_id = update_data.pop(
        "supervisor_id",
        None,
    )

    if (
        requested_supervisor_id is not None
        and requested_supervisor_id
        != current_supervisor.id
    ):
        raise AppException(
            status_code=403,
            message=(
                "Changing classroom ownership "
                "is not allowed."
            ),
        )

    # ==========================================
    # 4. تعديل الاسم
    # ==========================================

    if "name" in update_data:

        new_name = update_data["name"].strip()

        if not new_name:
            raise AppException(
                status_code=422,
                message="Classroom name is required.",
            )

        duplicate_result = await session.execute(
            select(Classroom).where(
                Classroom.supervisor_id
                == current_supervisor.id,

                Classroom.name == new_name,

                Classroom.id != classroom_id,
            )
        )

        duplicate = (
            duplicate_result.scalar_one_or_none()
        )

        if duplicate:
            raise AppException(
                status_code=409,
                message=(
                    "Another classroom with "
                    "this name already exists."
                ),
            )

        db_classroom.name = new_name

    # ==========================================
    # 5. حفظ
    # ==========================================

    try:
        await session.commit()
        await session.refresh(db_classroom)

    except Exception as exc:
        await session.rollback()

        raise AppException(
            status_code=500,
            message=f"Failed to update classroom: {str(exc)}",
        )

    return success_response(
        message="Classroom updated successfully",
        data=db_classroom,
    )



# @router.get("/get-classrooms")
# async def get_classrooms(
#     session: AsyncSession = Depends(get_session),
#     current_supervisor: Supervisor = Depends(
#         get_current_supervisor
#     ),
# ):
#     # ==========================================
#     # جلب صفوف المشرف الحالي فقط
#     # ==========================================

#     result = await session.execute(
#         select(Classroom)
#         .where(
#             Classroom.supervisor_id
#             == current_supervisor.id
#         )
#         .order_by(Classroom.id.asc())
#     )

#     classrooms = result.scalars().all()

#     return success_response(
#         message="Classrooms retrieved successfully",
#         data=classrooms,
#     )


@router.get("/get-classrooms")
async def get_classrooms(
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):
    student_count_subq = (
        select(func.count(Student.id))
        .where(Student.classroom_id == Classroom.id)
        .scalar_subquery()
    )

    query = (
        select(Classroom, student_count_subq.label("student_count"))
        .where(Classroom.supervisor_id == current_supervisor.id)
        .options(joinedload(Classroom.supervisor))
    )

    result = await session.execute(query)

    rows = result.all()

    if not rows:
        raise AppException(status_code=404, message="No classrooms found")

    classrooms_data = []
    for classroom_obj, student_count in rows:
        classroom_dict = jsonable_encoder(classroom_obj)

        classroom_dict["student_count"] = student_count

        if classroom_dict.get("supervisor"):
            classroom_dict["supervisor"].pop("password", None)

        classrooms_data.append(classroom_dict)

    return success_response(message="Classrooms retrieved successfully", data=classrooms_data)


@router.get("/get-classroom/{classroom_id}")
async def get_classroom(
    classroom_id: int,
    session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(
        get_current_supervisor
    ),
):
    result = await session.execute(
        select(Classroom).where(
            Classroom.id == classroom_id
        )
    )

    classroom = result.scalar_one_or_none()

    if classroom is None:
        raise AppException(
            status_code=404,
            message="Classroom not found.",
        )

    if classroom.supervisor_id != current_supervisor.id:
        raise AppException(
            status_code=403,
            message=(
                "You are not authorized "
                "to access this classroom."
            ),
        )

    return success_response(
        message="Classroom retrieved successfully",
        data=classroom,
    )


@router.delete("/delete-classroom/{classroom_id}")
async def delete_classroom(
    classroom_id: int,
    session: AsyncSession = Depends(get_session),
    current_supervisor: Supervisor = Depends(
        get_current_supervisor
    ),
):
    # ==========================================
    # 1. جلب الصف
    # ==========================================

    result = await session.execute(
        select(Classroom).where(
            Classroom.id == classroom_id
        )
    )

    classroom = result.scalar_one_or_none()

    if classroom is None:
        raise AppException(
            status_code=404,
            message="Classroom not found.",
        )

    # ==========================================
    # 2. التحقق من الملكية
    # ==========================================

    if classroom.supervisor_id != current_supervisor.id:
        raise AppException(
            status_code=403,
            message=(
                "You are not authorized "
                "to delete this classroom."
            ),
        )

    # ==========================================
    # 3. هل يحتوي على طلاب؟
    # ==========================================

    student_result = await session.execute(
        select(Student.id)
        .where(
            Student.classroom_id == classroom_id
        )
        .limit(1)
    )

    has_students = (
        student_result.scalar_one_or_none()
        is not None
    )

    if has_students:
        raise AppException(
            status_code=409,
            message=(
                "Cannot delete this classroom "
                "because it contains students."
            ),
        )

    # ==========================================
    # 4. هل يحتوي على حصص؟
    # ==========================================

    session_result = await session.execute(
        select(Session.id)
        .where(
            Session.classroom_id == classroom_id
        )
        .limit(1)
    )

    has_sessions = (
        session_result.scalar_one_or_none()
        is not None
    )

    if has_sessions:
        raise AppException(
            status_code=409,
            message=(
                "Cannot delete this classroom "
                "because it contains sessions."
            ),
        )

    # ==========================================
    # 5. حذف الصف
    # ==========================================

    try:
        await session.delete(classroom)
        await session.commit()

    except Exception as exc:
        await session.rollback()

        raise AppException(
            status_code=500,
            message=f"Failed to delete classroom: {str(exc)}",
        )

    return success_response(
        message="Classroom deleted successfully",
        data={
            "classroom_id": classroom_id,
        },
    )

