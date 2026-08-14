from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
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

router = APIRouter(prefix="/supervisor/manage/system", tags=["Supervisors Management"])

@router.get("/get-system-settings")
async def get_system_settings(session: AsyncSession = Depends(get_session),current_supervisor:Supervisor = Depends(get_current_supervisor)):
    query = await session.execute(select(SystemSetting))
    setting = query.scalars().first()
    if not setting:
        raise AppException(message="No settings found", status_code=404)
    return success_response(message="Settings available",data=setting)


@router.patch("/update-system-settings")
async def update_system_settings(
        system_setting: SystemSettingUpdate,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor),
):
    result = await session.execute(select(SystemSetting))
    setting = result.scalars().first()

    update_data = system_setting.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(setting, key, value)

    await session.commit()
    await session.refresh(setting)
    return success_response(message="System settings updated successfully", data=setting)

@router.post("/create-classroom")
async def create_classroom(
        classroom:ClassroomCreate,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):

    query = select(Classroom).where(Classroom.name == classroom.name)
    result = await session.execute(query)
    exists_class = result.scalar_one_or_none()
    if exists_class:
        raise AppException(status_code=422, message="Class already exists")
    new_class = Classroom(
        name=classroom.name,
        supervisor_id=current_supervisor.id
)

    session.add(new_class)
    await session.commit()
    await session.refresh(new_class)
    return success_response(message="Class created successfully", data=new_class)

@router.patch("/update-classroom/{classroom_id}")
async def update_classroom(
        classroom_id:int,
        classroom: ClassroomUpdate,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):
    query = select(Classroom).where(Classroom.id == classroom_id)
    result = await session.execute(query)
    db_class = result.scalars().first()

    if not db_class:
        raise AppException(status_code=404, message="Classroom not found")
    if db_class.supervisor_id != current_supervisor.id:
        raise AppException(status_code=403, message="You are not authorized to update this classroom.")
    if classroom.supervisor_id is not None and classroom.supervisor_id != current_supervisor.id:
        query_check = select(Supervisor.id).where(Supervisor.id == classroom.supervisor_id)
        result_check = await session.execute(query_check)
        target_supervisor_exists = result_check.scalar_one_or_none()
        if not target_supervisor_exists:
            raise AppException(status_code=404, message="The new Supervisor was not found in the database.")

    update_class = classroom.model_dump(exclude_unset=True)
    for key, value in update_class.items():
        setattr(db_class, key, value)
    await session.commit()
    await session.refresh(db_class)
    return success_response(message="Class updated successfully", data=db_class)

@router.get("/get-classrooms")
async def get_classrooms(
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):
    query = select(Classroom).where(Classroom.supervisor_id == current_supervisor.id).options(joinedload(Classroom.supervisor))
    result = await session.execute(query)
    classrooms = result.scalars().all()

    if not classrooms:
        raise AppException(status_code=404, message="No classrooms found")

    classrooms_dict = jsonable_encoder(classrooms)
    for classroom in classrooms_dict:
        if classroom.get("supervisor"):
            classroom["supervisor"].pop("password", None)

    return success_response(message="Class room retrieved successfully", data=classrooms_dict)


@router.get("/get-classroom/{classroom_id}")
async def get_classroom(
        classroom_id:int,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):
    query = select(Classroom).where(Classroom.id == classroom_id)
    result = await session.execute(query)
    classroom = result.scalar_one_or_none()
    if not classroom:
        raise AppException(status_code=404, message="No classroom found")
    if current_supervisor.id != classroom.supervisor_id:
        raise AppException(message="You are not authorized to retrieve this classroom.")
    return success_response(message="Class room retrieved successfully", data=classroom)

@router.delete("/delete-classroom/{classroom_id}")
async def delete_classroom(
        classroom_id: int,
        session: AsyncSession = Depends(get_session),
        current_supervisor: Supervisor = Depends(get_current_supervisor)
):
    query = select(Classroom).where(Classroom.id == classroom_id)
    result = await session.execute(query)
    classroom = result.scalars().first()
    if not classroom:
        raise AppException(status_code=404, message="Classroom not found")
    if classroom.supervisor_id != current_supervisor.id:
        raise AppException(status_code=403, message="You are not authorized to delete this classroom.")
    await session.delete(classroom)
    await session.commit()
    return success_response(message="Class deleted successfully")

