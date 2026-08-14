from http.client import HTTPException

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func

from app.core.exceptions import AppException
from app.core.security import create_access_token, get_current_supervisor, \
    get_password_hash, verify_password
from app.models import Supervisor
from app.utils.helpers import *
from app.database import get_session
from app.schemas.supervisor import (
    SupervisorCreate, SupervisorUpdate,
    ResponseWithoutPassword, LoginRequest
)

from typing import List
from app.schemas.supervisor import SupervisorResponsePrivate

router = APIRouter(prefix="/supervisor/auth", tags=["Supervisors Management"])


@router.post("/register/", response_model=SupervisorResponsePrivate)
async def create_supervisor(
    data: SupervisorCreate,
    session: AsyncSession = Depends(get_session)
):
    query = select(Supervisor).where(
        or_(
            Supervisor.email == data.email,
            func.lower(Supervisor.username) == data.username.lower()
        )
    )

    result = await session.execute(query)
    existing_user = result.scalar_one_or_none()

    if existing_user:
        if existing_user.email == data.email:
            raise AppException(status_code=422, message="This email already exists")
        raise AppException(status_code=422, message="This username already exists")
    new_supervisor = Supervisor(
        username=data.username,
        full_name=data.full_name,
        email=data.email,
        password=get_password_hash(data.password),
    )
    session.add(new_supervisor)
    await session.commit()
    await session.refresh(new_supervisor)
    access_token = create_access_token(data={"sub": str(new_supervisor.id)})
    return register_response(data=new_supervisor, token=access_token,token_type="Bearer")

@router.post("/login/")
async def login_supervisor(
        login_data: LoginRequest,
        session: AsyncSession = Depends(get_session)
):
    query = select(Supervisor).where(
        or_(
            Supervisor.email == login_data.email_username,
            Supervisor.username == login_data.email_username
        )
    )
    result = await session.execute(query)
    supervisor = result.scalar_one_or_none()

    if not supervisor:
        raise AppException(status_code=401, message="Invalid email or password, please try again")
    if not verify_password(login_data.password, supervisor.password):
        raise AppException(status_code=401, message="Invalid email or password")

    access_token = create_access_token(data={"sub": str(supervisor.id)})
    return login_response(
        data=supervisor,
        token=access_token,
        token_type="Bearer"
    )


@router.get("/get-supervisor-profile", response_model=SupervisorResponsePrivate)
async def get_supervisor(user: Supervisor = Depends(get_current_supervisor)):
    return success_response(message="Supervisor profile", data=user)


@router.get("/get-supervisors", response_model=List[SupervisorResponsePrivate])
async def get_supervisors(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Supervisor).offset(0).limit(100))
    supervisors = result.scalars().all()
    return success_response(message="Supervisors", data=supervisors)


@router.get("/get/{supervisor_id}", response_model=SupervisorResponsePrivate)
async def get_supervisor(supervisor_id: int,session: AsyncSession = Depends(get_session)):
    supervisor = await get_supervisor_by_id(session, supervisor_id)

    if not supervisor:
        raise AppException(message="Supervisor not found", status_code=404)
    return success_response(message="Supervisor retrieved successfully.", data=supervisor)


@router.patch("/update-account/{supervisor_id}", response_model=SupervisorResponsePrivate)
async def update_supervisor(
    supervisor_id: int,
    data: SupervisorUpdate,
    session: AsyncSession = Depends(get_session),
    current_user: Supervisor = Depends(get_current_supervisor)
):
    if current_user.id != supervisor_id:
        raise AppException(
            status_code=status.HTTP_403_FORBIDDEN,
            message="You are not authorized to perform this action."
        )
    supervisor = await get_supervisor_by_id(session, supervisor_id)

    update_data = data.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        if key == "password":
            raise AppException(status_code=403, message="You can\'t update this field")
        setattr(supervisor, key, value)

    await session.commit()
    await session.refresh(supervisor)
    return success_response(message="Supervisor updated successfully.", data=ResponseWithoutPassword(supervisor))


@router.delete("/delete-account/{supervisor_id}", status_code=status.HTTP_200_OK)
async def delete_supervisor(
    supervisor_id: int,
    session: AsyncSession = Depends(get_session),
current_user: Supervisor = Depends(get_current_supervisor)
):
    if current_user.id != supervisor_id:
        raise AppException(
            status_code=status.HTTP_403_FORBIDDEN,
            message="You are not authorized to perform this action."
        )
    supervisor = await get_supervisor_by_id(session, supervisor_id)
    await session.delete(supervisor)
    await session.commit()
    return success_response(message="Supervisor deleted successfully")



async def get_supervisor_by_id(session: AsyncSession, supervisor_id: int):
    result = await session.execute(select(Supervisor).where(Supervisor.id == supervisor_id))
    supervisor = result.scalars().first()
    return supervisor