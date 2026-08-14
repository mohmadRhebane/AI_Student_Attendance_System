
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func
from datetime import timedelta
from fastapi.security import OAuth2PasswordBearer
from app.core.exceptions import AppException, validation_exception_handler
from app.core.security import(
    get_password_hash,
    verify_password,
    verify_access_token,
    create_access_token
)
from app.models.supervisor import Supervisor
from app.schemas.supervisor import (
    SupervisorCreate,
    SupervisorUpdate, LoginRequest,
)


class SupervisorService:






    @staticmethod
    async def get_supervisor(session: AsyncSession, token: str):
        supervisor_id = verify_access_token(token)
        if not supervisor_id:
            raise AppException(status_code=401, message="Unauthenticated")
        try:
            supervisor_id_int = int(supervisor_id)
        except(TypeError | ValueError):
            raise AppException(status_code=401, message="Invalid token or expired")
        result = await session.execute(
            select(Supervisor).where(Supervisor.id == supervisor_id_int)
        )
        user = result.scalar_one_or_none()
        if not user:
            raise AppException(status_code=404, message="User not found")
        return user




    @staticmethod
    async def delete_supervisor(session: AsyncSession, supervisor_id: int):
        """حذف مشرف"""
        supervisor = await SupervisorService.get_supervisor_by_id(session, supervisor_id)

        await session.delete(supervisor)
        await session.commit()
        return True

