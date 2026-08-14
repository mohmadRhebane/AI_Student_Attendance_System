from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import ExpiredSignatureError, PyJWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppException
from app.database import get_session
from app.models.supervisor import Supervisor

from app.config import SECRET_KEY, ALGORITHM
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt

from pwdlib import PasswordHash

from app import config

password_hasher = PasswordHash.recommended()


ACCESS_TOKEN_EXPIRE_MINUTES = config.ACCESS_TOKEN_EXPIRE_MINUTES
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/supervisors/token")

def get_password_hash(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hasher.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"require_exp": True, "require_sub": True})
    except jwt.InvalidTokenError:
        return None
    else:
        return payload.get("sub")

# -==============================================



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/supervisor/auth/login/")


async def get_current_supervisor(
        token: str = Depends(oauth2_scheme),
        session: AsyncSession = Depends(get_session)
) -> Supervisor:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        supervisor_id: str = payload.get("sub")

        if supervisor_id is None:
            raise AppException(status_code=401, message="Could not validate credentials")

    except ExpiredSignatureError:
        raise AppException(status_code=401, message="Token has expired. Please log in again.")

    except PyJWTError:
        raise AppException(status_code=401, message="Invalid token. Could not validate credentials.")

    try:
        result = await session.execute(
            select(Supervisor).where(Supervisor.id == int(supervisor_id))
        )
        supervisor = result.scalar_one_or_none()

        if supervisor is None:
            raise AppException(status_code=401, message="Supervisor not found in database")

    except ValueError:
        raise AppException(status_code=401, message="Invalid credentials format.")

    return supervisor