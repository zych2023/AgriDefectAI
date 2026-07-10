from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import decode_access_token
from app.common.exceptions import UnauthorizedException, ForbiddenException

security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    token = credentials.credentials
    payload = decode_access_token(token)
    if payload is None:
        raise UnauthorizedException(message="Invalid or expired token")
    user_id = payload.get("sub")
    role = payload.get("role")
    if user_id is None or role is None:
        raise UnauthorizedException(message="Invalid token payload")

    # Import here to avoid circular import
    from app.modules.auth.models import Farmer, Expert

    if role == "farmer":
        result = await db.execute(select(Farmer).where(Farmer.id == int(user_id)))
        user = result.scalar_one_or_none()
    elif role in ("expert", "admin"):
        result = await db.execute(select(Expert).where(Expert.id == int(user_id)))
        user = result.scalar_one_or_none()
    else:
        raise UnauthorizedException(message="Unknown user role")

    if user is None:
        raise UnauthorizedException(message="User not found")

    # Store role as attribute for require_role checks
    user.role = role
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_security),
    db: AsyncSession = Depends(get_db),
):
    """Like get_current_user but returns None when no token is provided."""
    if credentials is None:
        return None
    token = credentials.credentials
    payload = decode_access_token(token)
    if payload is None:
        return None
    user_id = payload.get("sub")
    role = payload.get("role")
    if user_id is None or role is None:
        return None

    from app.modules.auth.models import Farmer, Expert

    if role == "farmer":
        result = await db.execute(select(Farmer).where(Farmer.id == int(user_id)))
        user = result.scalar_one_or_none()
    elif role in ("expert", "admin"):
        result = await db.execute(select(Expert).where(Expert.id == int(user_id)))
        user = result.scalar_one_or_none()
    else:
        return None

    if user is not None:
        user.role = role
    return user


def require_role(*roles: str):
    """返回一个 FastAPI 依赖，检查当前用户是否具有指定角色之一"""

    async def role_checker(current_user=Depends(get_current_user)):
        if current_user.role not in roles:
            raise ForbiddenException(message=f"Requires one of roles: {roles}")
        return current_user

    return role_checker
