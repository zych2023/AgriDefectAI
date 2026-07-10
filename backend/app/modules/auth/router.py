from typing import Union
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.deps import get_current_user
from app.common.response import ApiResponse
from app.modules.auth.schemas import RegisterRequest, LoginRequest, TokenResponse, FarmerResponse, ExpertResponse
from app.modules.auth.service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=ApiResponse)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    user = await AuthService.register(db, req)
    return ApiResponse.success(data=user, message="Registration successful")


@router.post("/login", response_model=ApiResponse[TokenResponse])
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    token_data = await AuthService.login(db, req)
    return ApiResponse.success(data=token_data)


@router.get("/me", response_model=ApiResponse)
async def me(current_user=Depends(get_current_user)):
    user_data = await AuthService.get_me(current_user)
    return ApiResponse.success(data=user_data)
