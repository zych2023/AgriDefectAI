from datetime import datetime
from typing import Union
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    real_name: str = Field(..., min_length=1, max_length=30)
    role: str = Field(default="farmer", pattern=r"^(farmer|expert)$")  # 注册类型
    phone: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str
    role: str = Field(default="farmer", pattern=r"^(farmer|expert)$")  # 登录时指定角色


class FarmerResponse(BaseModel):
    id: int
    username: str
    real_name: str
    role: str = "farmer"
    phone: str | None = None
    status: int
    registered_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExpertResponse(BaseModel):
    id: int
    username: str
    real_name: str
    role: str = "expert"
    phone: str | None = None
    specialty: str | None = None
    region: str | None = None
    status: int
    role_level: int = Field(alias="role", description="1-普通专家 2-管理员")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Union[FarmerResponse, ExpertResponse]
