from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.auth.models import Farmer, Expert
from app.modules.auth.schemas import RegisterRequest, LoginRequest, TokenResponse, FarmerResponse, ExpertResponse
from app.core.security import hash_password, verify_password, create_access_token
from app.common.exceptions import BusinessException


class AuthService:

    @staticmethod
    async def register(db: AsyncSession, req: RegisterRequest) -> FarmerResponse | ExpertResponse:
        if req.role == "farmer":
            model = Farmer
            result = await db.execute(select(Farmer).where(Farmer.username == req.username))
        else:
            model = Expert
            result = await db.execute(select(Expert).where(Expert.username == req.username))

        if result.scalar_one_or_none():
            raise BusinessException(code=409, message="Username already exists")

        user = model(
            username=req.username,
            password=hash_password(req.password),
            real_name=req.real_name,
            phone=req.phone,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        if isinstance(user, Farmer):
            return FarmerResponse.model_validate(user)
        return ExpertResponse.model_validate(user)

    @staticmethod
    async def login(db: AsyncSession, req: LoginRequest) -> TokenResponse:
        if req.role == "farmer":
            result = await db.execute(select(Farmer).where(Farmer.username == req.username))
            user = result.scalar_one_or_none()
            role_str = "farmer"
        else:
            result = await db.execute(select(Expert).where(Expert.username == req.username))
            user = result.scalar_one_or_none()
            role_str = "expert"

        if not user or not verify_password(req.password, user.password):
            raise BusinessException(code=401, message="Invalid username or password")

        if user.status == 0:
            raise BusinessException(code=403, message="Account is disabled")

        token = create_access_token({"sub": str(user.id), "role": role_str})

        if isinstance(user, Farmer):
            user_data = FarmerResponse.model_validate(user)
        else:
            user_data = ExpertResponse.model_validate(user)

        return TokenResponse(access_token=token, user=user_data)

    @staticmethod
    async def get_me(user) -> FarmerResponse | ExpertResponse:
        if isinstance(user, Farmer):
            return FarmerResponse.model_validate(user)
        return ExpertResponse.model_validate(user)
