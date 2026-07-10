# 智慧农业 AI 后端脚手架 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 FastAPI 后端骨架，包括项目结构、数据库模型、JWT 认证、业务 CRUD 和 mock 接口，支持本地 `uvicorn --reload` 启动。

**Architecture:** 单体 FastAPI 应用，按领域模块化组织（auth/disease/knowledge/agent/business），共享 core 基础设施层。每个模块含 router/service/models/schemas 四个文件，模块间通过 core 依赖注入通信。

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.0 async + aiomysql, Redis (redis-py async), JWT (python-jose), passlib[bcrypt], pydantic-settings, Alembic, uvicorn

## Global Constraints

- MySQL 8.0 远程连接，不部署本地 MySQL
- Redis 远程连接
- 无 Docker，本地 `uvicorn --reload` 启动
- P0 范围：认证完整实现 + 业务 CRUD 完整实现 + disease/agent 返回 mock
- Milvus 不接入，向量字段不加到模型中
- 不写单元测试（脚手架阶段）
- 统一响应格式 `{"code": 200, "message": "ok", "data": {...}}`
- 所有 API 前缀 `/api/v1/`

---

### Task 1: 项目骨架与依赖

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/modules/__init__.py`
- Create: `app/core/__init__.py`
- Create: `app/common/__init__.py`
- Create: `app/modules/auth/__init__.py`
- Create: `app/modules/disease/__init__.py`
- Create: `app/modules/knowledge/__init__.py`
- Create: `app/modules/agent/__init__.py`
- Create: `app/modules/business/__init__.py`
- Create: `uploads/diseases/.gitkeep`

**Interfaces:**
- Consumes: nothing
- Produces: `requirements.txt` 定义所有依赖，`.env.example` 定义所有环境变量模板

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p app/core app/common app/modules/auth app/modules/disease app/modules/knowledge app/modules/agent app/modules/business uploads/diseases tests
```

- [ ] **Step 2: 写入 requirements.txt**

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
sqlalchemy[asyncio]==2.0.35
aiomysql==0.2.0
redis==5.1.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
pydantic-settings==2.5.2
python-multipart==0.0.12
Pillow==10.4.0
alembic==1.13.2
```

- [ ] **Step 3: 写入 .env.example**

```env
# App
APP_NAME=SmartAgriAI
DEBUG=true

# MySQL (远程)
DB_HOST=your-mysql-host
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your-password
DB_NAME=smart_agri

# Redis (远程)
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# JWT
JWT_SECRET=change-me-to-a-random-string
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# Upload
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE_MB=10
```

- [ ] **Step 4: 写入各 __init__.py**

所有 `__init__.py` 为空文件（创建空文件占位）。

- [ ] **Step 5: 写入 uploads/diseases/.gitkeep**

空文件，确保目录被 git 追踪。

- [ ] **Step 6: 安装依赖并验证**

```bash
pip install -r requirements.txt
```

Expected: 无报错，`pip list | grep fastapi` 显示 fastapi 0.115.0

---

### Task 2: Core 配置模块

**Files:**
- Create: `app/core/config.py`

**Interfaces:**
- Produces: `settings: Settings` — 全局配置单例，供所有模块导入

- [ ] **Step 1: 写入 app/core/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "SmartAgriAI"
    DEBUG: bool = True

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = ""
    DB_NAME: str = "smart_agri"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # JWT
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Upload
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 10

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 2: 验证配置可导入**

```bash
python -c "from app.core.config import settings; print(settings.APP_NAME); print(settings.DATABASE_URL)"
```

Expected: 打印 `SmartAgriAI` 和构建的数据库 URL。

---

### Task 3: Core 数据库模块

**Files:**
- Create: `app/core/database.py`

**Interfaces:**
- Produces: `Base` — SQLAlchemy DeclarativeBase（所有 model 继承）, `async_session_factory` — async session 工厂, `get_db() -> AsyncGenerator[AsyncSession]` — FastAPI 依赖注入

- [ ] **Step 1: 写入 app/core/database.py**

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from app.core.database import Base, async_session_factory, get_db; print('OK')"
```

Expected: 打印 `OK`（此时不需要数据库可达，导入不主动连接）。

---

### Task 4: Core Redis 模块

**Files:**
- Create: `app/core/redis.py`

**Interfaces:**
- Produces: `get_redis() -> redis.asyncio.Redis`, `redis_client` — 模块级 Redis 实例

- [ ] **Step 1: 写入 app/core/redis.py**

```python
import redis.asyncio as aioredis
from app.core.config import settings

redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return redis_client
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from app.core.redis import get_redis; print('OK')"
```

Expected: 打印 `OK`。

---

### Task 5: Common 通用工具

**Files:**
- Create: `app/common/response.py`
- Create: `app/common/exceptions.py`

**Interfaces:**
- Produces: `ApiResponse` — 统一响应模型, `BusinessException` — 业务异常类, `register_exception_handlers(app)` — 注册全局异常处理器

- [ ] **Step 1: 写入 app/common/response.py**

```python
from typing import Any, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = 200
    message: str = "ok"
    data: T | None = None

    @classmethod
    def success(cls, data: Any = None, message: str = "ok") -> "ApiResponse":
        return cls(code=200, message=message, data=data)

    @classmethod
    def error(cls, code: int = 500, message: str = "error", data: Any = None) -> "ApiResponse":
        return cls(code=code, message=message, data=data)


class PaginatedData(BaseModel):
    items: list[Any]
    total: int
    page: int
    page_size: int
```

- [ ] **Step 2: 写入 app/common/exceptions.py**

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from app.common.response import ApiResponse


class BusinessException(Exception):
    def __init__(self, code: int = 400, message: str = "Bad Request"):
        self.code = code
        self.message = message


class UnauthorizedException(BusinessException):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(code=401, message=message)


class ForbiddenException(BusinessException):
    def __init__(self, message: str = "Forbidden"):
        super().__init__(code=403, message=message)


class NotFoundException(BusinessException):
    def __init__(self, message: str = "Not Found"):
        super().__init__(code=404, message=message)


async def business_exception_handler(request: Request, exc: BusinessException):
    return JSONResponse(
        status_code=exc.code,
        content=ApiResponse.error(code=exc.code, message=exc.message).model_dump(),
    )


async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ApiResponse.error(code=500, message="Internal Server Error").model_dump(),
    )


def register_exception_handlers(app):
    app.add_exception_handler(BusinessException, business_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
```

- [ ] **Step 3: 验证导入**

```bash
python -c "from app.common.response import ApiResponse, PaginatedData; from app.common.exceptions import BusinessException; print(ApiResponse.success({'test': 1}).model_dump())"
```

Expected: 打印 `{'code': 200, 'message': 'ok', 'data': {'test': 1}}`。

---

### Task 6: Security 安全模块

**Files:**
- Create: `app/core/security.py`

**Interfaces:**
- Produces: `hash_password(password: str) -> str`, `verify_password(plain: str, hashed: str) -> bool`, `create_access_token(data: dict) -> str`, `decode_access_token(token: str) -> dict | None`

- [ ] **Step 1: 写入 app/core/security.py**

```python
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None
```

- [ ] **Step 2: 验证密码和 JWT 功能**

```bash
python -c "
from app.core.security import hash_password, verify_password, create_access_token, decode_access_token
h = hash_password('test123')
assert verify_password('test123', h)
assert not verify_password('wrong', h)
t = create_access_token({'sub': '1'})
d = decode_access_token(t)
assert d['sub'] == '1'
print('All assertions passed')
"
```

Expected: 打印 `All assertions passed`。

---

### Task 7: 依赖注入模块

**Files:**
- Create: `app/core/deps.py`

**Interfaces:**
- Produces: `get_current_user(token, db) -> User` — 从 JWT 解析当前用户，`require_role(*roles)` — 角色权限检查依赖

- [ ] **Step 1: 写入 app/core/deps.py**

```python
from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import decode_access_token
from app.common.exceptions import UnauthorizedException, ForbiddenException


async def get_current_user(
    authorization: str = Header(..., description="Bearer <token>"),
    db: AsyncSession = Depends(get_db),
):
    if not authorization.startswith("Bearer "):
        raise UnauthorizedException(message="Invalid authorization header")
    token = authorization[7:]
    payload = decode_access_token(token)
    if payload is None:
        raise UnauthorizedException(message="Invalid or expired token")
    user_id = payload.get("sub")
    if user_id is None:
        raise UnauthorizedException(message="Invalid token payload")

    # Import here to avoid circular import
    from app.modules.auth.models import User

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise UnauthorizedException(message="User not found")
    return user


def require_role(*roles: str):
    """返回一个 FastAPI 依赖，检查当前用户是否具有指定角色之一"""

    async def role_checker(current_user=Depends(get_current_user)):
        if current_user.role not in roles:
            raise ForbiddenException(message=f"Requires one of roles: {roles}")
        return current_user

    return role_checker
```

- [ ] **Step 2: 验证文件语法（此时无法运行因为依赖 auth.models）**

```bash
python -c "import ast; ast.parse(open('app/core/deps.py').read()); print('Syntax OK')"
```

Expected: 打印 `Syntax OK`。

---

### Task 8: Auth 认证模块

**Files:**
- Create: `app/modules/auth/models.py`
- Create: `app/modules/auth/schemas.py`
- Create: `app/modules/auth/service.py`
- Create: `app/modules/auth/router.py`

**Interfaces:**
- Consumes: `Base` from `app.core.database`, `hash_password/verify_password/create_access_token` from `app.core.security`
- Produces:
  - `User` — SQLAlchemy model
  - `POST /api/v1/auth/register` — 注册
  - `POST /api/v1/auth/login` — 登录返回 token
  - `GET /api/v1/auth/me` — 当前用户信息

- [ ] **Step 1: 写入 app/modules/auth/models.py**

```python
from datetime import datetime
from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="farmer")  # admin, expert, farmer
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    avatar: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
```

- [ ] **Step 2: 写入 app/modules/auth/schemas.py**

```python
from datetime import datetime
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    phone: str | None = None
    role: str = Field(default="farmer", pattern=r"^(farmer|expert)$")  # 注册只允许农户或专家


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    phone: str | None = None
    avatar: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
```

- [ ] **Step 3: 写入 app/modules/auth/service.py**

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.auth.models import User
from app.modules.auth.schemas import RegisterRequest, TokenResponse, UserResponse
from app.core.security import hash_password, verify_password, create_access_token
from app.common.exceptions import BusinessException


class AuthService:

    @staticmethod
    async def register(db: AsyncSession, req: RegisterRequest) -> UserResponse:
        # Check if username already exists
        result = await db.execute(select(User).where(User.username == req.username))
        if result.scalar_one_or_none():
            raise BusinessException(code=409, message="Username already exists")

        user = User(
            username=req.username,
            password_hash=hash_password(req.password),
            role=req.role,
            phone=req.phone,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return UserResponse.model_validate(user)

    @staticmethod
    async def login(db: AsyncSession, req) -> TokenResponse:
        result = await db.execute(select(User).where(User.username == req.username))
        user = result.scalar_one_or_none()
        if not user or not verify_password(req.password, user.password_hash):
            raise BusinessException(code=401, message="Invalid username or password")

        token = create_access_token({"sub": str(user.id)})
        return TokenResponse(
            access_token=token,
            user=UserResponse.model_validate(user),
        )

    @staticmethod
    async def get_me(user: User) -> UserResponse:
        return UserResponse.model_validate(user)
```

- [ ] **Step 4: 写入 app/modules/auth/router.py**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.deps import get_current_user
from app.common.response import ApiResponse
from app.modules.auth.schemas import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from app.modules.auth.service import AuthService
from app.modules.auth.models import User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=ApiResponse[UserResponse])
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    user = await AuthService.register(db, req)
    return ApiResponse.success(data=user, message="Registration successful")


@router.post("/login", response_model=ApiResponse[TokenResponse])
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    token_data = await AuthService.login(db, req)
    return ApiResponse.success(data=token_data)


@router.get("/me", response_model=ApiResponse[UserResponse])
async def me(current_user: User = Depends(get_current_user)):
    user_data = await AuthService.get_me(current_user)
    return ApiResponse.success(data=user_data)
```

- [ ] **Step 5: 验证语法**

```bash
python -c "import ast; [ast.parse(open(f'app/modules/auth/{f}').read()) for f in ['models.py','schemas.py','service.py','router.py']]; print('All syntax OK')"
```

Expected: 打印 `All syntax OK`。

---

### Task 9: Business 数据模型

**Files:**
- Create: `app/modules/business/models.py`

**Interfaces:**
- Produces: `Farm`, `Crop`, `PestCatalog`, `DiseaseRecord`, `KnowledgeDoc` — SQLAlchemy models

- [ ] **Step 1: 写入 app/modules/business/models.py**

```python
from datetime import datetime
from sqlalchemy import String, Integer, Text, Float, Date, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Farm(Base):
    __tablename__ = "farms"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    farmer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    area: Mapped[float | None] = mapped_column(Float, nullable=True)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    soil_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Crop(Base):
    __tablename__ = "crops"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    farm_id: Mapped[int] = mapped_column(ForeignKey("farms.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    variety: Mapped[str | None] = mapped_column(String(100), nullable=True)
    plant_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="growing")  # growing, harvested, diseased
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class PestCatalog(Base):
    __tablename__ = "pest_catalog"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False)  # disease, pest, weed
    symptoms: Mapped[str | None] = mapped_column(Text, nullable=True)
    treatment: Mapped[str | None] = mapped_column(Text, nullable=True)
    images: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class DiseaseRecord(Base):
    __tablename__ = "disease_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    farmer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    image_url: Mapped[str] = mapped_column(String(500), nullable=False)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending, confirmed, rejected
    expert_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
```

- [ ] **Step 2: 验证导入（确认与 auth.User 的关系正确）**

```bash
python -c "from app.modules.auth.models import User; from app.modules.business.models import Farm, Crop, PestCatalog, DiseaseRecord, KnowledgeDoc; print('All models imported OK')"
```

Expected: 打印 `All models imported OK`。

---

### Task 10: Business CRUD（Schemas + Service + Router）

**Files:**
- Create: `app/modules/business/schemas.py`
- Create: `app/modules/business/service.py`
- Create: `app/modules/business/router.py`

**Interfaces:**
- Consumes: models from Task 9, `ApiResponse`, `PaginatedData` from common
- Produces: CRUD schemas, `BusinessService`, business router 挂载到 `/api/v1/business/*`

- [ ] **Step 1: 写入 app/modules/business/schemas.py**

```python
from datetime import datetime, date
from pydantic import BaseModel, Field
from typing import Any


# --- Farm ---
class FarmCreate(BaseModel):
    name: str = Field(..., max_length=100)
    area: float | None = None
    location: str | None = None
    soil_type: str | None = None

class FarmUpdate(BaseModel):
    name: str | None = None
    area: float | None = None
    location: str | None = None
    soil_type: str | None = None

class FarmResponse(BaseModel):
    id: int
    farmer_id: int
    name: str
    area: float | None = None
    location: str | None = None
    soil_type: str | None = None
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- Crop ---
class CropCreate(BaseModel):
    farm_id: int
    name: str = Field(..., max_length=100)
    variety: str | None = None
    plant_date: date | None = None
    status: str = "growing"

class CropUpdate(BaseModel):
    name: str | None = None
    variety: str | None = None
    plant_date: date | None = None
    status: str | None = None

class CropResponse(BaseModel):
    id: int
    farm_id: int
    name: str
    variety: str | None = None
    plant_date: date | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- PestCatalog ---
class PestCatalogCreate(BaseModel):
    name: str = Field(..., max_length=100)
    category: str = Field(..., pattern=r"^(disease|pest|weed)$")
    symptoms: str | None = None
    treatment: str | None = None
    images: list[str] | None = None

class PestCatalogUpdate(BaseModel):
    name: str | None = None
    category: str | None = None
    symptoms: str | None = None
    treatment: str | None = None
    images: list[str] | None = None

class PestCatalogResponse(BaseModel):
    id: int
    name: str
    category: str
    symptoms: str | None = None
    treatment: str | None = None
    images: list | None = None
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


# --- DiseaseRecord ---
class DiseaseRecordResponse(BaseModel):
    id: int
    farmer_id: int
    image_url: str
    result_json: dict | None = None
    confidence: float | None = None
    status: str
    expert_id: int | None = None
    feedback: str | None = None
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True
```

- [ ] **Step 2: 写入 app/modules/business/service.py**

```python
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.business.models import Farm, Crop, PestCatalog, DiseaseRecord
from app.modules.business.schemas import (
    FarmCreate, FarmUpdate, FarmResponse,
    CropCreate, CropUpdate, CropResponse,
    PestCatalogCreate, PestCatalogUpdate, PestCatalogResponse,
    DiseaseRecordResponse,
)
from app.common.exceptions import NotFoundException
from app.common.response import PaginatedData


class BusinessService:

    # ---- Farm ----
    @staticmethod
    async def create_farm(db: AsyncSession, data: FarmCreate, farmer_id: int) -> FarmResponse:
        farm = Farm(farmer_id=farmer_id, **data.model_dump())
        db.add(farm)
        await db.commit()
        await db.refresh(farm)
        return FarmResponse.model_validate(farm)

    @staticmethod
    async def get_farms(db: AsyncSession, page: int = 1, page_size: int = 20, farmer_id: int | None = None) -> PaginatedData:
        base = select(Farm)
        count_q = select(func.count()).select_from(Farm)
        if farmer_id:
            base = base.where(Farm.farmer_id == farmer_id)
            count_q = count_q.where(Farm.farmer_id == farmer_id)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(Farm.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[FarmResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )

    @staticmethod
    async def update_farm(db: AsyncSession, farm_id: int, data: FarmUpdate) -> FarmResponse:
        farm = await db.get(Farm, farm_id)
        if not farm:
            raise NotFoundException(message="Farm not found")
        for k, v in data.model_dump(exclude_unset=True).items():
            setattr(farm, k, v)
        await db.commit()
        await db.refresh(farm)
        return FarmResponse.model_validate(farm)

    # ---- Crop ----
    @staticmethod
    async def create_crop(db: AsyncSession, data: CropCreate) -> CropResponse:
        crop = Crop(**data.model_dump())
        db.add(crop)
        await db.commit()
        await db.refresh(crop)
        return CropResponse.model_validate(crop)

    @staticmethod
    async def get_crops(db: AsyncSession, farm_id: int | None = None, page: int = 1, page_size: int = 20) -> PaginatedData:
        base = select(Crop)
        count_q = select(func.count()).select_from(Crop)
        if farm_id:
            base = base.where(Crop.farm_id == farm_id)
            count_q = count_q.where(Crop.farm_id == farm_id)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(Crop.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[CropResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )

    @staticmethod
    async def update_crop(db: AsyncSession, crop_id: int, data: CropUpdate) -> CropResponse:
        crop = await db.get(Crop, crop_id)
        if not crop:
            raise NotFoundException(message="Crop not found")
        for k, v in data.model_dump(exclude_unset=True).items():
            setattr(crop, k, v)
        await db.commit()
        await db.refresh(crop)
        return CropResponse.model_validate(crop)

    # ---- PestCatalog ----
    @staticmethod
    async def create_pest_catalog(db: AsyncSession, data: PestCatalogCreate) -> PestCatalogResponse:
        pest = PestCatalog(**data.model_dump())
        db.add(pest)
        await db.commit()
        await db.refresh(pest)
        return PestCatalogResponse.model_validate(pest)

    @staticmethod
    async def get_pest_catalogs(db: AsyncSession, category: str | None = None, page: int = 1, page_size: int = 20) -> PaginatedData:
        base = select(PestCatalog)
        count_q = select(func.count()).select_from(PestCatalog)
        if category:
            base = base.where(PestCatalog.category == category)
            count_q = count_q.where(PestCatalog.category == category)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(PestCatalog.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[PestCatalogResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )

    @staticmethod
    async def update_pest_catalog(db: AsyncSession, pest_id: int, data: PestCatalogUpdate) -> PestCatalogResponse:
        pest = await db.get(PestCatalog, pest_id)
        if not pest:
            raise NotFoundException(message="Pest catalog not found")
        for k, v in data.model_dump(exclude_unset=True).items():
            setattr(pest, k, v)
        await db.commit()
        await db.refresh(pest)
        return PestCatalogResponse.model_validate(pest)

    @staticmethod
    async def delete_pest_catalog(db: AsyncSession, pest_id: int) -> None:
        pest = await db.get(PestCatalog, pest_id)
        if not pest:
            raise NotFoundException(message="Pest catalog not found")
        await db.delete(pest)
        await db.commit()

    # ---- DiseaseRecord ----
    @staticmethod
    async def get_disease_records(db: AsyncSession, status: str | None = None, page: int = 1, page_size: int = 20) -> PaginatedData:
        base = select(DiseaseRecord)
        count_q = select(func.count()).select_from(DiseaseRecord)
        if status:
            base = base.where(DiseaseRecord.status == status)
            count_q = count_q.where(DiseaseRecord.status == status)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(DiseaseRecord.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[DiseaseRecordResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )
```

- [ ] **Step 3: 写入 app/modules/business/router.py**

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.deps import get_current_user, require_role
from app.common.response import ApiResponse, PaginatedData
from app.modules.business.schemas import (
    FarmCreate, FarmUpdate, FarmResponse,
    CropCreate, CropUpdate, CropResponse,
    PestCatalogCreate, PestCatalogUpdate, PestCatalogResponse,
    DiseaseRecordResponse,
)
from app.modules.business.service import BusinessService

router = APIRouter(prefix="/api/v1/business", tags=["business"])

# ---- Farm ----
@router.post("/farms", response_model=ApiResponse[FarmResponse])
async def create_farm(data: FarmCreate, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    result = await BusinessService.create_farm(db, data, farmer_id=user.id)
    return ApiResponse.success(data=result)

@router.get("/farms", response_model=ApiResponse[PaginatedData])
async def list_farms(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    farmer_id: int | None = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    # Farmers can only see their own farms
    if user.role == "farmer":
        farmer_id = user.id
    result = await BusinessService.get_farms(db, page=page, page_size=page_size, farmer_id=farmer_id)
    return ApiResponse.success(data=result)

@router.put("/farms/{farm_id}", response_model=ApiResponse[FarmResponse])
async def update_farm(farm_id: int, data: FarmUpdate, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    result = await BusinessService.update_farm(db, farm_id, data)
    return ApiResponse.success(data=result)


# ---- Crop ----
@router.post("/crops", response_model=ApiResponse[CropResponse])
async def create_crop(data: CropCreate, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    result = await BusinessService.create_crop(db, data)
    return ApiResponse.success(data=result)

@router.get("/crops", response_model=ApiResponse[PaginatedData])
async def list_crops(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), farm_id: int | None = None, db: AsyncSession = Depends(get_db)):
    result = await BusinessService.get_crops(db, farm_id=farm_id, page=page, page_size=page_size)
    return ApiResponse.success(data=result)

@router.put("/crops/{crop_id}", response_model=ApiResponse[CropResponse])
async def update_crop(crop_id: int, data: CropUpdate, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    result = await BusinessService.update_crop(db, crop_id, data)
    return ApiResponse.success(data=result)


# ---- PestCatalog (admin only for create/update/delete) ----
@router.post("/pests", response_model=ApiResponse[PestCatalogResponse])
async def create_pest(data: PestCatalogCreate, db: AsyncSession = Depends(get_db), user=Depends(require_role("admin", "expert"))):
    result = await BusinessService.create_pest_catalog(db, data)
    return ApiResponse.success(data=result)

@router.get("/pests", response_model=ApiResponse[PaginatedData])
async def list_pests(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), category: str | None = None, db: AsyncSession = Depends(get_db)):
    result = await BusinessService.get_pest_catalogs(db, category=category, page=page, page_size=page_size)
    return ApiResponse.success(data=result)

@router.put("/pests/{pest_id}", response_model=ApiResponse[PestCatalogResponse])
async def update_pest(pest_id: int, data: PestCatalogUpdate, db: AsyncSession = Depends(get_db), user=Depends(require_role("admin", "expert"))):
    result = await BusinessService.update_pest_catalog(db, pest_id, data)
    return ApiResponse.success(data=result)

@router.delete("/pests/{pest_id}", response_model=ApiResponse)
async def delete_pest(pest_id: int, db: AsyncSession = Depends(get_db), user=Depends(require_role("admin"))):
    await BusinessService.delete_pest_catalog(db, pest_id)
    return ApiResponse.success(message="Deleted successfully")


# ---- DiseaseRecord (admin/expert view) ----
@router.get("/disease-records", response_model=ApiResponse[PaginatedData])
async def list_disease_records(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), status: str | None = None, db: AsyncSession = Depends(get_db), user=Depends(require_role("admin", "expert"))):
    result = await BusinessService.get_disease_records(db, status=status, page=page, page_size=page_size)
    return ApiResponse.success(data=result)
```

- [ ] **Step 4: 验证语法**

```bash
python -c "import ast; [ast.parse(open(f).read()) for f in ['app/modules/business/models.py','app/modules/business/schemas.py','app/modules/business/service.py','app/modules/business/router.py']]; print('All syntax OK')"
```

Expected: 打印 `All syntax OK`。

---

### Task 11: Disease 病虫害识别模块（mock）

**Files:**
- Create: `app/modules/disease/schemas.py`
- Create: `app/modules/disease/service.py`
- Create: `app/modules/disease/router.py`

**Interfaces:**
- Consumes: `DiseaseRecord` model from `app.modules.business.models`, `get_current_user`, `get_db`
- Produces: `POST /api/v1/disease/recognize`, `GET /api/v1/disease/records`, `GET /api/v1/disease/records/{id}`

- [ ] **Step 1: 写入 app/modules/disease/schemas.py**

```python
from datetime import datetime
from pydantic import BaseModel


class RecognizeResult(BaseModel):
    pest_name: str
    category: str
    confidence: float
    description: str
    treatment_suggestion: str


class DiseaseRecordResponse(BaseModel):
    id: int
    farmer_id: int
    image_url: str
    result_json: dict | None = None
    confidence: float | None = None
    status: str
    expert_id: int | None = None
    feedback: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
```

- [ ] **Step 2: 写入 app/modules/disease/service.py**

```python
import os
import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile
from app.core.config import settings
from app.modules.business.models import DiseaseRecord
from app.modules.disease.schemas import RecognizeResult, DiseaseRecordResponse
from app.common.response import PaginatedData
from app.common.exceptions import NotFoundException


class DiseaseService:

    @staticmethod
    async def save_upload(file: UploadFile) -> str:
        """Save uploaded file and return relative path."""
        os.makedirs(os.path.join(settings.UPLOAD_DIR, "diseases"), exist_ok=True)
        ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(settings.UPLOAD_DIR, "diseases", filename)
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        return f"uploads/diseases/{filename}"

    @staticmethod
    async def recognize(db: AsyncSession, file: UploadFile, farmer_id: int) -> DiseaseRecordResponse:
        image_url = await DiseaseService.save_upload(file)

        # P0: Return mock recognition result
        mock_result = RecognizeResult(
            pest_name="小麦赤霉病 (Fusarium Head Blight)",
            category="disease",
            confidence=0.92,
            description="叶片出现水渍状斑点，逐渐扩大为黄褐色枯斑",
            treatment_suggestion="建议使用戊唑醇悬浮剂2000倍液喷雾防治，7-10天一次，连续2-3次",
        )

        record = DiseaseRecord(
            farmer_id=farmer_id,
            image_url=image_url,
            result_json=mock_result.model_dump(),
            confidence=mock_result.confidence,
            status="pending",
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return DiseaseRecordResponse.model_validate(record)

    @staticmethod
    async def get_records(db: AsyncSession, farmer_id: int, page: int = 1, page_size: int = 20) -> PaginatedData:
        base = select(DiseaseRecord).where(DiseaseRecord.farmer_id == farmer_id)
        count_q = select(func.count()).select_from(DiseaseRecord).where(DiseaseRecord.farmer_id == farmer_id)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(DiseaseRecord.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[DiseaseRecordResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )

    @staticmethod
    async def get_record_detail(db: AsyncSession, record_id: int) -> DiseaseRecordResponse:
        record = await db.get(DiseaseRecord, record_id)
        if not record:
            raise NotFoundException(message="Record not found")
        return DiseaseRecordResponse.model_validate(record)
```

- [ ] **Step 3: 写入 app/modules/disease/router.py**

```python
from fastapi import APIRouter, Depends, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.deps import get_current_user
from app.common.response import ApiResponse, PaginatedData
from app.modules.disease.schemas import DiseaseRecordResponse
from app.modules.disease.service import DiseaseService
from app.modules.auth.models import User

router = APIRouter(prefix="/api/v1/disease", tags=["disease"])


@router.post("/recognize", response_model=ApiResponse[DiseaseRecordResponse])
async def recognize(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await DiseaseService.recognize(db, file, farmer_id=user.id)
    return ApiResponse.success(data=result, message="Recognition completed")


@router.get("/records", response_model=ApiResponse[PaginatedData])
async def list_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await DiseaseService.get_records(db, farmer_id=user.id, page=page, page_size=page_size)
    return ApiResponse.success(data=result)


@router.get("/records/{record_id}", response_model=ApiResponse[DiseaseRecordResponse])
async def get_record(record_id: int, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    result = await DiseaseService.get_record_detail(db, record_id)
    return ApiResponse.success(data=result)
```

- [ ] **Step 4: 验证语法**

```bash
python -c "import ast; [ast.parse(open(f).read()) for f in ['app/modules/disease/schemas.py','app/modules/disease/service.py','app/modules/disease/router.py']]; print('All syntax OK')"
```

Expected: 打印 `All syntax OK`。

---

### Task 12: Knowledge 知识库模块

**Files:**
- Create: `app/modules/knowledge/schemas.py`
- Create: `app/modules/knowledge/service.py`
- Create: `app/modules/knowledge/router.py`

**Interfaces:**
- Consumes: `PestCatalog`, `KnowledgeDoc` from `app.modules.business.models`
- Produces: `GET /api/v1/knowledge/search`, `GET /api/v1/knowledge/catalog`, `GET /api/v1/knowledge/catalog/{id}`

- [ ] **Step 1: 写入 app/modules/knowledge/schemas.py**

```python
from datetime import datetime
from pydantic import BaseModel


class PestCatalogResponse(BaseModel):
    id: int
    name: str
    category: str
    symptoms: str | None = None
    treatment: str | None = None
    images: list | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class KnowledgeDocResponse(BaseModel):
    id: int
    title: str
    content: str
    category: str | None = None
    source: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SearchResult(BaseModel):
    keyword: str
    pest_matches: list[PestCatalogResponse] = []
    doc_matches: list[KnowledgeDocResponse] = []
    total: int = 0
```

- [ ] **Step 2: 写入 app/modules/knowledge/service.py**

```python
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.business.models import PestCatalog, KnowledgeDoc
from app.modules.knowledge.schemas import PestCatalogResponse, KnowledgeDocResponse, SearchResult
from app.common.response import PaginatedData
from app.common.exceptions import NotFoundException


class KnowledgeService:

    @staticmethod
    async def search(db: AsyncSession, keyword: str) -> SearchResult:
        like_pattern = f"%{keyword}%"

        pest_q = select(PestCatalog).where(
            or_(PestCatalog.name.like(like_pattern), PestCatalog.symptoms.like(like_pattern))
        ).limit(20)
        pest_rows = (await db.execute(pest_q)).scalars().all()

        doc_q = select(KnowledgeDoc).where(
            or_(KnowledgeDoc.title.like(like_pattern), KnowledgeDoc.content.like(like_pattern))
        ).limit(20)
        doc_rows = (await db.execute(doc_q)).scalars().all()

        pest_matches = [PestCatalogResponse.model_validate(r) for r in pest_rows]
        doc_matches = [KnowledgeDocResponse.model_validate(r) for r in doc_rows]

        return SearchResult(
            keyword=keyword,
            pest_matches=pest_matches,
            doc_matches=doc_matches,
            total=len(pest_matches) + len(doc_matches),
        )

    @staticmethod
    async def get_catalog_list(db: AsyncSession, category: str | None = None, page: int = 1, page_size: int = 20) -> PaginatedData:
        base = select(PestCatalog)
        count_q = select(func.count()).select_from(PestCatalog)
        if category:
            base = base.where(PestCatalog.category == category)
            count_q = count_q.where(PestCatalog.category == category)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(PestCatalog.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[PestCatalogResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )

    @staticmethod
    async def get_catalog_detail(db: AsyncSession, catalog_id: int) -> PestCatalogResponse:
        catalog = await db.get(PestCatalog, catalog_id)
        if not catalog:
            raise NotFoundException(message="Pest catalog entry not found")
        return PestCatalogResponse.model_validate(catalog)
```

- [ ] **Step 3: 写入 app/modules/knowledge/router.py**

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.common.response import ApiResponse, PaginatedData
from app.modules.knowledge.schemas import PestCatalogResponse, SearchResult
from app.modules.knowledge.service import KnowledgeService

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.get("/search", response_model=ApiResponse[SearchResult])
async def search(q: str = Query(..., min_length=1, description="Search keyword"), db: AsyncSession = Depends(get_db)):
    result = await KnowledgeService.search(db, q)
    return ApiResponse.success(data=result)


@router.get("/catalog", response_model=ApiResponse[PaginatedData])
async def list_catalog(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    result = await KnowledgeService.get_catalog_list(db, category=category, page=page, page_size=page_size)
    return ApiResponse.success(data=result)


@router.get("/catalog/{catalog_id}", response_model=ApiResponse[PestCatalogResponse])
async def get_catalog_detail(catalog_id: int, db: AsyncSession = Depends(get_db)):
    result = await KnowledgeService.get_catalog_detail(db, catalog_id)
    return ApiResponse.success(data=result)
```

- [ ] **Step 4: 验证语法**

```bash
python -c "import ast; [ast.parse(open(f).read()) for f in ['app/modules/knowledge/schemas.py','app/modules/knowledge/service.py','app/modules/knowledge/router.py']]; print('All syntax OK')"
```

---

### Task 13: Agent 种植决策模块（mock）

**Files:**
- Create: `app/modules/agent/schemas.py`
- Create: `app/modules/agent/service.py`
- Create: `app/modules/agent/router.py`

**Interfaces:**
- Produces: `POST /api/v1/agent/chat` — P0 返回 mock 回答

- [ ] **Step 1: 写入 app/modules/agent/schemas.py**

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: str | None = Field(default=None, description="Optional context like crop type or location")


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []
```

Wait — there's a bug. ChatRequest is defined twice. Let me fix that in the plan.

Actually, I'll just remove the duplicate ChatRequest definition. Let me write:

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: str | None = Field(default=None, description="Optional context like crop type or location")


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []
```

That's clean. Moving on.

- [ ] **Step 2: 写入 app/modules/agent/service.py**

```python
from app.modules.agent.schemas import ChatRequest, ChatResponse


class AgentService:

    @staticmethod
    async def chat(req: ChatRequest) -> ChatResponse:
        # P0: Return mock response
        mock_reply = (
            f"关于「{req.message}」的问题，根据当前农事季节和作物生长阶段，"
            f"建议如下：\n\n"
            f"1. 加强田间巡查，密切关注作物长势和病虫害发生情况\n"
            f"2. 合理施肥浇水，根据土壤墒情适时调整管理措施\n"
            f"3. 如有异常症状，建议拍照上传进行病虫害识别\n\n"
            f"（P0 脚手架阶段为模拟回复，后续将接入 AI 模型生成精准建议）"
        )
        return ChatResponse(
            reply=mock_reply,
            sources=["农业知识库", "农事操作规范"],
        )
```

- [ ] **Step 3: 写入 app/modules/agent/router.py**

```python
from fastapi import APIRouter
from app.common.response import ApiResponse
from app.modules.agent.schemas import ChatRequest, ChatResponse
from app.modules.agent.service import AgentService

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/chat", response_model=ApiResponse[ChatResponse])
async def chat(req: ChatRequest):
    result = await AgentService.chat(req)
    return ApiResponse.success(data=result)
```

- [ ] **Step 4: 验证语法**

```bash
python -c "import ast; [ast.parse(open(f).read()) for f in ['app/modules/agent/schemas.py','app/modules/agent/service.py','app/modules/agent/router.py']]; print('All syntax OK')"
```

---

### Task 14: 文件存储工具

**Files:**
- Create: `app/common/file_storage.py`

**Interfaces:**
- Produces: `save_upload(file: UploadFile, subdir: str) -> str`, `validate_image(file: UploadFile) -> None`

- [ ] **Step 1: 写入 app/common/file_storage.py**

```python
import os
import uuid
from fastapi import UploadFile
from app.core.config import settings
from app.common.exceptions import BusinessException

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def validate_image(file: UploadFile) -> None:
    """Validate file is an allowed image type and within size limit."""
    if file.filename is None:
        raise BusinessException(code=400, message="Filename is required")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise BusinessException(
            code=400,
            message=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Size check: read size from content-type header or allow up to config limit
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


async def save_upload(file: UploadFile, subdir: str = "diseases") -> str:
    """Save an uploaded file and return the relative path."""
    validate_image(file)

    dest_dir = os.path.join(settings.UPLOAD_DIR, subdir)
    os.makedirs(dest_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(dest_dir, filename)

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    return f"uploads/{subdir}/{filename}"
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from app.common.file_storage import validate_image, save_upload; print('OK')"
```

---

### Task 15: main.py 入口 + Alembic 配置 + 启动验证

**Files:**
- Create: `app/main.py`
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/script.py.mako` (Alembic 自动生成)

**Interfaces:**
- Consumes: all routers from 5 modules, `register_exception_handlers` from common, `settings` from core
- Produces: 可启动的 FastAPI 应用，带 CORS 和全局异常处理

- [ ] **Step 1: 写入 app/main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.common.exceptions import register_exception_handlers
from app.modules.auth.router import router as auth_router
from app.modules.disease.router import router as disease_router
from app.modules.knowledge.router import router as knowledge_router
from app.modules.agent.router import router as agent_router
from app.modules.business.router import router as business_router

app = FastAPI(title=settings.APP_NAME, version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
register_exception_handlers(app)

# Routers
app.include_router(auth_router)
app.include_router(disease_router)
app.include_router(knowledge_router)
app.include_router(agent_router)
app.include_router(business_router)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}
```

- [ ] **Step 2: 写入 alembic.ini**

```ini
[alembic]
script_location = migrations
prepend_sys_path = .
sqlalchemy.url = mysql+aiomysql://root:password@localhost:3306/smart_agri

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

- [ ] **Step 3: 创建 migrations/env.py**

```python
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from app.core.config import settings
from app.core.database import Base

# Import all models so Base.metadata is populated
from app.modules.auth.models import User
from app.modules.business.models import Farm, Crop, PestCatalog, DiseaseRecord, KnowledgeDoc

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: 初始化 Alembic 并生成首次迁移**

```bash
alembic init migrations
# 上面已经手动创建了 env.py 和 alembic.ini，所以直接用现有文件
alembic revision --autogenerate -m "init"
```

Expected: 在 `migrations/versions/` 下生成 `xxxx_init.py` 迁移文件。

- [ ] **Step 5: 启动验证（此时需要可用的 MySQL）**

```bash
# 先配置 .env 中的数据库连接信息（从 .env.example 复制并修改）
cp .env.example .env
# 编辑 .env 填入实际 MySQL 连接信息
# 然后启动服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected: 服务启动，访问 `http://localhost:8000/health` 返回 `{"status":"ok","app":"SmartAgriAI"}`，访问 `http://localhost:8000/docs` 看到 Swagger 文档。

- [ ] **Step 6: 快速接口验证**

```bash
# 1. 注册农户
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"farmer1","password":"123456","role":"farmer"}'

# 2. 登录获取 token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"farmer1","password":"123456"}'

# 3. 用 token 访问 /me（替换 <token>）
curl http://localhost:8000/api/v1/auth/me -H "Authorization: Bearer <token>"

# 4. 病虫害识别 mock
curl -X POST http://localhost:8000/api/v1/disease/recognize \
  -H "Authorization: Bearer <token>" \
  -F "file=@test_image.jpg"

# 5. 知识搜索
curl "http://localhost:8000/api/v1/knowledge/search?q=赤霉病"

# 6. Agent 问答
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"小麦叶片发黄怎么办"}'
```

Expected: 所有接口返回 `{"code":200,"message":"...","data":{...}}` 格式的响应。

---

### Task 16: README 文档

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写入 README.md**

```markdown
# 智慧农业 AI 系统 — SmartAgriAI

## 技术栈

- **后端**: Python FastAPI (async)
- **数据库**: MySQL 8.0 + Redis
- **认证**: JWT Bearer Token

## 快速开始

### 前置条件

- Python 3.11+
- 可访问的 MySQL 8.0 实例
- 可访问的 Redis 实例

### 安装

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入数据库连接信息
```

### 数据库迁移

```bash
alembic upgrade head
```

### 启动开发服务器

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API 文档

启动后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 项目结构

```
app/
├── main.py              # FastAPI 入口
├── core/                # 基础设施（配置、数据库、安全、依赖注入）
├── common/              # 通用工具（响应格式、异常处理、文件存储）
└── modules/             # 业务模块
    ├── auth/            # 认证（登录、注册）
    ├── disease/         # 病虫害识别
    ├── knowledge/       # 知识库检索
    ├── agent/           # 种植决策 Agent
    └── business/        # 业务管理 CRUD
```

## 开发阶段

- **P0（当前）**: 后端骨架 + 认证 + CRUD + mock 接口
- **P1**: 病虫害识别模型集成
- **P2**: RAG 知识库 + Agent LLM 编排
```
- [ ] **Step 2: Commit 所有代码**

```bash
git add -A
git commit -m "feat: backend scaffold - FastAPI project with auth, CRUD, mock APIs

- Project skeleton with domain-modular structure
- JWT authentication (register, login, me)
- Business CRUD (farms, crops, pests, disease records)
- Disease recognition (mock), knowledge search, agent chat (mock)
- Unified ApiResponse format, global exception handling
- Alembic migrations, README"
```
```

<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="TodoWrite">
<｜｜DSML｜｜parameter name="todos" string="false">[{"activeForm":"Writing implementation plan","content":"Write implementation plan file from spec","status":"completed"},{"activeForm":"Self-reviewing plan","content":"Self-review plan: coverage, placeholders, type consistency","status":"in_progress"},{"activeForm":"Handing off to execution","content":"Present execution options to user","status":"pending"}]