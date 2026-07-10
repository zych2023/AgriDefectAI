from datetime import datetime
from sqlalchemy import String, Integer, BigInteger, Text, DateTime, DECIMAL, func
from sqlalchemy.dialects.mysql import TINYINT, ENUM
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Farmer(Base):
    __tablename__ = "farmers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="农户ID")
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="登录账号")
    password: Mapped[str] = mapped_column(String(60), nullable=False, comment="密码哈希(bcrypt)")
    real_name: Mapped[str] = mapped_column(String(30), nullable=False, comment="真实姓名")
    phone: Mapped[str | None] = mapped_column(String(20), unique=True, nullable=True, comment="手机号")
    status: Mapped[int] = mapped_column(TINYINT, nullable=False, default=1, comment="账户状态 0-禁用 1-启用")
    registered_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="注册时间")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间"
    )


class Expert(Base):
    __tablename__ = "experts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="专家ID")
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="登录账号")
    password: Mapped[str] = mapped_column(String(60), nullable=False, comment="密码哈希")
    real_name: Mapped[str] = mapped_column(String(30), nullable=False, comment="真实姓名")
    phone: Mapped[str | None] = mapped_column(String(20), unique=True, nullable=True, comment="手机号")
    specialty: Mapped[str | None] = mapped_column(String(200), nullable=True, comment="专长领域")
    region: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="服务区域")
    status: Mapped[int] = mapped_column(TINYINT, nullable=False, default=1, comment="0-禁用 1-启用")
    role: Mapped[int] = mapped_column(TINYINT, nullable=False, default=1, comment="1-普通专家 2-高级专家/管理员")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
