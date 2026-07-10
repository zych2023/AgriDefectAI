from datetime import datetime
from sqlalchemy import String, Text, Float, DateTime, ForeignKey, JSON, func
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    farmer_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("farmers.id", ondelete="CASCADE"),
        nullable=False, index=True, comment="农户ID"
    )
    image_url: Mapped[str] = mapped_column(String(500), nullable=False, comment="上传图片路径")
    disease: Mapped[str] = mapped_column(String(100), nullable=False, comment="识别病害名")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, comment="置信度")
    top5_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="Top-5完整结果")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
