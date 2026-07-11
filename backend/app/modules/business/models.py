from datetime import datetime
from sqlalchemy import String, Integer, BigInteger, Text, Float, DECIMAL, DateTime, ForeignKey, func
from sqlalchemy.dialects.mysql import TINYINT, ENUM, INTEGER
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Field(Base):
    __tablename__ = "fields"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="农田ID")
    farmer_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("farmers.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False, index=True, comment="所属农户ID"
    )
    field_name: Mapped[str | None] = mapped_column(String(50), nullable=True, comment="地块名称/编号")
    area: Mapped[float | None] = mapped_column(DECIMAL(10, 2), nullable=True, comment="面积(亩)")
    location: Mapped[str | None] = mapped_column(String(200), nullable=True, comment="位置描述")
    soil_type: Mapped[str | None] = mapped_column(String(50), nullable=True, comment="土壤类型(沙土/黏土/壤土等)")
    current_crop_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("crops.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True, comment="当前种植作物ID"
    )
    status: Mapped[str] = mapped_column(
        ENUM("fallow", "planted", "harvested", "resting"), nullable=False, default="fallow",
        comment="田地状态(休耕/已种植/已收获/休整)"
    )
    has_pest_disease: Mapped[int] = mapped_column(TINYINT(1), nullable=False, default=0, comment="是否有病虫害 0-无 1-有")
    remarks: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="备注")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Crop(Base):
    __tablename__ = "crops"

    id: Mapped[int] = mapped_column(INTEGER(unsigned=True), primary_key=True, autoincrement=True, comment="作物ID")
    name: Mapped[str] = mapped_column(String(50), nullable=False, comment="作物名称")
    variety: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="品种")
    growth_cycle: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="生长周期(天)")
    description: Mapped[str | None] = mapped_column(Text, nullable=True, comment="描述")


class Pest(Base):
    __tablename__ = "pests"

    id: Mapped[int] = mapped_column(INTEGER(unsigned=True), primary_key=True, autoincrement=True, comment="病虫害ID")
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="名称")
    type: Mapped[str] = mapped_column(
        ENUM("disease", "pest", "weed"), nullable=False, default="disease",
        comment="类型(病害/虫害/草害)"
    )
    symptoms: Mapped[str | None] = mapped_column(Text, nullable=True, comment="典型症状描述")
    pathogen: Mapped[str | None] = mapped_column(String(200), nullable=True, comment="病原/学名")
    prevention: Mapped[str | None] = mapped_column(Text, nullable=True, comment="防治方法")
    example_image_url: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="示例图片")


class FieldPestRecord(Base):
    __tablename__ = "field_pest_records"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    field_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True, comment="农田ID"
    )
    pest_id: Mapped[int] = mapped_column(
        INTEGER(unsigned=True), ForeignKey("pests.id", ondelete="CASCADE"), nullable=False, index=True, comment="病虫害ID"
    )
    detected_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="发现时间")
    severity: Mapped[str] = mapped_column(
        ENUM("mild", "moderate", "severe"), default="mild", comment="严重程度(轻度/中度/重度)"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, comment="备注")


class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
