from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.business.models import Field, Crop, Pest, FieldPestRecord
from app.modules.business.schemas import (
    FieldCreate, FieldUpdate, FieldResponse,
    CropCreate, CropUpdate, CropResponse,
    PestCreate, PestUpdate, PestResponse,
    FieldPestRecordResponse,
)
from app.common.exceptions import NotFoundException
from app.common.response import PaginatedData


class BusinessService:

    # ---- Field ----
    @staticmethod
    async def create_field(db: AsyncSession, data: FieldCreate, farmer_id: int) -> FieldResponse:
        field = Field(farmer_id=farmer_id, **data.model_dump())
        db.add(field)
        await db.commit()
        await db.refresh(field)
        return FieldResponse.model_validate(field)

    @staticmethod
    async def get_fields(db: AsyncSession, page: int = 1, page_size: int = 20, farmer_id: int | None = None) -> PaginatedData:
        base = select(Field)
        count_q = select(func.count()).select_from(Field)
        if farmer_id:
            base = base.where(Field.farmer_id == farmer_id)
            count_q = count_q.where(Field.farmer_id == farmer_id)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(Field.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[FieldResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )

    @staticmethod
    async def update_field(db: AsyncSession, field_id: int, data: FieldUpdate) -> FieldResponse:
        field = await db.get(Field, field_id)
        if not field:
            raise NotFoundException(message="Field not found")
        for k, v in data.model_dump(exclude_unset=True).items():
            setattr(field, k, v)
        await db.commit()
        await db.refresh(field)
        return FieldResponse.model_validate(field)

    # ---- Crop ----
    @staticmethod
    async def create_crop(db: AsyncSession, data: CropCreate) -> CropResponse:
        crop = Crop(**data.model_dump())
        db.add(crop)
        await db.commit()
        await db.refresh(crop)
        return CropResponse.model_validate(crop)

    @staticmethod
    async def get_crops(db: AsyncSession, page: int = 1, page_size: int = 20) -> PaginatedData:
        count_q = select(func.count()).select_from(Crop)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(select(Crop).order_by(Crop.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
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

    # ---- Pest ----
    @staticmethod
    async def create_pest(db: AsyncSession, data: PestCreate) -> PestResponse:
        pest = Pest(**data.model_dump())
        db.add(pest)
        await db.commit()
        await db.refresh(pest)
        return PestResponse.model_validate(pest)

    @staticmethod
    async def get_pests(db: AsyncSession, pest_type: str | None = None, page: int = 1, page_size: int = 20) -> PaginatedData:
        base = select(Pest)
        count_q = select(func.count()).select_from(Pest)
        if pest_type:
            base = base.where(Pest.type == pest_type)
            count_q = count_q.where(Pest.type == pest_type)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(Pest.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[PestResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )

    @staticmethod
    async def update_pest(db: AsyncSession, pest_id: int, data: PestUpdate) -> PestResponse:
        pest = await db.get(Pest, pest_id)
        if not pest:
            raise NotFoundException(message="Pest not found")
        for k, v in data.model_dump(exclude_unset=True).items():
            setattr(pest, k, v)
        await db.commit()
        await db.refresh(pest)
        return PestResponse.model_validate(pest)

    @staticmethod
    async def delete_pest(db: AsyncSession, pest_id: int) -> None:
        pest = await db.get(Pest, pest_id)
        if not pest:
            raise NotFoundException(message="Pest not found")
        await db.delete(pest)
        await db.commit()

    # ---- FieldPestRecord ----
    @staticmethod
    async def get_field_pest_records(db: AsyncSession, severity: str | None = None, page: int = 1, page_size: int = 20) -> PaginatedData:
        base = select(FieldPestRecord)
        count_q = select(func.count()).select_from(FieldPestRecord)
        if severity:
            base = base.where(FieldPestRecord.severity == severity)
            count_q = count_q.where(FieldPestRecord.severity == severity)
        total = (await db.execute(count_q)).scalar()
        rows = (await db.execute(base.order_by(FieldPestRecord.id.desc()).offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return PaginatedData(
            items=[FieldPestRecordResponse.model_validate(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )
