from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile
from app.common.file_storage import save_upload
from app.modules.business.models import FieldPestRecord, Field
from app.modules.disease.schemas import RecognizeResult, FieldPestRecordResponse
from app.common.response import PaginatedData
from app.common.exceptions import NotFoundException, BusinessException


class DiseaseService:

    @staticmethod
    async def recognize(
        db: AsyncSession, file: UploadFile, field_id: int, user_id: int
    ) -> FieldPestRecordResponse:
        # Verify field belongs to user
        field = await db.get(Field, field_id)
        if not field:
            raise NotFoundException(message="Field not found")
        if field.farmer_id != user_id:
            raise BusinessException(code=403, message="Access denied to this field")

        # Save uploaded image
        await save_upload(file, sub_dir="diseases")

        # P0: Mock recognition result
        mock_result = RecognizeResult(
            pest_name="小麦赤霉病 (Fusarium Head Blight)",
            pest_id=1,
            type="disease",
            confidence=0.92,
            severity="moderate",
            description="叶片出现水渍状斑点，逐渐扩大为黄褐色枯斑",
            prevention="建议使用戊唑醇悬浮剂2000倍液喷雾防治，7-10天一次，连续2-3次",
        )

        record = FieldPestRecord(
            field_id=field_id,
            pest_id=mock_result.pest_id,
            severity=mock_result.severity,
            notes=f"[Mock P0] {mock_result.description} | Prevention: {mock_result.prevention}",
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return FieldPestRecordResponse.model_validate(record)

    @staticmethod
    async def get_records(
        db: AsyncSession, user_id: int, page: int = 1, page_size: int = 20
    ) -> PaginatedData:
        # Get fields owned by this user
        field_ids = (
            await db.execute(select(Field.id).where(Field.farmer_id == user_id))
        ).scalars().all()

        base = select(FieldPestRecord).where(FieldPestRecord.field_id.in_(field_ids))
        count_q = (
            select(func.count())
            .select_from(FieldPestRecord)
            .where(FieldPestRecord.field_id.in_(field_ids))
        )
        total = (await db.execute(count_q)).scalar()
        rows = (
            await db.execute(
                base.order_by(FieldPestRecord.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        ).scalars().all()
        return PaginatedData(
            items=[FieldPestRecordResponse.model_validate(r) for r in rows],
            total=total,
            page=page,
            page_size=page_size,
        )

    @staticmethod
    async def get_record_detail(
        db: AsyncSession, record_id: int, user_id: int
    ) -> FieldPestRecordResponse:
        record = await db.get(FieldPestRecord, record_id)
        if not record:
            raise NotFoundException(message="Record not found")

        # Verify field ownership
        field = await db.get(Field, record.field_id)
        if not field or field.farmer_id != user_id:
            raise NotFoundException(message="Record not found")

        return FieldPestRecordResponse.model_validate(record)
