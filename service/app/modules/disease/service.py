import sys
from pathlib import Path
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile
import numpy as np
from PIL import Image

# Import DiseasePredictor from model-training
_MODEL_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "model-training"
sys.path.insert(0, str(_MODEL_ROOT / "api"))
from infer import DiseasePredictor

from app.common.file_storage import save_upload
from app.modules.business.models import FieldPestRecord, Field
from app.modules.disease.schemas import PredictionItem, RecognizeResponse, FieldPestRecordResponse
from app.common.response import PaginatedData
from app.common.exceptions import NotFoundException, BusinessException

# Lazy-load predictor (loads model on first request)
_predictor: DiseasePredictor | None = None


def get_predictor() -> DiseasePredictor:
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor(
            model_path=str(_MODEL_ROOT / "checkpoints" / "model.onnx"),
            backend="onnx",
        )
    return _predictor


class DiseaseService:

    @staticmethod
    async def recognize(
        db: AsyncSession, file: UploadFile, field_id: int, user_id: int
    ) -> RecognizeResponse:
        # Verify field belongs to user
        field = await db.get(Field, field_id)
        if not field:
            raise NotFoundException(message="Field not found")
        if field.farmer_id != user_id:
            raise BusinessException(code=403, message="Access denied to this field")

        # Save to disk, then read back for inference
        image_path = await save_upload(file, sub_dir="diseases")

        # Read saved image for inference
        image = Image.open(Path(image_path)).convert("RGB")
        image_np = np.array(image)

        predictor = get_predictor()
        result = predictor.predict(image_np)

        top = result["top_prediction"]
        pest_id = top["class_id"] + 1  # class_id is 0-based, DB pest_id is 1-based

        # Map confidence to severity
        conf = top["confidence"]
        if conf >= 0.8:
            severity = "severe"
        elif conf >= 0.5:
            severity = "moderate"
        else:
            severity = "mild"

        notes = f"AI识别: {top['disease']} (置信度: {conf:.2%})"

        record = FieldPestRecord(
            field_id=field_id,
            pest_id=pest_id,
            severity=severity,
            notes=notes,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)

        # Build response
        top5 = [
            PredictionItem(
                rank=item["rank"],
                class_id=item["class_id"],
                disease=item["disease"],
                confidence=item["confidence"],
                advice=item.get("advice", ""),
                pesticide=item.get("pesticide", ""),
            )
            for item in result["top5"]
        ]

        return RecognizeResponse(
            id=record.id,
            field_id=record.field_id,
            pest_id=record.pest_id,
            detected_at=record.detected_at,
            severity=record.severity,
            top_prediction=top5[0],
            top5=top5,
            notes=record.notes,
        )

    @staticmethod
    async def get_records(
        db: AsyncSession, user_id: int, page: int = 1, page_size: int = 20
    ) -> PaginatedData:
        field_ids = (
            await db.execute(select(Field.id).where(Field.farmer_id == user_id))
        ).scalars().all()

        if not field_ids:
            return PaginatedData(items=[], total=0, page=page, page_size=page_size)

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
        field = await db.get(Field, record.field_id)
        if not field or field.farmer_id != user_id:
            raise NotFoundException(message="Record not found")
        return FieldPestRecordResponse.model_validate(record)
