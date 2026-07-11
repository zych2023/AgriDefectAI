import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

# Import DiseasePredictor from model-training
_MODEL_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "model-training"
sys.path.insert(0, str(_MODEL_ROOT / "api"))
from infer import DiseasePredictor

from app.modules.disease.models import RecognitionLog
from app.modules.disease.schemas import PredictionItem, RecognizeResponse, RecognitionLogItem
from app.common.response import PaginatedData
from app.common.exceptions import NotFoundException

# Lazy-load predictor
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
        file: UploadFile,
        db: AsyncSession | None = None,
        farmer_id: int | None = None,
    ) -> RecognizeResponse:
        # Read image bytes
        contents = await file.read()

        # Run inference
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        predictor = get_predictor()
        result = predictor.predict(image_np)

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

        log_id = None
        created_at = None

        # Save to history if user is logged in
        if db is not None and farmer_id is not None:
            top = result["top_prediction"]
            image_url = await _save_upload_image(contents, file.filename or "image.jpg")

            log = RecognitionLog(
                farmer_id=farmer_id,
                image_url=image_url,
                disease=top["disease"],
                confidence=top["confidence"],
                top5_json={"top5": [item.model_dump() for item in top5]},
            )
            db.add(log)
            await db.commit()
            await db.refresh(log)
            log_id = log.id
            created_at = log.created_at

        return RecognizeResponse(
            id=log_id,
            top_prediction=top5[0],
            top5=top5,
            created_at=created_at,
        )

    @staticmethod
    async def get_records(
        db: AsyncSession, farmer_id: int, page: int = 1, page_size: int = 20
    ) -> PaginatedData:
        base = select(RecognitionLog).where(RecognitionLog.farmer_id == farmer_id)
        count_q = (
            select(func.count())
            .select_from(RecognitionLog)
            .where(RecognitionLog.farmer_id == farmer_id)
        )
        total = (await db.execute(count_q)).scalar()
        rows = (
            await db.execute(
                base.order_by(RecognitionLog.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        ).scalars().all()

        items = []
        for r in rows:
            top5 = None
            if r.top5_json and "top5" in r.top5_json:
                top5 = [PredictionItem(**item) for item in r.top5_json["top5"]]
            items.append(RecognitionLogItem(
                id=r.id,
                image_url=r.image_url,
                disease=r.disease,
                confidence=r.confidence,
                top5=top5,
                created_at=r.created_at,
            ))

        return PaginatedData(items=items, total=total, page=page, page_size=page_size)


async def _save_upload_image(contents: bytes, filename: str) -> str:
    import uuid, os
    from app.core.config import settings

    ext = os.path.splitext(filename)[1] or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(settings.UPLOAD_DIR, "diseases")
    os.makedirs(dest, exist_ok=True)
    filepath = os.path.join(dest, fname)
    with open(filepath, "wb") as f:
        f.write(contents)
    return f"uploads/diseases/{fname}"
