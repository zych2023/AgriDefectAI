import sys
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import UploadFile

# Import DiseasePredictor from model-training
_MODEL_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "model-training"
sys.path.insert(0, str(_MODEL_ROOT / "api"))
from infer import DiseasePredictor

from app.modules.disease.schemas import PredictionItem, RecognizeResponse

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
    async def recognize(file: UploadFile) -> RecognizeResponse:
        # Read image bytes and run inference
        import io
        contents = await file.read()
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

        return RecognizeResponse(top_prediction=top5[0], top5=top5)
