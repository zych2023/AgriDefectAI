from datetime import datetime
from pydantic import BaseModel


class PredictionItem(BaseModel):
    rank: int
    class_id: int
    disease: str
    confidence: float
    advice: str = ""
    pesticide: str = ""


class RecognizeResponse(BaseModel):
    id: int | None = None  # recognition_logs 记录ID，未登录时为None
    top_prediction: PredictionItem
    top5: list[PredictionItem]
    created_at: datetime | None = None


class RecognitionLogItem(BaseModel):
    id: int
    image_url: str
    disease: str
    confidence: float
    top5: list[PredictionItem] | None = None
    created_at: datetime

    class Config:
        from_attributes = True
