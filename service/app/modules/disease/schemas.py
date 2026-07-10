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
    id: int
    field_id: int
    pest_id: int
    detected_at: datetime
    severity: str
    top_prediction: PredictionItem
    top5: list[PredictionItem]
    notes: str | None = None


class FieldPestRecordResponse(BaseModel):
    id: int
    field_id: int
    pest_id: int
    detected_at: datetime
    severity: str
    notes: str | None = None

    class Config:
        from_attributes = True
