from datetime import datetime
from pydantic import BaseModel


class RecognizeRequest(BaseModel):
    field_id: int  # Which field the image is from


class RecognizeResult(BaseModel):
    pest_name: str
    pest_id: int
    type: str  # disease, pest, weed
    confidence: float
    severity: str  # mild, moderate, severe
    description: str
    prevention: str


class FieldPestRecordResponse(BaseModel):
    id: int
    field_id: int
    pest_id: int
    detected_at: datetime
    severity: str
    notes: str | None = None

    class Config:
        from_attributes = True
