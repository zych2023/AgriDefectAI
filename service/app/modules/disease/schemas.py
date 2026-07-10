from pydantic import BaseModel


class PredictionItem(BaseModel):
    rank: int
    class_id: int
    disease: str
    confidence: float
    advice: str = ""
    pesticide: str = ""


class RecognizeResponse(BaseModel):
    top_prediction: PredictionItem
    top5: list[PredictionItem]
