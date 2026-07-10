from datetime import datetime
from pydantic import BaseModel


class PestResponse(BaseModel):
    id: int
    name: str
    type: str
    symptoms: str | None = None
    pathogen: str | None = None
    prevention: str | None = None
    example_image_url: str | None = None

    class Config:
        from_attributes = True


class KnowledgeDocResponse(BaseModel):
    id: int
    title: str
    content: str
    category: str | None = None
    source: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SearchResult(BaseModel):
    keyword: str
    pest_matches: list[PestResponse] = []
    doc_matches: list[KnowledgeDocResponse] = []
    total: int = 0
