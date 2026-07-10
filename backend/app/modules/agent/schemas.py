from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: str | None = Field(default=None, description="Optional context like crop type or location")


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []
