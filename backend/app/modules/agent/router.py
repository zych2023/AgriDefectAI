from fastapi import APIRouter
from app.common.response import ApiResponse
from app.modules.agent.schemas import ChatRequest, ChatResponse
from app.modules.agent.service import AgentService

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/chat", response_model=ApiResponse[ChatResponse])
async def chat(req: ChatRequest):
    result = await AgentService.chat(req)
    return ApiResponse.success(data=result)
