from app.modules.agent.schemas import ChatRequest, ChatResponse


class AgentService:

    @staticmethod
    async def chat(req: ChatRequest) -> ChatResponse:
        # P0: Return mock response
        mock_reply = (
            f"关于「{req.message}」的问题，根据当前农事季节和作物生长阶段，"
            f"建议如下：\n\n"
            f"1. 加强田间巡查，密切关注作物长势和病虫害发生情况\n"
            f"2. 合理施肥浇水，根据土壤墒情适时调整管理措施\n"
            f"3. 如有异常症状，建议拍照上传进行病虫害识别\n\n"
            f"（P0 脚手架阶段为模拟回复，后续将接入 AI 模型生成精准建议）"
        )
        return ChatResponse(
            reply=mock_reply,
            sources=["农业知识库", "农事操作规范"],
        )
