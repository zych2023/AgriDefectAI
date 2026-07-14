"""
=============================================================================
智慧农业 RAG 问答接口 — FastAPI HTTP 服务
=============================================================================
启动后提供 POST /ask 接口，接收问题 JSON，返回 AI 回答文本。

启动方式：
  python api_server.py
  python api_server.py --port 8080
  python api_server.py --host 0.0.0.0 --port 8080

默认监听 0.0.0.0:8899（所有网卡）
"""

import argparse
import logging
import sys
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import config
from milvus_client import MilvusHybridClient
from main import QueryPipeline
import qa_logger

# ============================================================================
# 日志
# ============================================================================
logging.basicConfig(
    level=getattr(logging, config.log.level),
    format=config.log.format,
    datefmt=config.log.datefmt,
)
logger = logging.getLogger("agri_api")

# ============================================================================
# 全局管线（启动时初始化，所有请求复用）
# ============================================================================
_pipeline: Optional[QueryPipeline] = None
_milvus_client: Optional[MilvusHybridClient] = None


def init_pipeline() -> QueryPipeline:
    """初始化 RAG 管线：连接 Milvus → 加载 Collection → 恢复 BM25"""
    global _milvus_client, _pipeline

    logger.info("正在初始化 RAG 管线...")
    _milvus_client = MilvusHybridClient()
    _milvus_client.connect()
    _milvus_client.init_collection(drop_if_exists=False)

    stats = _milvus_client.collection_stats()
    count = stats.get("num_entities", 0)
    if count == 0:
        raise RuntimeError(
            "知识库为空！请先运行 python main.py --ingest-only 构建知识库。"
        )
    logger.info("知识库已就绪: %d 条记录", count)

    _milvus_client.load_corpus_and_fit_bm25()
    _pipeline = QueryPipeline(_milvus_client)

    # 初始化问答日志库
    qa_logger.init_db()

    logger.info("RAG 管线初始化完成")
    return _pipeline


# ============================================================================
# FastAPI 应用
# ============================================================================
app = FastAPI(
    title="智慧农业AI问答接口",
    description="基于 Advanced RAG 的农业病虫害识别与种植决策问答服务",
    version="1.0.0",
)

# CORS（允许跨域调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 请求 / 响应模型
# ============================================================================
class AskRequest(BaseModel):
    fid: str
    question: str

class AskResponse(BaseModel):
    answer: str
    success: bool = True

class HistoryItem(BaseModel):
    id: int
    fid: str
    question: str
    answer: str
    success: bool
    created_at: str

class HistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[HistoryItem]

class RetrieveDoc(BaseModel):
    content: str
    source_file: str
    page: int
    score: float

class RetrieveResponse(BaseModel):
    documents: List[RetrieveDoc]


# ============================================================================
# 接口
# ============================================================================
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """农业知识问答接口 —— 输入问题，返回回答"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if not req.fid or not req.fid.strip():
        raise HTTPException(status_code=400, detail="fid 不能为空")

    logger.info("收到问题 [farmer=%s]: %s", req.fid, question[:60])

    try:
        result = _pipeline.answer(question)
        qa_logger.save(req.fid, question, result.answer, success=True)
    except Exception as e:
        logger.error("问答处理失败: %s", e, exc_info=True)
        qa_logger.save(req.fid, question, "", success=False, error_msg=str(e))
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    return AskResponse(answer=result.answer)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: AskRequest) -> RetrieveResponse:
    """知识检索接口 —— 返回相关知识文档，不做 LLM 生成。供外部 Agent 调用。"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if not req.fid or not req.fid.strip():
        raise HTTPException(status_code=400, detail="fid 不能为空")

    logger.info("检索请求 [fid=%s]: %s", req.fid, question[:60])

    try:
        docs = _pipeline.retrieve(question)
    except Exception as e:
        logger.error("检索失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

    return RetrieveResponse(documents=[
        RetrieveDoc(
            content=d.get("content", ""),
            source_file=d.get("source_file", ""),
            page=d.get("page", -1),
            score=round(d.get("rerank_score", d.get("rrf_score", 0.0)), 6),
        )
        for d in docs
    ])


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "records": _milvus_client.collection_stats().get("num_entities", 0),
        "qa_history_count": qa_logger.total_count(),
    }


@app.get("/history", response_model=HistoryResponse)
async def history(
    fid: Optional[str] = None, page: int = 1, page_size: int = 20
) -> HistoryResponse:
    """分页查询问答历史（可按 fid 过滤，不传则查全部）"""
    total = qa_logger.total_count(fid=fid or None)
    items = qa_logger.get_list(fid=fid or None, page=page, page_size=page_size)
    return HistoryResponse(total=total, page=page, page_size=page_size, items=items)


@app.get("/history/search", response_model=HistoryResponse)
async def history_search(
    keyword: str,
    fid: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> HistoryResponse:
    """按关键词搜索问答历史（可按 fid 过滤）"""
    total = qa_logger.search_count(keyword, fid=fid or None)
    items = qa_logger.search(keyword, fid=fid or None, page=page, page_size=page_size)
    return HistoryResponse(total=total, page=page, page_size=page_size, items=items)


# ============================================================================
# 启动入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="农业 RAG 问答接口服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0，监听所有网卡）")
    parser.add_argument("--port", type=int, default=8899, help="监听端口（默认 8899）")
    args = parser.parse_args()

    # 初始化
    init_pipeline()

    logger.info("启动 API 服务: http://%s:%d", args.host, args.port)
    logger.info("接口文档: http://%s:%d/docs", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
