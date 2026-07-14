"""
=============================================================================
智慧农业AI病虫害识别与种植决策系统 — Advanced RAG 主入口
=============================================================================
完整闭环流水线：
  加载 → 语义切分 → 向量化 → Milvus 入库（去重）
  → 用户提问 → 查询扩展 → 混合检索 → RRF 融合
  → Cross-Encoder 精排（可降级）→ LLM 流式生成（带引用） → 输出结果

用法：
  python main.py                   # 智能模式：自动检测知识库
  ·            # 直接进入对话（跳过构建）
  python main.py --rebuild         # 强制重建知识库后对话
  python main.py --ingest-only     # 仅构建知识库后退出
"""

import argparse
import logging
import sys
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from config import config
from document_loader import load_agriculture_documents
from chunker import AgriSemanticChunker
from milvus_client import MilvusHybridClient
from query_expander import QueryExpander
from retriever import HybridRetriever
from reranker import CrossEncoderReranker
from generator import AnswerGenerator, GenerationResult

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=getattr(logging, config.log.level),
    format=config.log.format,
    datefmt=config.log.datefmt,
)
logger = logging.getLogger("agri_rag")


# ============================================================================
# 阶段一：知识库构建管线
# ============================================================================

class KnowledgeBaseBuilder:
    """
    知识库构建器

    串联：文档加载 → 语义切分 → BM25 训练 → 稠密向量编码 → Milvus 入库
    """

    def __init__(self):
        self.milvus_client = MilvusHybridClient()
        self.chunker = AgriSemanticChunker()

    def build(self, rebuild: bool = True) -> MilvusHybridClient:
        """
        执行完整的知识库构建流程。

        Args:
            rebuild: 是否删除已有 Collection 后重建

        Returns:
            已连接并初始化好的 MilvusHybridClient（可用于后续检索）
        """
        logger.info("=" * 60)
        logger.info("  阶段一：知识库构建开始")
        logger.info("=" * 60)

        # ---- 1. 连接 Milvus ----
        self.milvus_client.connect()

        # ---- 2. 加载文档 ----
        logger.info("[1/6] 加载农业资料文档...")
        raw_docs = load_agriculture_documents()
        if not raw_docs:
            logger.error("未加载到任何文档，请检查 ./data/ 目录是否存在资料文件")
            raise RuntimeError("文档加载失败：data 目录为空或不存在")
        logger.info("  共加载 %d 个原始文档片段 (PDF页/DOCX段/TXT全文)", len(raw_docs))

        # ---- 3. 语义切分 ----
        logger.info("[2/6] 执行语义切分 + 兜底截断（逐句调 Embedding API 检测语义边界，可能较慢）...")
        chunks = self.chunker.split_documents(raw_docs)
        chunks = AgriSemanticChunker.assign_doc_ids(chunks)
        logger.info("  最终得到 %d 个有效文本块", len(chunks))

        # 统计信息
        avg_len = sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
        truncated = sum(1 for c in chunks if c.metadata.get("is_truncated", False))
        logger.info("  平均块长度: %.0f 字符 | 被截断块数: %d", avg_len, truncated)

        # ---- 4. 初始化 Collection ----
        logger.info("[3/6] 初始化 Milvus Collection...")
        self.milvus_client.init_collection(drop_if_exists=rebuild)

        # ---- 5. 训练 BM25 并生成稠密向量 ----
        logger.info("[4/6] 训练 BM25 模型...")
        corpus_texts = [c.page_content for c in chunks]
        self.milvus_client.fit_bm25(corpus_texts)

        logger.info("[5/6] 生成稠密向量（这可能需要几分钟，取决于资料规模）...")
        dense_vectors = self._generate_dense_vectors(corpus_texts)
        logger.info("  稠密向量生成完成: %d 条", len(dense_vectors))

        # ---- 6. 入库（去重） ----
        logger.info("[6/6] 数据入库（带去重检查）...")
        inserted = self.milvus_client.insert_chunks(chunks, dense_vectors)
        logger.info("  本次新增: %d 条 | 跳过重复: %d 条",
                     inserted, len(chunks) - inserted)

        # 最终统计
        stats = self.milvus_client.collection_stats()
        logger.info("=" * 60)
        logger.info("  知识库构建完成！")
        logger.info("  Collection: %s", stats.get("name"))
        logger.info("  总记录数: %d", stats.get("num_entities"))
        logger.info("=" * 60)

        return self.milvus_client

    def _generate_dense_vectors(
        self, texts: List[str], batch_size: int = 20
    ) -> List[List[float]]:
        """
        批量生成稠密向量。

        使用 MAASEmbeddings.embed_documents() 逐批调用 MAAS API。
        """
        all_vectors: List[List[float]] = []
        total = len(texts)

        pbar = tqdm(total=total, desc="生成稠密向量", unit="条")
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            try:
                vectors = self.chunker.embeddings.embed_documents(batch)
                all_vectors.extend(vectors)
                pbar.update(len(batch))
            except Exception as e:
                logger.error("稠密向量生成失败 (batch %d-%d): %s", i, i + batch_size, e)
                # 失败时用零向量占位
                for _ in batch:
                    all_vectors.append([0.0] * config.embedding.dim)
                pbar.update(len(batch))

        pbar.close()
        return all_vectors


# ============================================================================
# 阶段二：查询管线
# ============================================================================

class QueryPipeline:
    """
    查询管线

    串联：查询扩展 → 混合检索 → RRF 融合 → Cross-Encoder 精排 → LLM 生成
    """

    def __init__(self, milvus_client: MilvusHybridClient):
        self.milvus_client = milvus_client
        self.query_expander = QueryExpander()
        self.retriever = HybridRetriever(milvus_client)
        self.reranker: CrossEncoderReranker = None  # 延迟加载
        self.generator = AnswerGenerator()

    def _get_reranker(self) -> Optional[CrossEncoderReranker]:
        """延迟初始化 Cross-Encoder（总开关关闭时返回 None）"""
        if not config.retrieval.reranker_enabled:
            return None
        if self.reranker is None:
            logger.info("首次加载 Cross-Encoder 模型...")
            self.reranker = CrossEncoderReranker()
        return self.reranker

    def answer(self, question: str) -> GenerationResult:
        """
        处理单个用户问题，返回带引用的生成结果。

        Args:
            question: 用户原始问题（口语化自然语言）

        Returns:
            GenerationResult 包含 answer + citations + references
        """
        logger.info("-" * 50)
        logger.info("用户问题: %s", question)

        # ---- 1. 查询扩展（可通过 config 关闭） ----
        expanded_queries = self._expand_queries(question)
        if not expanded_queries:
            return GenerationResult(
                answer="查询扩展失败，请重试。",
                citations=[],
                references=[],
            )

        # ---- 2. 混合检索 + RRF ----
        logger.info("[混合检索] 执行双路检索 + RRF 融合...")
        rrf_candidates = self.retriever.retrieve(expanded_queries)
        if not rrf_candidates:
            return GenerationResult(
                answer="抱歉，在当前农业知识库中未检索到与您问题相关的信息。"
                       "建议您：\n1. 尝试使用更具体的关键词描述问题\n"
                       "2. 检查知识库是否已包含相关领域的资料",
                citations=[],
                references=[],
            )

        # ---- 3. Cross-Encoder 精排 ----
        reranker = self._get_reranker()
        if reranker is not None:
            logger.info("[精排] Cross-Encoder 重排序...")
            top_docs = reranker.rerank(question, rrf_candidates)
        else:
            logger.info("[精排] 已关闭，使用 RRF 融合结果 Top-2")
            for cand in rrf_candidates:
                cand["rerank_score"] = cand.get("rrf_score", 0.0)
            top_docs = rrf_candidates[:2]

        # ---- 4. LLM 生成 ----
        logger.info("[生成] 调用 LLM 生成回答...")
        result = self.generator.generate(question, top_docs)
        return result

    def _expand_queries(self, question: str) -> List[str]:
        """根据配置决定是否做 LLM 查询扩展"""
        if config.retrieval.query_expansion_enabled:
            logger.info("[查询扩展] 生成子问题...")
            return self.query_expander.expand(question)
        else:
            logger.info("[查询扩展] 已关闭，直接用原始问题检索")
            return [question]

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        仅检索，不生成回答。返回精排后的知识文档列表。

        供外部 Agent 调用：Agent 拿检索结果用自己的 LLM 生成回答。

        Returns:
            [{content, source_file, page, score, ...}, ...]
        """
        logger.info("-" * 50)
        logger.info("检索请求: %s", question)

        # ---- 1. 查询扩展（可通过 config 关闭） ----
        expanded_queries = self._expand_queries(question)
        if not expanded_queries:
            return []

        # ---- 2. 混合检索 + RRF ----
        rrf_candidates = self.retriever.retrieve(expanded_queries)
        if not rrf_candidates:
            return []

        # ---- 3. Cross-Encoder 精排 ----
        reranker = self._get_reranker()
        if reranker is not None:
            top_docs = reranker.rerank(question, rrf_candidates)
        else:
            for cand in rrf_candidates:
                cand["rerank_score"] = cand.get("rrf_score", 0.0)
            top_docs = rrf_candidates[:2]

        return top_docs


# ============================================================================
# 交互式控制台
# ============================================================================

def interactive_loop(pipeline: QueryPipeline) -> None:
    """交互式问答循环"""
    print("\n" + "=" * 60)
    print("  智慧农业AI助手 — Advanced RAG 系统")
    print("  输入您的问题，系统将基于知识库给出专业回答")
    print("  输入 'quit' / 'exit' / 'q' 退出")
    print("  输入 'stats' 查看知识库统计")
    print("=" * 60 + "\n")

    try:
        while True:
            question = input("🧑 农户 > ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            if question.lower() == "stats":
                stats = pipeline.milvus_client.collection_stats()
                print(f"\n📊 知识库统计:")
                print(f"   Collection: {stats.get('name', 'N/A')}")
                print(f"   记录数: {stats.get('num_entities', 'N/A')}")
                print()
                continue

            # 执行查询
            result = pipeline.answer(question)

            # 流式输出回答
            print("\n🤖 AI助手 > ", end="")
            if result.answer:
                # 模拟流式输出效果（逐字符打印）
                for char in result.answer:
                    print(char, end="", flush=True)
            print("\n")

            # # 输出引用来源
            # if result.citations:
            #     print("📎 引用来源:")
            #     for i, citation in enumerate(result.citations, 1):
            #         print(f"   [{i}] {citation.to_markdown()}")
            #         if citation.content_snippet:
            #             print(f"       相关片段: {citation.content_snippet[:120]}...")
            #     print()

    except KeyboardInterrupt:
        print("\n\n用户中断，再见！")
    except Exception as e:
        logger.error("交互循环异常: %s", e, exc_info=True)
        print(f"\n发生错误: {e}")


# ============================================================================
# 命令行入口
# ============================================================================

def _collection_has_data() -> int:
    """检查 Milvus 中是否已有知识库数据，返回记录数（0 表示无数据）"""
    try:
        from pymilvus import connections, utility
        connections.connect(alias="_probe", uri=config.milvus.uri, timeout=10)
        if not utility.has_collection(config.milvus.collection_name):
            connections.disconnect("_probe")
            return 0
        from pymilvus import Collection
        coll = Collection(config.milvus.collection_name)
        coll.load()
        count = coll.num_entities
        connections.disconnect("_probe")
        return count
    except Exception as e:
        logger.debug("知识库探测失败: %s", e)
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="智慧农业AI病虫害识别与种植决策系统 — Advanced RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                   # 智能模式：知识库不存在则构建，已存在则直接对话
  python main.py --chat            # 跳过构建检查，直接进入对话（知识库必须已存在）
  python main.py --rebuild         # 强制重建知识库后进入对话
  python main.py --ingest-only     # 仅构建/更新知识库，不进入对话
  python main.py --ingest-only --rebuild  # 强制重建知识库后退出
        """,
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="强制删除已有 Collection 并重建知识库",
    )
    parser.add_argument(
        "--ingest-only", action="store_true",
        help="仅执行知识库构建/更新，不进入交互对话",
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="跳过构建/探测，直接进入交互对话（知识库必须已存在）",
    )
    args = parser.parse_args()

    # ================================================================
    # 决策逻辑：
    #   --chat        → 跳过一切，直接对话
    #   --rebuild     → 强制重建 → 看 --ingest-only 决定是否进对话
    #   --ingest-only → 构建/更新 → 退出
    #   默认          → 探测 Collection → 有数据则直接对话，无数据则构建
    # ================================================================

    milvus_client: Optional[MilvusHybridClient] = None

    # ---- 纯对话模式 ----
    if args.chat and not args.rebuild and not args.ingest_only:
        milvus_client = MilvusHybridClient()
        milvus_client.connect()
        milvus_client.init_collection(drop_if_exists=False)
        stats = milvus_client.collection_stats()
        if stats.get("num_entities", 0) == 0:
            logger.error("知识库为空！请先运行 python main.py 构建知识库。")
            milvus_client.disconnect()
            return
        logger.info("进入对话模式（知识库已有 %d 条记录）", stats["num_entities"])
        milvus_client.load_corpus_and_fit_bm25()

    # ---- 强制重建 ----
    elif args.rebuild:
        builder = KnowledgeBaseBuilder()
        milvus_client = builder.build(rebuild=True)
        if args.ingest_only:
            logger.info("知识库重建完成，退出。")
            milvus_client.disconnect()
            return

    # ---- 仅入库（Collection 若已存在则增量追加） ----
    elif args.ingest_only:
        existing = _collection_has_data()
        builder = KnowledgeBaseBuilder()
        milvus_client = builder.build(rebuild=(existing == 0))
        logger.info("知识库构建完成，退出。")
        milvus_client.disconnect()
        return

    # ---- 智能默认模式 ----
    else:
        existing = _collection_has_data()
        if existing > 0:
            logger.info("检测到知识库已有 %d 条记录，跳过构建，直接进入对话。", existing)
            logger.info("如需重建请使用: python main.py --rebuild")
            milvus_client = MilvusHybridClient()
            milvus_client.connect()
            milvus_client.init_collection(drop_if_exists=False)
            milvus_client.load_corpus_and_fit_bm25()
        else:
            logger.info("知识库为空，开始自动构建...")
            builder = KnowledgeBaseBuilder()
            milvus_client = builder.build(rebuild=True)

    # ---------- 交互式问答 ----------
    try:
        query_pipeline = QueryPipeline(milvus_client)
        interactive_loop(query_pipeline)
    finally:
        if milvus_client:
            milvus_client.disconnect()
        logger.info("系统已关闭")


if __name__ == "__main__":
    main()
