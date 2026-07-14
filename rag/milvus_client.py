"""
=============================================================================
Milvus 向量数据库客户端 — 混合检索 Collection 设计与管理
=============================================================================
设计同时包含 FloatVector（稠密）和 SparseFloatVector（稀疏）的 Collection，
支持：
  - 稠密向量索引：IVF_FLAT / COSINE
  - 稀疏向量索引：SPARSE_INVERTED_INDEX / IP
  - 基于 MD5 doc_id 的去重插入
  - 若 Collection 已存在则自动删除重建（方便开发调试）
"""

import logging
import time
from typing import List, Optional, Dict, Any, Set

import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
    MilvusClient,
)
from pymilvus.model.sparse import BM25EmbeddingFunction
from tqdm import tqdm

from config import config

logger = logging.getLogger(__name__)

# ============================================================================
# Collection Schema 字段定义常量
# ============================================================================
FIELD_ID = "id"
FIELD_DOC_ID = "doc_id"
FIELD_CONTENT = "content"
FIELD_DENSE = "dense_vector"
FIELD_SPARSE = "sparse_vector"
FIELD_SOURCE = "source_file"
FIELD_PAGE = "page"
FIELD_CHUNK_IDX = "chunk_index"
FIELD_TRUNCATED = "is_truncated"
FIELD_SOURCE_PATH = "source_path"
FIELD_FILE_TYPE = "file_type"


class MilvusHybridClient:
    """
    Milvus 混合检索客户端

    职责：
    - 管理 Collection 生命周期（创建 / 删除重建）
    - 构建稠密 + 稀疏双向量索引
    - 基于 doc_id 去重插入
    - 提供稠密 / 稀疏两路检索接口
    """

    def __init__(
        self,
        milvus_cfg: Optional[object] = None,
    ):
        self.milvus_cfg = milvus_cfg or config.milvus
        self.collection: Optional[Collection] = None
        self.bm25_ef: Optional[BM25EmbeddingFunction] = None

    # ========================================================================
    # 连接管理
    # ========================================================================

    def connect(self) -> None:
        """建立与 Milvus 服务的连接"""
        try:
            connections.connect(
                alias="default",
                uri=self.milvus_cfg.uri,
                db_name=self.milvus_cfg.db_name,
                timeout=self.milvus_cfg.timeout,
            )
            logger.info("Milvus 连接成功: %s", self.milvus_cfg.uri)
        except MilvusException as e:
            logger.error("Milvus 连接失败: %s", e)
            raise

    def disconnect(self) -> None:
        """断开 Milvus 连接"""
        try:
            connections.disconnect("default")
            logger.info("Milvus 连接已断开")
        except Exception:
            pass

    # ========================================================================
    # Collection 初始化（若已存在则删除重建）
    # ========================================================================

    def init_collection(self, drop_if_exists: bool = True) -> Collection:
        """
        初始化 Collection —— 若已存在则删除后重建。

        Schema 字段：
          id            INT64   主键（自增）
          doc_id        VARCHAR    MD5 业务唯一标识（用于去重）
          content       VARCHAR    原始文本内容
          dense_vector  FLOAT_VECTOR  稠密向量
          sparse_vector SPARSE_FLOAT_VECTOR  稀疏向量
          source_file   VARCHAR    来源文件名
          source_path   VARCHAR    来源文件完整路径
          page          INT64      页码
          chunk_index   INT64      文本块序号
          is_truncated  BOOL       是否被二次截断
          file_type     VARCHAR    文件类型
        """
        collection_name = self.milvus_cfg.collection_name

        # 删除已有 Collection
        if drop_if_exists and utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info("已删除旧 Collection: %s", collection_name)

        if utility.has_collection(collection_name):
            logger.info("Collection 已存在，直接加载: %s", collection_name)
            self.collection = Collection(collection_name)
            self.collection.load()
            return self.collection

        # 定义 Schema
        fields = [
            FieldSchema(name=FIELD_ID, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=FIELD_DOC_ID, dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name=FIELD_CONTENT, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name=FIELD_DENSE,
                dtype=DataType.FLOAT_VECTOR,
                dim=config.embedding.dim,
            ),
            FieldSchema(name=FIELD_SPARSE, dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name=FIELD_SOURCE, dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name=FIELD_SOURCE_PATH, dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name=FIELD_PAGE, dtype=DataType.INT64),
            FieldSchema(name=FIELD_CHUNK_IDX, dtype=DataType.INT64),
            FieldSchema(name=FIELD_TRUNCATED, dtype=DataType.BOOL),
            FieldSchema(name=FIELD_FILE_TYPE, dtype=DataType.VARCHAR, max_length=16),
        ]

        schema = CollectionSchema(fields, description="农业知识库 — 混合检索")
        self.collection = Collection(name=collection_name, schema=schema)
        logger.info("Collection 创建成功: %s (dim=%d)", collection_name, config.embedding.dim)

        # 创建索引
        self._create_indexes()

        # 加载到内存
        self.collection.load()
        logger.info("Collection 已加载到内存")
        return self.collection

    def _create_indexes(self) -> None:
        """创建稠密向量与稀疏向量索引"""
        if self.collection is None:
            raise RuntimeError("Collection 未初始化")

        # 稠密向量索引
        dense_index_params = {
            "index_type": self.milvus_cfg.dense_index_type,
            "metric_type": self.milvus_cfg.dense_metric_type,
            "params": {"nlist": self.milvus_cfg.dense_nlist},
        }
        self.collection.create_index(
            field_name=FIELD_DENSE,
            index_params=dense_index_params,
        )
        logger.info(
            "稠密索引创建完成: %s / %s (nlist=%d)",
            self.milvus_cfg.dense_index_type,
            self.milvus_cfg.dense_metric_type,
            self.milvus_cfg.dense_nlist,
        )

        # 稀疏向量索引
        sparse_index_params = {
            "index_type": self.milvus_cfg.sparse_index_type,
            "metric_type": self.milvus_cfg.sparse_metric_type,
        }
        self.collection.create_index(
            field_name=FIELD_SPARSE,
            index_params=sparse_index_params,
        )
        logger.info(
            "稀疏索引创建完成: %s / %s",
            self.milvus_cfg.sparse_index_type,
            self.milvus_cfg.sparse_metric_type,
        )

    # ========================================================================
    # BM25 模型训练
    # ========================================================================

    def fit_bm25(self, corpus_texts: List[str]) -> None:
        """
        用全部语料训练 BM25 模型，用于后续编码文档和查询的稀疏向量。

        BM25EmbeddingFunction 构造函数接收分词器（analyzer），
        语料通过 .fit() 方法传入训练。

        Args:
            corpus_texts: 所有文本块的文本内容列表
        """
        if not corpus_texts:
            logger.warning("BM25 训练语料为空，跳过")
            return
        logger.info("开始训练 BM25 模型，语料规模: %d", len(corpus_texts))
        # jieba 中文分词器（零外部数据依赖，避免 NLTK stopwords 下载问题）
        import jieba
        analyzer = lambda text: list(jieba.cut(text))
        self.bm25_ef = BM25EmbeddingFunction(analyzer)
        # BM25 fit 对中文语料逐条分词，可能较慢，加进度条
        self.bm25_ef.fit(
            tqdm(corpus_texts, desc="BM25 训练", unit="条")
        )
        logger.info("BM25 模型训练完成")

    # ========================================================================
    # 数据插入（带去重）
    # ========================================================================

    def get_existing_doc_ids(self) -> Set[str]:
        """查询 Milvus 中已存在的所有 doc_id"""
        if self.collection is None:
            raise RuntimeError("Collection 未初始化")

        existing_ids: Set[str] = set()
        try:
            # 分页查询所有 doc_id
            offset = 0
            batch_size = 10000
            while True:
                results = self.collection.query(
                    expr="id >= 0",
                    output_fields=[FIELD_DOC_ID],
                    offset=offset,
                    limit=batch_size,
                )
                if not results:
                    break
                for row in results:
                    existing_ids.add(row[FIELD_DOC_ID])
                if len(results) < batch_size:
                    break
                offset += batch_size
        except Exception as e:
            logger.warning("查询已有 doc_id 时出错（可能 Collection 为空）: %s", e)

        logger.info("已有 doc_id 数量: %d", len(existing_ids))
        return existing_ids

    def insert_chunks(
        self,
        chunks: List[Any],
        dense_vectors: List[List[float]],
    ) -> int:
        """
        将文本块 + 稠密向量 + 稀疏向量 插入 Milvus，跳过已存在的 doc_id。

        Args:
            chunks:      Document 对象列表（包含 page_content 和 metadata）
            dense_vectors: 对应的稠密向量列表

        Returns:
            实际插入的新记录数量
        """
        if self.collection is None:
            raise RuntimeError("Collection 未初始化")
        if self.bm25_ef is None:
            raise RuntimeError("BM25 模型未训练，请先调用 fit_bm25()")

        existing_ids = self.get_existing_doc_ids()

        # 筛出全新块
        new_chunks_with_idx: List[tuple] = []  # (index, chunk)
        for idx, chunk in enumerate(chunks):
            doc_id = chunk.metadata.get("doc_id", "")
            if doc_id in existing_ids:
                logger.debug("跳过重复块: doc_id=%s", doc_id[:16])
            else:
                new_chunks_with_idx.append((idx, chunk))

        if not new_chunks_with_idx:
            logger.info("所有文本块均已存在，无需插入")
            return 0

        logger.info("准备插入 %d 条新记录（跳过 %d 条重复）",
                     len(new_chunks_with_idx),
                     len(chunks) - len(new_chunks_with_idx))

        # 新块的稀疏向量 —— BM25EmbeddingFunction 返回 scipy 稀疏矩阵，
        # 需统一转为 {int: float} dict 格式才能被 pymilvus 正确序列化
        new_texts = [c.page_content for _, c in new_chunks_with_idx]
        raw_sparse = self.bm25_ef.encode_documents(new_texts)
        sparse_vectors = []
        for sv in raw_sparse:
            if isinstance(sv, dict):
                sparse_vectors.append({int(k): float(v) for k, v in sv.items()})
            elif hasattr(sv, "tocoo"):
                # scipy 稀疏矩阵 → {(row, col): value} → {col: value}（每行只有一个 row=0）
                coo = sv.tocoo()
                d = {int(c): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}
                sparse_vectors.append(d)
            else:
                logger.warning("未知稀疏向量类型: %s，跳过", type(sv).__name__)
                sparse_vectors.append({})

        # 诊断
        if sparse_vectors:
            sv0 = sparse_vectors[0]
            logger.info("稀疏向量格式诊断: type=dict len=%d sample=%s",
                         len(sv0), dict(list(sv0.items())[:5]))

        # 组装 list-of-dicts 并逐批插入
        rows: List[Dict[str, Any]] = []
        for i, (orig_idx, chunk) in enumerate(new_chunks_with_idx):
            meta = chunk.metadata
            rows.append({
                FIELD_DOC_ID: meta.get("doc_id", ""),
                FIELD_CONTENT: chunk.page_content,
                FIELD_DENSE: dense_vectors[orig_idx],
                FIELD_SPARSE: sparse_vectors[i],
                FIELD_SOURCE: meta.get("source_file", ""),
                FIELD_SOURCE_PATH: meta.get("source_path", ""),
                FIELD_PAGE: meta.get("page", 0),
                FIELD_CHUNK_IDX: meta.get("chunk_index", 0),
                FIELD_TRUNCATED: meta.get("is_truncated", False),
                FIELD_FILE_TYPE: meta.get("file_type", ""),
            })

        try:
            mc = MilvusClient(uri=self.milvus_cfg.uri, db_name=self.milvus_cfg.db_name)
            mr = mc.insert(collection_name=self.milvus_cfg.collection_name, data=rows)
            self.collection.flush()
            logger.info(
                "插入成功: %d 条记录 | 等待索引就绪中...",
                len(rows),
            )
            time.sleep(2)
            return len(rows)
        except MilvusException as e:
            logger.error("Milvus 插入失败: %s", e)
            raise

    # ========================================================================
    # 检索接口
    # ========================================================================

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        稠密向量相似度检索（COSINE）。

        Returns:
            [{content, source_file, page, score, ...}, ...]
        """
        if self.collection is None:
            raise RuntimeError("Collection 未初始化")

        search_params = {
            "metric_type": self.milvus_cfg.dense_metric_type,
            "params": {"nprobe": 16},
        }
        results = self.collection.search(
            data=[query_vector],
            anns_field=FIELD_DENSE,
            param=search_params,
            limit=top_k,
            output_fields=[
                FIELD_CONTENT, FIELD_SOURCE, FIELD_SOURCE_PATH,
                FIELD_PAGE, FIELD_DOC_ID, FIELD_CHUNK_IDX, FIELD_FILE_TYPE,
            ],
        )
        return self._parse_search_results(results, "dense")

    def search_sparse(
        self,
        query_text: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        稀疏向量 BM25 关键词检索。

        Args:
            query_text: 原始查询文本（由 BM25 模型编码为稀疏向量）

        Returns:
            [{content, source_file, page, score, ...}, ...]
        """
        if self.collection is None:
            raise RuntimeError("Collection 未初始化")
        if self.bm25_ef is None:
            raise RuntimeError("BM25 模型未训练")

        sparse_vector = self.bm25_ef.encode_queries([query_text])
        search_params = {"metric_type": self.milvus_cfg.sparse_metric_type}
        results = self.collection.search(
            data=sparse_vector,
            anns_field=FIELD_SPARSE,
            param=search_params,
            limit=top_k,
            output_fields=[
                FIELD_CONTENT, FIELD_SOURCE, FIELD_SOURCE_PATH,
                FIELD_PAGE, FIELD_DOC_ID, FIELD_CHUNK_IDX, FIELD_FILE_TYPE,
            ],
        )
        return self._parse_search_results(results, "sparse")

    def _parse_search_results(
        self, results: List[Any], source: str
    ) -> List[Dict[str, Any]]:
        """将 pymilvus 搜索结果解析为统一字典格式"""
        parsed = []
        for hits in results:
            for hit in hits:
                parsed.append({
                    "content": hit.entity.get(FIELD_CONTENT, ""),
                    "source_file": hit.entity.get(FIELD_SOURCE, ""),
                    "source_path": hit.entity.get(FIELD_SOURCE_PATH, ""),
                    "page": hit.entity.get(FIELD_PAGE, -1),
                    "doc_id": hit.entity.get(FIELD_DOC_ID, ""),
                    "chunk_index": hit.entity.get(FIELD_CHUNK_IDX, -1),
                    "file_type": hit.entity.get(FIELD_FILE_TYPE, ""),
                    "score": hit.distance,
                    "retrieval_source": source,  # "dense" | "sparse"
                })
        return parsed

    # ========================================================================
    # BM25 模型恢复（对话模式启动时从已有 Collection 重建）
    # ========================================================================

    def load_corpus_and_fit_bm25(self) -> None:
        """
        从 Milvus Collection 中读取全部 content 文本，重新拟合 BM25 模型。

        用于直接进入对话模式时恢复稀疏检索能力。
        """
        if self.collection is None:
            raise RuntimeError("Collection 未初始化")

        logger.info("从 Milvus 加载全部语料以重建 BM25 模型...")
        corpus_texts: List[str] = []
        offset = 0
        batch_size = 5000
        while True:
            results = self.collection.query(
                expr="id >= 0",
                output_fields=[FIELD_CONTENT],
                offset=offset,
                limit=batch_size,
            )
            if not results:
                break
            corpus_texts.extend(r[FIELD_CONTENT] for r in results)
            offset += batch_size

        logger.info("加载语料 %d 条，开始拟合 BM25...", len(corpus_texts))
        self.fit_bm25(corpus_texts)
        logger.info("BM25 模型恢复完成")

    # ========================================================================
    # 统计信息
    # ========================================================================

    def collection_stats(self) -> Dict[str, Any]:
        """获取 Collection 基本统计信息"""
        if self.collection is None:
            return {"status": "not_initialized"}
        try:
            self.collection.flush()
            return {
                "name": self.collection.name,
                "num_entities": self.collection.num_entities,
                "schema_fields": [f.name for f in self.collection.schema.fields],
            }
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# 便捷函数
# ============================================================================

def create_milvus_client() -> MilvusHybridClient:
    """工厂函数：创建并连接 MilvusHybridClient"""
    client = MilvusHybridClient()
    client.connect()
    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.log.format)
    client = create_milvus_client()
    try:
        coll = client.init_collection(drop_if_exists=True)
        stats = client.collection_stats()
        print(stats)
    finally:
        client.disconnect()
