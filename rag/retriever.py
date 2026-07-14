"""
=============================================================================
混合检索与 RRF 融合模块
=============================================================================
针对每个扩展子问题同时发起稠密 + 稀疏双路检索，汇集所有召回结果后
执行倒数排名融合（RRF, Reciprocal Rank Fusion）算法进行首次合并排序。
"""

import logging
from typing import List, Dict, Any, Set, Optional

from maas_embedding import MAASEmbeddings

from config import config
from milvus_client import MilvusHybridClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器

    流程：
    1. 对每个扩展子问题：
       a. 用 Embedding 模型编码为稠密向量 → 稠密向量检索
       b. 用 BM25 模型编码为稀疏向量 → 稀疏向量检索
    2. 汇集所有子问题 × 两路检索的结果
    3. 执行 RRF 算法融合排序
    4. 返回 Top-N 候选文档
    """

    def __init__(
        self,
        milvus_client: MilvusHybridClient,
        retrieval_cfg: Optional[object] = None,
    ):
        self.client = milvus_client
        self.retrieval_cfg = retrieval_cfg or config.retrieval

        # 稠密向量编码器（使用 MAAS 兼容的 Embeddings，对标 text_embedding.py）
        emb_cfg = config.embedding
        self.embeddings = MAASEmbeddings(
            base_url=emb_cfg.base_url,
            api_key=emb_cfg.api_key,
            model_name=emb_cfg.model_name,
            dim=emb_cfg.dim,
        )

    # ========================================================================
    # 双路检索
    # ========================================================================

    def retrieve(
        self,
        expanded_queries: List[str],
    ) -> List[Dict[str, Any]]:
        """
        对扩展查询列表中的每个子问题执行双路检索并 RRF 融合。

        Args:
            expanded_queries: 扩展后的子问题列表

        Returns:
            RRF 融合排序后的候选文档列表
        """
        if not expanded_queries:
            logger.warning("扩展查询列表为空")
            return []

        all_candidates: List[Dict[str, Any]] = []
        seen_doc_ids: Set[str] = set()

        for i, query in enumerate(expanded_queries):
            logger.info("检索子问题 [%d/%d]: %s", i + 1, len(expanded_queries), query[:60])
            try:
                candidates = self._dual_retrieve(query)
                for cand in candidates:
                    # 同一子问题的两路可能重复 → 按 doc_id 去重
                    did = cand.get("doc_id", "")
                    if did not in seen_doc_ids:
                        seen_doc_ids.add(did)
                        cand["query_index"] = i  # 记录来自哪个子问题
                        all_candidates.append(cand)
                logger.info("  子问题 [%d] 召回 %d 个唯一候选", i + 1, len(candidates))
            except Exception as e:
                logger.error("子问题 [%d] 检索失败: %s", i + 1, e)
                continue

        if not all_candidates:
            logger.warning("所有检索均未返回结果")
            return []

        logger.info("全部子问题检索完成，共 %d 个唯一候选", len(all_candidates))

        # RRF 融合
        fused = self._rrf_fusion(all_candidates)
        logger.info("RRF 融合完成，返回 Top-%d", len(fused))
        return fused

    def _dual_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        针对单个查询执行稠密 + 稀疏双路检索并简单合并去重。
        """
        dense_k = self.retrieval_cfg.dense_top_k
        sparse_k = self.retrieval_cfg.sparse_top_k

        # 稠密向量检索
        query_vector = self.embeddings.embed_query(query)
        dense_results = self.client.search_dense(query_vector, top_k=dense_k)
        logger.debug("  稠密检索: %d 条", len(dense_results))

        # 稀疏向量检索
        sparse_results = self.client.search_sparse(query, top_k=sparse_k)
        logger.debug("  稀疏检索: %d 条", len(sparse_results))

        # 合并去重（按 doc_id），记录检索来源
        merged: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(dense_results):
            did = item.get("doc_id", f"__dense_{idx}")
            if did not in merged:
                item["dense_rank"] = idx + 1  # 1-indexed
                item["sparse_rank"] = float("inf")
                merged[did] = item

        for idx, item in enumerate(sparse_results):
            did = item.get("doc_id", f"__sparse_{idx}")
            if did in merged:
                merged[did]["sparse_rank"] = idx + 1
            else:
                item["sparse_rank"] = idx + 1
                item["dense_rank"] = float("inf")
                merged[did] = item

        return list(merged.values())

    # ========================================================================
    # RRF 倒数排名融合
    # ========================================================================

    def _rrf_fusion(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        倒数排名融合（Reciprocal Rank Fusion）

        公式: RRF_score(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))
        其中 k=60（平滑参数），R 为各路检索通道集合。

        融合后按 RRF 分数降序排列，取前 rrf_fusion_top_k 个候选。
        """
        k = self.retrieval_cfg.rrf_k

        # 按子问题分组 — 每个子问题各有一路 dense + 一路 sparse
        query_groups: Dict[int, List[Dict[str, Any]]] = {}
        for cand in candidates:
            qi = cand.get("query_index", 0)
            if qi not in query_groups:
                query_groups[qi] = []
            query_groups[qi].append(cand)

        # 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}   # doc_id → accumulated RRF score
        doc_registry: Dict[str, Dict[str, Any]] = {}

        for qi, group in query_groups.items():
            # 在每个子问题 × 每个检索通道（dense/sparse）内部分别排名
            for channel in ["dense", "sparse"]:
                rank_key = f"{channel}_rank"
                ranked = sorted(
                    [c for c in group if c.get(rank_key, float("inf")) != float("inf")],
                    key=lambda x: x.get(rank_key, float("inf")),
                )
                for rank_idx, cand in enumerate(ranked):
                    did = cand.get("doc_id", "")
                    rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank_idx + 1)
                    if did not in doc_registry:
                        doc_registry[did] = cand

        # 按 RRF 分数降序排序
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = min(self.retrieval_cfg.rrf_fusion_top_k, len(sorted_docs))

        fused = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = dict(doc_registry[doc_id])
            doc["rrf_score"] = round(rrf_score, 6)
            fused.append(doc)

        logger.info(
            "RRF 融合: %d 个候选 → Top-%d (k=%d)",
            len(candidates),
            len(fused),
            k,
        )
        return fused


# ============================================================================
# 便捷函数
# ============================================================================

def hybrid_retrieve(
    client: MilvusHybridClient,
    expanded_queries: List[str],
) -> List[Dict[str, Any]]:
    """便捷函数：混合检索 + RRF 融合"""
    retriever = HybridRetriever(client)
    return retriever.retrieve(expanded_queries)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.log.format)
    # 需要先初始化 Milvus 并插入数据
    print("此模块需要配合完整的 pipeline 运行，请使用 main.py")
