"""
=============================================================================
Cross-Encoder 精排模块
=============================================================================
加载 BAAI/bge-reranker-v2-m3 等 Cross-Encoder 模型，将 RRF 融合后
的候选文档与用户原始问题进行交叉编码评分，按相关性得分二次精排，
最终选取得分最高的 Top-N 个文档片段作为 LLM 生成的参考上下文。

支持：
  - HuggingFace 镜像加速（HF_ENDPOINT 环境变量 / config 配置）
  - 模型加载失败时自动降级：跳过精排，直接使用 RRF 排序结果
"""

import logging
import os
from typing import List, Dict, Any, Optional

from config import config

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder 重排序器

    使用 sentence-transformers 加载 Cross-Encoder 模型（默认
    BAAI/bge-reranker-v2-m3），对候选文档与用户问题进行逐对
    交叉编码，计算相关性得分并降序重排。

    若模型加载失败（网络不通等），自动降级为跳过精排，
    直接用 RRF 融合结果作为最终候选。此时日志会输出降级警告。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        retrieval_cfg: Optional[object] = None,
    ):
        self.retrieval_cfg = retrieval_cfg or config.retrieval
        self.device = device or self.retrieval_cfg.reranker_device
        self.top_k = self.retrieval_cfg.reranker_top_k
        self._model = None
        self._model_available = False

        # 解析模型路径：显式指定 > 本地目录 > HuggingFace 在线
        if model_name:
            self.model_name = model_name
        elif self.retrieval_cfg.reranker_model_name:
            self.model_name = self.retrieval_cfg.reranker_model_name
        elif (self.retrieval_cfg.reranker_local_path
              and os.path.isdir(self.retrieval_cfg.reranker_local_path)):
            self.model_name = self.retrieval_cfg.reranker_local_path
            logger.info("使用本地 Cross-Encoder 模型: %s", self.model_name)
        else:
            self.model_name = self.retrieval_cfg.reranker_hf_model
            # 设置 HuggingFace 镜像（在线下载时需要）
            hf_endpoint = self.retrieval_cfg.hf_endpoint
            if hf_endpoint and not os.getenv("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = hf_endpoint
                logger.info("已设置 HF_ENDPOINT=%s", hf_endpoint)

        self._load_model()

    def _load_model(self) -> None:
        """加载 Cross-Encoder 模型（失败时降级而非崩溃）"""
        try:
            from sentence_transformers import CrossEncoder

            # CUDA 不可用时自动退 CPU
            device = self.device
            if device.startswith("cuda"):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA 不可用（PyTorch 为 CPU 版本），退至 CPU")
                        device = "cpu"
                except ImportError:
                    device = "cpu"

            logger.info("正在加载 Cross-Encoder 模型: %s (device=%s)",
                         self.model_name, device)
            self._model = CrossEncoder(
                self.model_name,
                device=device,
                trust_remote_code=True,
            )
            self._model_available = True
            logger.info("Cross-Encoder 模型加载完成")
        except ImportError:
            logger.warning(
                "sentence-transformers 未安装，精排将跳过。"
                "安装: pip install sentence-transformers"
            )
            self._model_available = False
        except Exception as e:
            logger.warning(
                "Cross-Encoder 模型加载失败（网络不通或模型不存在），"
                "将跳过精排，直接使用 RRF 融合结果。错误: %s", e
            )
            self._model_available = False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        对候选文档进行 Cross-Encoder 重排序。

        若模型不可用，直接返回 RRF 排序的 Top-N 候选（降级模式）。
        """
        if not candidates:
            logger.warning("候选文档列表为空，跳过重排序")
            return []

        # ---- 降级模式：模型不可用时直接用 RRF 结果 ----
        if not self._model_available:
            logger.warning(
                "[降级] Cross-Encoder 不可用，使用 RRF 融合结果 Top-%d",
                min(self.top_k, len(candidates)),
            )
            for cand in candidates:
                cand["rerank_score"] = cand.get("rrf_score", 0.0)
            return candidates[:self.top_k]

        logger.info("Cross-Encoder 重排序: %d 个候选 → Top-%d",
                     len(candidates), self.top_k)

        # 构造 (query, document) 对
        pairs = [(query, cand.get("content", "")) for cand in candidates]

        # 计算相关性得分
        try:
            scores = self._model.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
                convert_to_tensor=True,
            )
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            elif hasattr(scores, "cpu"):
                scores = scores.cpu().tolist()
        except Exception as e:
            logger.error("Cross-Encoder 评分失败: %s，降级使用 RRF 结果", e)
            for cand in candidates:
                cand["rerank_score"] = cand.get("rrf_score", 0.0)
            return candidates[:self.top_k]

        # 绑定得分
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = round(float(score), 6)

        # 按得分降序排列
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", 0.0),
            reverse=True,
        )

        # 取 Top-N
        top_candidates = sorted_candidates[:self.top_k]

        for rank, cand in enumerate(top_candidates):
            logger.info(
                "  [%d] score=%.4f | file=%s | page=%s | preview=%s",
                rank + 1,
                cand.get("rerank_score", 0.0),
                cand.get("source_file", "?"),
                cand.get("page", "?"),
                cand.get("content", "")[:60].replace("\n", " "),
            )

        return top_candidates


# ============================================================================
# 便捷函数
# ============================================================================

def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """便捷函数：Cross-Encoder 重排序"""
    reranker = CrossEncoderReranker()
    return reranker.rerank(query, candidates)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.log.format)
    reranker = CrossEncoderReranker()
    test_query = "玉米叶子发黄是什么病虫害？"
    test_candidates = [
        {"content": "玉米大斑病症状：叶片出现大型梭形或不规则形病斑...",
         "source_file": "玉米病害列表.txt", "page": 0, "rrf_score": 0.05},
        {"content": "缺氮症状：植株生长缓慢，叶片由下而上逐渐变黄...",
         "source_file": "施肥1.txt", "page": 0, "rrf_score": 0.03},
    ]
    result = reranker.rerank(test_query, test_candidates)
    print("\n=== 重排序结果 ===")
    for i, doc in enumerate(result):
        print(f"[{i+1}] score={doc.get('rerank_score', 0):.4f} | "
              f"{doc['source_file']} | {doc['content'][:60]}...")
