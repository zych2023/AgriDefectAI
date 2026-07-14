"""
=============================================================================
阿里云 MAAS Embedding 封装 — 对标 text_embedding.py 的调用方式
=============================================================================
绕过 langchain_openai.OpenAIEmbeddings 的内部批处理逻辑，
直接使用原生 openai.OpenAI 客户端逐条调用 MAAS Embedding API，
确保请求格式与已验证可用的 text_embedding.py 完全一致。

同时实现 LangChain 的 Embeddings 接口，可直接注入 SemanticChunker
和 HybridRetriever 中使用。
"""

import logging
from typing import List, Optional

from openai import OpenAI
from langchain_core.embeddings import Embeddings

from config import config

logger = logging.getLogger(__name__)


class MAASEmbeddings(Embeddings):
    """
    阿里云 MAAS 文本嵌入模型封装

    调用方式严格对标 text_embedding.py：
      client = OpenAI(base_url=..., api_key=...)
      response = client.embeddings.create(model=..., input=text, dimensions=...)
      return response.data[0].embedding

    实现 LangChain Embeddings 接口（embed_documents / embed_query），
    可无缝替换 OpenAIEmbeddings。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dim: Optional[int] = None,
        batch_size: int = 16,  # 每批并发调用的文本数
    ):
        emb_cfg = config.embedding
        self._base_url = base_url or emb_cfg.base_url
        self._api_key = api_key or emb_cfg.api_key
        self._model_name = model_name or emb_cfg.model_name
        self._dim = dim or emb_cfg.dim
        self._batch_size = batch_size

        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )
        logger.info(
            "MAASEmbeddings 初始化: model=%s dim=%d base_url=%s",
            self._model_name,
            self._dim,
            self._base_url,
        )

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询文本（对标 text_embedding.py 调用方式）"""
        if not text or not text.strip():
            logger.warning("embed_query 收到空文本，返回零向量")
            return [0.0] * self._dim

        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=text,
                dimensions=self._dim,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("embed_query 失败: %s", e)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档文本。

        阿里云 MAAS API 对批量 input 的格式校验可能与 langchain 内部
        封装存在兼容性问题，因此这里采用逐条调用（与 text_embedding.py
        完全一致的调用方式），确保每条请求都能被正确解析。

        对于大批量场景（如 SemanticChunker），使用内部小批量串行发送，
        在可靠性与效率之间取得平衡。
        """
        if not texts:
            return []

        # 过滤空文本（用空格占位避免 API 报错）
        sanitized = [t if t and t.strip() else " " for t in texts]

        all_embeddings: List[List[float]] = []
        total = len(sanitized)

        for i in range(0, total, self._batch_size):
            batch = sanitized[i : i + self._batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            progress = min(i + self._batch_size, total)
            if total > self._batch_size:
                logger.debug("embed_documents 进度: %d/%d", progress, total)

        return all_embeddings

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        逐条调用 MAAS API 嵌入一个批次内的所有文本。

        每条调用完全对标 text_embedding.py 的格式：
          client.embeddings.create(model=..., input=<单条文本>, dimensions=...)
        """
        results: List[List[float]] = []
        for text in texts:
            try:
                response = self._client.embeddings.create(
                    model=self._model_name,
                    input=text,
                    dimensions=self._dim,
                )
                results.append(response.data[0].embedding)
            except Exception as e:
                logger.error("MAAS embedding 单条调用失败: %s", e)
                # 失败时用零向量占位，避免整个流程中断
                results.append([0.0] * self._dim)
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.log.format)

    emb = MAASEmbeddings()

    # 测试单条
    vec = emb.embed_query("玉米叶子发黄是怎么回事")
    print(f"单条嵌入: dim={len(vec)}, 前5维={vec[:5]}")

    # 测试批量
    texts = ["水稻稻瘟病的防治方法", "小麦锈病的识别特征", "果树施肥的最佳时期"]
    vecs = emb.embed_documents(texts)
    print(f"批量嵌入: count={len(vecs)}, dim={len(vecs[0])}")
