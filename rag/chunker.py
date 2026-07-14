"""
=============================================================================
语义切分模块 — SemanticChunker + 兜底硬截断
=============================================================================
优先使用基于 Embedding 的语义边界切分（SemanticChunker），确保完整的知识
点不会被拦腰斩断；随后对每个文本块做长度检查，超出阈值的块交给
RecursiveCharacterTextSplitter 进行二次硬截断并标记截断状态。
"""

import hashlib
import logging
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from maas_embedding import MAASEmbeddings
from config import config

logger = logging.getLogger(__name__)


# ============================================================================
# Embedding 进度包装器 —— 拦截 embed_documents 调用以驱动进度条
# ============================================================================

class ProgressEmbeddings:
    """包装 MAASEmbeddings，在 SemanticChunker 逐句调 API 时更新进度条"""

    def __init__(self, embeddings: MAASEmbeddings, total: int, desc: str = "语义切分"):
        self._embeddings = embeddings
        self.pbar = tqdm(total=total, desc=desc, unit="句")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._embeddings.embed_documents(texts)
        self.pbar.update(len(texts))
        return result

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def close(self):
        self.pbar.close()


# ============================================================================
# 语义切分器
# ============================================================================


class AgriSemanticChunker:
    """
    农业知识语义分块器

    工作流程：
    1. 使用 SemanticChunker（基于 Embedding 的语义突变检测）切分文档
    2. 遍历每个语义块，若字符长度 > max_chunk_size，
       则使用 RecursiveCharacterTextSplitter 二次截断，
       并在元数据中标记 is_truncated=True
    3. 为每个最终文本块生成 MD5 哈希作为业务唯一标识
    """

    def __init__(
        self,
        embedding_config: Optional[object] = None,
        chunking_config: Optional[object] = None,
    ):
        emb_cfg = embedding_config or config.embedding
        chunk_cfg = chunking_config or config.chunking

        self.max_chunk_size = chunk_cfg.max_chunk_size
        self.breakpoint_threshold_type = chunk_cfg.breakpoint_threshold_type
        self.breakpoint_threshold_amount = chunk_cfg.breakpoint_threshold_amount

        # 使用 MAAS 兼容的 Embeddings 实例（对标 text_embedding.py 调用方式）
        self.embeddings = MAASEmbeddings(
            base_url=emb_cfg.base_url,
            api_key=emb_cfg.api_key,
            model_name=emb_cfg.model_name,
            dim=emb_cfg.dim,
        )

        # 语义切分器（延后到 split_documents 时创建，以便注入进度包装）
        _sem_kwargs = {}
        if self.breakpoint_threshold_type:
            _sem_kwargs["breakpoint_threshold_type"] = self.breakpoint_threshold_type
        if self.breakpoint_threshold_amount is not None:
            _sem_kwargs["breakpoint_threshold_amount"] = self.breakpoint_threshold_amount
        self._sem_kwargs = _sem_kwargs
        self.semantic_chunker: Optional[SemanticChunker] = None

        # 兜底硬截断器（中文优先分隔符）
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_cfg.fallback_chunk_size,
            chunk_overlap=chunk_cfg.fallback_chunk_overlap,
            separators=chunk_cfg.fallback_separators,
        )

    @staticmethod
    def _count_sentences(text: str) -> List[str]:
        """按中英文句子分隔符拆分文本，返回句子列表（用于估算进度）"""
        # 匹配中英文句子结尾：。！？!? 换行
        sentences = re.split(r'(?<=[。！？!?\n])\s*', text)
        return [s for s in sentences if s.strip()]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表执行语义切分 + 兜底截断，返回最终的文本块列表。

        每个文本块的 metadata 包含：
          - 继承自原始文档的 source_path / source_file / file_type / page
          - chunk_index:       最终块序号（全局递增）
          - is_truncated:      是否经历过二次硬截断
          - semantic_split:    标记此块来自语义切分管道
        """
        if not documents:
            logger.warning("输入文档列表为空，跳过切分")
            return []

        # 先估算句子总数，用于进度条
        total_sentences = sum(len(self._count_sentences(d.page_content)) for d in documents)
        logger.info("开始语义切分，共 %d 个原始文档片段，约 %d 句（逐句调 API 检测语义边界）...",
                     len(documents), total_sentences)

        # 用进度包装器创建 SemanticChunker
        wrapped_embeddings = ProgressEmbeddings(self.embeddings, total_sentences)
        chunker = SemanticChunker(embeddings=wrapped_embeddings, **self._sem_kwargs)

        try:
            semantic_chunks = chunker.split_documents(documents)
        finally:
            wrapped_embeddings.close()

        logger.info("语义切分完成，得到 %d 个语义块", len(semantic_chunks))

        # —— 第二层：长度兜底 ——
        final_chunks: List[Document] = []
        truncation_count = 0

        for chunk in semantic_chunks:
            text = chunk.page_content
            if len(text) <= self.max_chunk_size:
                # 长度合规，直接保留
                final_chunks.append(chunk)
            else:
                # 超出阈值，二次硬截断
                sub_chunks = self.fallback_splitter.split_documents([chunk])
                for sub in sub_chunks:
                    sub.metadata.update(chunk.metadata)
                    sub.metadata["is_truncated"] = True
                final_chunks.extend(sub_chunks)
                truncation_count += 1

        if truncation_count > 0:
            logger.info(
                "兜底截断: %d 个语义块因超出 %d 字符被二次切分",
                truncation_count,
                self.max_chunk_size,
            )

        # —— 注入全局 chunk_index 并剔除空块 ——
        valid_chunks: List[Document] = []
        for idx, chunk in enumerate(final_chunks):
            text = chunk.page_content.strip()
            if not text:
                continue
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["semantic_split"] = True
            # 确保 is_truncated 字段存在
            if "is_truncated" not in chunk.metadata:
                chunk.metadata["is_truncated"] = False
            valid_chunks.append(chunk)

        logger.info("最终有效文本块: %d 个", len(valid_chunks))
        return valid_chunks

    # ========================================================================
    # MD5 业务唯一标识
    # ========================================================================

    @staticmethod
    def compute_md5(text: str) -> str:
        """计算文本内容的 MD5 哈希（用于去重）"""
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def assign_doc_ids(chunks: List[Document]) -> List[Document]:
        """
        为每个文本块计算并注入 doc_id（MD5 哈希）。
        doc_id 同时写入 metadata 和 page_content 的 prefix 不改变原内容。
        """
        for chunk in chunks:
            chunk.metadata["doc_id"] = AgriSemanticChunker.compute_md5(
                chunk.page_content
            )
        return chunks


# ============================================================================
# 便捷函数
# ============================================================================

def create_chunks(documents: List[Document]) -> List[Document]:
    """便捷函数：一步完成语义切分 + 兜底截断 + MD5 标识注入"""
    chunker = AgriSemanticChunker()
    chunks = chunker.split_documents(documents)
    chunks = AgriSemanticChunker.assign_doc_ids(chunks)
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.log.format)
    from document_loader import load_agriculture_documents

    docs = load_agriculture_documents()
    chunks = create_chunks(docs)
    for c in chunks[:5]:
        print(
            f"[doc_id={c.metadata['doc_id'][:8]}...] "
            f"file={c.metadata['source_file']} "
            f"len={len(c.page_content)} "
            f"trunc={c.metadata['is_truncated']} "
            f"preview={c.page_content[:80]}..."
        )
