"""
=============================================================================
文档加载与预处理模块
=============================================================================
利用 LangChain DirectoryLoader + 多格式加载器，将 ./data/ 下所有
PDF / DOCX / TXT 文件统一加载为 Document 对象，并提取文件路径等元数据。
"""

import os
import logging
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document
from tqdm import tqdm

from config import config

logger = logging.getLogger(__name__)


# ============================================================================
# 多格式 Loader 映射
# ============================================================================

def _pdf_loader(path: str) -> List[Document]:
    """PDF 加载器 —— 使用 PyPDFLoader 按页加载"""
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        logger.debug("PDF 加载完成 [%s]: %d 页", os.path.basename(path), len(docs))
        return docs
    except Exception as e:
        logger.error("PDF 加载失败 [%s]: %s", path, e)
        return []


def _docx_loader(path: str) -> List[Document]:
    """DOCX 加载器 —— 使用 Docx2txtLoader"""
    try:
        loader = Docx2txtLoader(path)
        docs = loader.load()
        logger.debug("DOCX 加载完成 [%s]: %d 个文档对象", os.path.basename(path), len(docs))
        return docs
    except Exception as e:
        logger.error("DOCX 加载失败 [%s]: %s", path, e)
        return []


def _txt_loader(path: str) -> List[Document]:
    """TXT 加载器 —— 使用 TextLoader（UTF-8 编码，兼容中文）"""
    try:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        logger.debug("TXT 加载完成 [%s]: %d 个文档对象", os.path.basename(path), len(docs))
        return docs
    except Exception as e:
        logger.error("TXT 加载失败 [%s]: %s", path, e)
        return []


# ============================================================================
# 统一加载入口
# ============================================================================

class AgriDocumentLoader:
    """
    农业资料统一加载器

    遍历 ./data/ 目录，根据后缀名分发到对应加载器，
    加载后统一注入元数据（source_file, source_path, file_type, page 等）。
    """

    def __init__(self, data_dir: str = config.DATA_DIR):
        self.data_dir = data_dir
        self._loaders = {
            ".pdf": _pdf_loader,
            ".docx": _docx_loader,
            ".doc": _docx_loader,
            ".txt": _txt_loader,
        }

    def load_all(self) -> List[Document]:
        """
        加载 data_dir 下所有支持的文档，返回 Document 列表。
        每个 Document 的 metadata 至少包含：
          - source_path:  文件完整路径
          - source_file:  文件名
          - file_type:    文件类型（pdf/docx/txt）
          - page:         页码（PDF 适用，其他类型为 0）
        """
        if not os.path.isdir(self.data_dir):
            logger.error("资料目录不存在: %s", self.data_dir)
            return []

        all_docs: List[Document] = []
        supported_extensions = tuple(self._loaders.keys())

        # 先收集所有需要处理的文件路径
        file_paths: List[tuple] = []
        for root, _dirs, files in os.walk(self.data_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in supported_extensions:
                    file_paths.append((os.path.join(root, fname), fname, ext))
                else:
                    logger.warning("跳过不支持的文件类型: %s", fname)

        for full_path, fname, ext in tqdm(file_paths, desc="加载文档", unit="文件"):
            logger.debug("正在加载: %s", fname)

            loader_fn = self._loaders[ext]
            docs = loader_fn(full_path)

            # —— 注入/补全元数据 ——
            for idx, doc in enumerate(docs):
                doc.metadata["source_path"] = full_path
                doc.metadata["source_file"] = fname
                doc.metadata["file_type"] = ext.lstrip(".")
                # PDF 加载器通常已自带 page 元数据；其他类型补齐
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 0
                # 记录在源文件中的片段序号
                doc.metadata["source_chunk_index"] = idx

            all_docs.extend(docs)

        logger.info("全部加载完成: 共 %d 个原始文档片段", len(all_docs))
        return all_docs


# ============================================================================
# 便捷函数
# ============================================================================

def load_agriculture_documents(data_dir: str = config.DATA_DIR) -> List[Document]:
    """便捷函数：加载农业资料目录下所有文档"""
    loader = AgriDocumentLoader(data_dir)
    return loader.load_all()


if __name__ == "__main__":
    # 快速测试
    logging.basicConfig(level=logging.INFO, format=config.log.format)
    docs = load_agriculture_documents()
    for d in docs[:5]:
        print(f"[{d.metadata.get('source_file')}] "
              f"page={d.metadata.get('page')} "
              f"preview={d.page_content[:80]}...")
