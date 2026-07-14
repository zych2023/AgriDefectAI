"""
=============================================================================
智慧农业AI病虫害识别与种植决策系统 — Advanced RAG 统一配置模块
=============================================================================
所有可调参数集中于此，方便后期调优与维护。

LLM 对话模型 和 Embedding 嵌入模型 各自独立配置，
支持不同的 base_url / api_key / model_name。
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

# 自动加载 rag/.env（如果存在）
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_ENV_PATH):
    load_dotenv(_ENV_PATH)

# ============================================================================
# 项目路径
# ============================================================================
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")


# ============================================================================
# LLM 对话模型 API 配置
# ============================================================================
@dataclass
class LLMConfig:
    """大语言模型（对话 / 查询扩展 / 答案生成）API 配置 —— 请通过环境变量或 .env 设置"""
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""            # 必须通过 AGRI_LLM_API_KEY 环境变量设置
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.3
    max_tokens: int = 2048
    streaming: bool = True


# ============================================================================
# Embedding 嵌入模型 API 配置（独立于 LLM）
# ============================================================================
@dataclass
class EmbeddingConfig:
    """Embedding 向量化模型 API 配置 —— 请通过环境变量或 .env 设置"""
    base_url: str = (
        "https://ws-2fky9xuq9x0cdg19.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    api_key: str = ""            # 必须通过 AGRI_EMBEDDING_API_KEY 环境变量设置
    model_name: str = "text-embedding-v4"
    dim: int = 1024


# ============================================================================
# Milvus 连接配置
# ============================================================================
@dataclass
class MilvusConfig:
    """Milvus 向量数据库连接配置"""
    uri: str = "http://127.0.0.1:19530"
    db_name: str = "default"
    collection_name: str = "agri_knowledge_base"
    # 稠密向量索引
    dense_index_type: str = "IVF_FLAT"
    dense_metric_type: str = "COSINE"
    dense_nlist: int = 128
    # 稀疏向量索引
    sparse_index_type: str = "SPARSE_INVERTED_INDEX"
    sparse_metric_type: str = "IP"            # 手动稀疏向量用 IP（BM25 仅限内置 Function）
    # 连接超时（秒）
    timeout: int = 30


# ============================================================================
# 文档加载与分块配置
# ============================================================================
@dataclass
class ChunkingConfig:
    """文本分块策略配置"""
    # SemanticChunker 配置
    breakpoint_threshold_type: str = "percentile"   # "percentile" | "standard_deviation" | "interquartile"
    breakpoint_threshold_amount: Optional[float] = None  # None 使用默认值（percentile=95, sd=3, iqr=1.5）

    # 最大分块长度（字符数），超过则启动 RecursiveCharacterTextSplitter 兜底
    max_chunk_size: int = 1024
    # RecursiveCharacterTextSplitter 兜底切分参数
    fallback_chunk_size: int = 1000
    fallback_chunk_overlap: int = 50
    # 兜底切分分隔符（中文优先）
    fallback_separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""
    ])


# ============================================================================
# 检索配置
# ============================================================================
@dataclass
class RetrievalConfig:
    """检索与重排序配置"""
    # 每路检索召回的候选数量
    dense_top_k: int = 20        # 稠密向量路召回数
    sparse_top_k: int = 20       # 稀疏向量路召回数

    # RRF 融合参数
    rrf_k: int = 60              # RRF 平滑参数
    rrf_fusion_top_k: int = 10   # RRF 融合后保留的候选数（精排候选数，值越小越快）

    # Cross-Encoder 重排序
    reranker_enabled: bool = False  # 总开关：False=跳过精排，True=启用
    # 优先使用本地模型目录（下载后替换），留空则从 HuggingFace 在线加载
    reranker_model_name: str = ""   # 留空 = 使用下方 hf_model
    reranker_local_path: str = os.path.join(BASE_DIR, "models", "bge-reranker-v2-m3")
    reranker_hf_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 2      # 重排序后选入上下文的 Top-N
    reranker_device: str = "cuda"  # "cpu" | "cuda" | "cuda:0"
    # HuggingFace 镜像（国内网络加速，设为空字符串则不使用）
    hf_endpoint: str = "https://hf-mirror.com"

    # 查询扩展
    query_expansion_enabled: bool = False  # 总开关：True=LLM查询扩展，False=原始问题直接检索
    num_expanded_queries: int = 2  # 扩展生成的子问题数量（不含原问题）

    # 相似度阈值（0~1，小于此值的候选被过滤）
    similarity_threshold: float = 0.0


# ============================================================================
# 日志配置
# ============================================================================
@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"          # DEBUG | INFO | WARNING | ERROR
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# 全局配置实例（单例模式）
# ============================================================================
class _GlobalConfig:
    """全局配置聚合器，模块级直接引用"""

    # 项目路径（模块级常量也暴露在此）
    BASE_DIR: str
    DATA_DIR: str

    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.llm = LLMConfig()              # 对话模型
        self.embedding = EmbeddingConfig()   # 嵌入模型（独立）
        self.milvus = MilvusConfig()
        self.chunking = ChunkingConfig()
        self.retrieval = RetrievalConfig()
        self.log = LogConfig()
        # 自动从 .env / 环境变量覆盖敏感配置（如 api_key）
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """从环境变量覆盖关键配置（__init__ 自动调用）"""
        self.override_from_env()

    def override_from_env(self) -> None:
        """从环境变量覆盖关键配置"""
        # LLM 配置
        for attr_name in ["base_url", "api_key", "model_name"]:
            env_key = f"AGRI_LLM_{attr_name.upper()}"
            if os.getenv(env_key):
                setattr(self.llm, attr_name, os.getenv(env_key))

        # Embedding 配置
        for attr_name in ["base_url", "api_key", "model_name"]:
            env_key = f"AGRI_EMBEDDING_{attr_name.upper()}"
            if os.getenv(env_key):
                setattr(self.embedding, attr_name, os.getenv(env_key))
        if os.getenv("AGRI_EMBEDDING_DIM"):
            self.embedding.dim = int(os.getenv("AGRI_EMBEDDING_DIM"))

        # Milvus 配置
        if os.getenv("MILVUS_URI"):
            self.milvus.uri = os.getenv("MILVUS_URI")
        if os.getenv("MILVUS_COLLECTION"):
            self.milvus.collection_name = os.getenv("MILVUS_COLLECTION")


config = _GlobalConfig()
