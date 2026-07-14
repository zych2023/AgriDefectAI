# 农业知识 RAG 模块

基于 LangChain + Milvus 的 Advanced RAG 知识检索模块，为智慧农业病虫害识别与种植决策系统提供知识库支持。

## 架构

```
离线：data/ → document_loader → chunker → 向量化 → Milvus 入库
在线：用户问题 → [查询扩展] → 混合检索 → RRF融合 → [精排] → 返回文档
```

- `[查询扩展]` — 可选，LLM 将口语扩展为多角度子问题（`config.py` 开关控制）
- `[精排]` — 可选，Cross-Encoder 对候选文档二次打分（默认关闭）

## 文件说明

| 文件 | 职责 |
|------|------|
| `config.py` | 所有参数集中管理，关键功能均有开关 |
| `document_loader.py` | PDF / DOCX / TXT 多格式加载，注入元数据 |
| `chunker.py` | 语义分块 + 超长兜底截断 + MD5 去重标识 |
| `maas_embedding.py` | 阿里云 MaaS Embedding 封装（兼容 LangChain 接口） |
| `milvus_client.py` | Milvus 连接、建表、双索引、去重插入、双路检索 |
| `retriever.py` | 稠密+稀疏混合检索 + RRF 倒数排名融合 |
| `reranker.py` | Cross-Encoder 精排，加载失败自动降级 |
| `query_expander.py` | LLM 查询扩展（口语→多角度专业子问题） |
| `generator.py` | LLM 流式生成 + 引用标注（仅交互模式用） |
| `main.py` | 知识库构建 + `QueryPipeline`（含 `answer()` 和 `retrieve()`） |
| `api_server.py` | FastAPI 服务，对外提供 5 个 HTTP 接口 |
| `qa_logger.py` | SQLite 问答记录，自动存 `/ask` 的调用历史 |
| `requirements.txt` | Python 依赖 |
| `data/` | 农业知识资料（扔 PDF/DOCX/TXT 进去） |
| `models/` | 下载的精排模型文件（gitignore，约 2.2GB） |
| `API_DOC.md` | HTTP API 完整调用文档 |

## 快速开始

### 1. 环境要求

- Python ≥ 3.9
- Docker（用于运行 Milvus 向量数据库）

### 2. 启动 Milvus

```bash
docker run -d --name milvus-standalone \
  -e ETCD_USE_EMBED=true \
  -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e COMMON_STORAGETYPE=local \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.5.4 milvus run standalone
```

> **注意：** 必须使用 `v2.5.4`，不要用 `latest`（v2.6.x 在 Windows Docker Desktop 上有兼容性问题）。

验证 Milvus 是否启动成功：

```bash
docker ps | grep milvus
```

### 3. 安装依赖

```bash
cd rag
pip install -r requirements.txt
```

### 4. 配置密钥

编辑 `config.py`，修改 `LLMConfig` 和 `EmbeddingConfig` 中的 `base_url`、`api_key`。

### 5. 准备资料

把农业知识文件（PDF、DOCX、TXT）放入 `data/`。

### 6. 构建知识库

```bash
python main.py --ingest-only
```

构建过程分 6 步，每步均有 `tqdm` 进度条：

| 步骤 | 耗时 | 说明 |
|------|------|------|
| [1/6] 加载文档 | <10s | 本地 I/O，读取 `data/` 下所有文件 |
| [2/6] 语义切分 | 1~4 min | 逐句调 Embedding API 检测语义边界，**最慢的步骤** |
| [3/6] 初始化 Collection | <5s | 在 Milvus 中建表、建索引 |
| [4/6] BM25 训练 | <10s | 本地分词 + 统计，语料规模越大越慢 |
| [5/6] 稠密向量 | 1~4 min | 逐批调 MaaS API，受网络延迟影响 |
| [6/6] 入库 | <5s | 写入 Milvus，自动 MD5 去重 |

> 约 30 份资料、500\~800 个文本块，总耗时通常 **3\~8 分钟**，大部分时间花在 API 调用上。

## 两种运行方式

### 方式 A：API 服务（推荐）

```bash
python api_server.py
# 或指定端口
python api_server.py --port 8080
```

服务默认监听 `0.0.0.0:8899`，浏览器打开 Swagger 文档：

```
http://127.0.0.1:8899/docs
```

| 接口 | 方法 | 说明 | 适用方 |
|------|------|------|--------|
| `/ask` | POST | 完整问答（检索+LLM 生成） | 前端用户 |
| `/retrieve` | POST | 纯检索，返回知识文档 | **外部 Agent** |
| `/history` | GET | 分页查问答历史 | 前端 |
| `/history/search` | GET | 关键词搜索历史 | 前端 |
| `/health` | GET | 健康检查 | 运维 |

`/ask` vs `/retrieve`：

| | `/ask` | `/retrieve` |
|------|--------|-------------|
| 返回 | LLM 生成的回答 | 原始知识文档列表 |
| 调 LLM | 是 | **否** |
| 耗时 | 3~8s | **1~2s** |
| 给谁用 | 直接展示给用户 | Agent 拿文档自己生成 |

### 方式 B：命令行交互

```bash
python main.py                  # 智能模式：自动检测知识库
python main.py --chat           # 跳过构建，直接对话
python main.py --rebuild        # 重建知识库后对话
python main.py --ingest-only    # 仅构建，不进入对话
```

## 检索流程

```
用户问题
  │
  ├── 查询扩展（关→直接用原始问题）
  │
  ├── 稠密检索(语义) ──┬── 并行 ──┐
  └── 稀疏检索(BM25) ──┘          │
       │                          │
       ▼                          ▼
  合并去重 → RRF 融合取 Top-10
       │
       ▼
  精排（关→直接用 RRF 前 2）
       │
       ▼
  返回 Top-2 文档片段: [{content, source_file, page, score}, ...]
```

### Agent 调用示例

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8899/retrieve",
    json={"fid": "agent_001", "question": "玉米叶子黄了怎么回事"},
    timeout=30
)
docs = resp.json()["documents"]
# [{"content": "...", "source_file": "玉米病害.txt", "page": 3, "score": 0.92}, ...]

# Agent 把 content 拼进自己的 prompt，用自己的 LLM 生成回答
context = "\n".join(d["content"] for d in docs)
```

详见 [`API_DOC.md`](API_DOC.md)。

## 配置参考

关键开关在 `config.py` → `RetrievalConfig`：

| 参数 | 默认 | 说明 |
|------|------|------|
| `query_expansion_enabled` | `False` | LLM 查询扩展开关 |
| `num_expanded_queries` | `2` | 扩展子问题数（开关开时才生效） |
| `reranker_enabled` | `False` | Cross-Encoder 精排开关 |
| `dense_top_k` | `20` | 稠密向量召回数 |
| `sparse_top_k` | `20` | 稀疏向量召回数 |
| `rrf_fusion_top_k` | `10` | RRF 融合后保留数 |
| `reranker_top_k` | `2` | 精排后返回数 |

## 增量添加资料

```bash
# 1. 把新文件放入 data/
# 2. 增量入库（已有数据 MD5 去重，不重复插入）
python main.py --ingest-only
# 3. 重启 API
python api_server.py
```

## 常见问题

**Milvus 连不上？**
```bash
docker ps | grep milvus    # 确认容器在跑
docker logs milvus-standalone | tail -20  # 查看日志
```
如果容器退出了，检查是否使用了正确的版本（`v2.5.4`，非 `latest`）。

**构建很慢正常吗？** 正常。步骤 [2/6] 语义切分和 [5/6] 稠密向量生成都需要调远程 Embedding API，每次网络往返 1~3 秒。30 份资料预计 3~8 分钟，进度条会实时显示当前进度。

**查询扩展开还是关？** 关→延迟 1-2s，混合检索本身召回已足够。开→口语化问题更全，但多一次 LLM 调用。

**怎么重建？**
```bash
python main.py --rebuild --ingest-only
```

**API 服务端口被占用？**
```bash
python api_server.py --port 8080    # 换端口
```
