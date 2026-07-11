# 智慧农业 AI 系统 — 后端脚手架设计

**日期**: 2026-07-10  
**范围**: P0 后端脚手架  
**状态**: 已确认

---

## 技术选型

| 维度 | 选择 |
|------|------|
| 后端框架 | Python FastAPI (async) |
| 业务数据库 | MySQL 8.0（远程连接） |
| 缓存 | Redis（远程连接） |
| 向量库 | Milvus（P1/P2 接入，P0 预留给接口） |
| 架构 | 单体 API，领域模块化 |
| 认证 | JWT Bearer Token |
| 文件存储 | 本地磁盘 |
| 开发环境 | 无 Docker，直接 uvicorn 启动，依赖外部服务 |

---

## 项目目录结构

```
service/
├── app/
│   ├── main.py                  # FastAPI 入口，挂载路由
│   ├── modules/                 # 业务模块（领域驱动）
│   │   ├── auth/                # 认证模块
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # /api/v1/auth/*
│   │   │   ├── service.py       # 登录/注册/刷新逻辑
│   │   │   ├── models.py        # User 表
│   │   │   └── schemas.py       # 请求/响应体
│   │   ├── disease/             # 病虫害识别模块
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # /api/v1/disease/*
│   │   │   ├── service.py       # 识别逻辑（P0 返回 mock）
│   │   │   ├── models.py        # DiseaseRecord 表
│   │   │   └── schemas.py
│   │   ├── knowledge/           # 知识库 RAG 模块
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # /api/v1/knowledge/*
│   │   │   ├── service.py       # 检索逻辑（P0 关键词匹配）
│   │   │   ├── models.py        # KnowledgeDoc, PestCatalog 表
│   │   │   └── schemas.py
│   │   ├── agent/               # 种植决策 Agent
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # /api/v1/agent/*
│   │   │   ├── service.py       # Agent 编排（P0 返回 mock）
│   │   │   └── schemas.py
│   │   └── business/            # 业务管理 CRUD
│   │       ├── __init__.py
│   │       ├── router.py        # /api/v1/business/*
│   │       ├── service.py
│   │       ├── models.py        # Farm, Crop 表
│   │       └── schemas.py
│   ├── core/                    # 共享基础设施
│   │   ├── __init__.py
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── security.py          # JWT 生成/校验, 密码哈希
│   │   ├── database.py          # SQLAlchemy async engine + session
│   │   ├── redis.py             # Redis async 连接
│   │   └── deps.py              # FastAPI 依赖注入（get_db, get_current_user）
│   └── common/                  # 通用工具
│       ├── __init__.py
│       ├── exceptions.py        # 全局异常处理
│       ├── response.py          # 统一响应格式
│       └── file_storage.py      # 文件上传/存储工具
├── migrations/                  # Alembic 迁移
├── uploads/                     # 本地文件存储
│   └── diseases/                # 病虫害识别图片
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

---

## 数据库模型

| 表名 | 核心字段 | 说明 |
|------|---------|------|
| `users` | id, username, password_hash, role(admin/expert/farmer), phone, avatar, created_at | 统一用户表，role 区分身份 |
| `farms` | id, farmer_id(FK→users), name, area, location, soil_type, created_at | 农田信息 |
| `crops` | id, farm_id(FK→farms), name, variety, plant_date, status, created_at | 作物种植记录 |
| `pest_catalog` | id, name, category(disease/pest/weed), symptoms, treatment, images(JSON), created_at | 病虫害图鉴（P0 建表，P2 加向量） |
| `disease_records` | id, farmer_id(FK→users), image_url, result_json(JSON), confidence, status(pending/confirmed/rejected), expert_id(FK→users), feedback, created_at | 识别记录 |
| `knowledge_docs` | id, title, content, category, source, created_at | 农技文档（P0 建表，P2 加向量+RAG） |

> P0 不涉及 Milvus，向量字段后续版本再加。

---

## API 路由

所有接口前缀 `/api/v1/`，统一响应格式：

```json
{ "code": 200, "message": "ok", "data": {...} }
```

| 模块 | 方法 | 路径 | 说明 | P0 状态 |
|------|------|------|------|---------|
| auth | POST | /auth/login | 登录返回 JWT | 完整实现 |
| auth | POST | /auth/register | 农户注册 | 完整实现 |
| auth | GET | /auth/me | 当前用户信息 | 完整实现 |
| disease | POST | /disease/recognize | 上传图片识别 | 返回 mock |
| disease | GET | /disease/records | 识别记录列表(分页) | 完整实现 |
| disease | GET | /disease/records/{id} | 识别详情 | 完整实现 |
| knowledge | GET | /knowledge/search | 知识搜索 | 关键词匹配 |
| knowledge | GET | /knowledge/catalog | 图鉴列表(分页) | 完整实现 |
| knowledge | GET | /knowledge/catalog/{id} | 图鉴详情 | 完整实现 |
| agent | POST | /agent/chat | 农技问答 | 返回 mock |
| business | GET/POST | /business/farmers | 农户管理(管理员) | 完整 CRUD |
| business | GET/POST/PUT | /business/farms | 农田管理 | 完整 CRUD |
| business | GET/POST/PUT | /business/crops | 作物管理 | 完整 CRUD |
| business | GET/POST/PUT/DELETE | /business/pests | 图鉴管理(管理员) | 完整 CRUD |
| business | GET | /business/disease-records | 识别记录管理(管理员) | 完整 CRUD |

---

## 核心基础设施

| 组件 | 选型 | 用途 |
|------|------|------|
| ORM | SQLAlchemy 2.0 (async) | MySQL 异步操作 |
| 迁移 | Alembic | DDL 版本管理 |
| Redis | redis-py (async) | 缓存、token 黑名单 |
| 配置 | pydantic-settings | .env 加载 |
| 密码 | passlib[bcrypt] | 密码哈希 |
| JWT | python-jose | token 签发验证 |
| 图片 | Pillow + python-multipart | 上传与预处理 |
| 服务启动 | uvicorn --reload | 开发 hot-reload |

---

## 分层交付计划

| 阶段 | 内容 |
|------|------|
| **P0（本次）** | 项目骨架 + DB 模型 + 认证 + 业务 CRUD + mock 接口 + 全局错误处理 |
| P1（后续） | 病虫害识别模型集成、真实推理 |
| P2（后续） | Milvus RAG 检索、Agent LLM 编排 |

---

## 不包含（P0 边界外）

- Milvus 向量存储接入
- AI 模型推理（返回 mock）
- LLM / Agent 编排
- 前端页面
- Docker 部署
- 单元测试（脚手架阶段先不写，后续补）
