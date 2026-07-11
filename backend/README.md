# SmartAgriAI — Intelligent Agriculture AI System

SmartAgriAI is an intelligent agriculture backend service built with Python FastAPI. It provides crop disease recognition, agricultural knowledge retrieval, AI-powered farming decisions, and farm business management capabilities.

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Backend Framework | Python FastAPI (async) |
| Database | MySQL 8.0 (remote) |
| Cache | Redis (remote) |
| ORM | SQLAlchemy 2.0 (async) |
| Migration | Alembic |
| Authentication | JWT Bearer Token (python-jose) |
| Password Hashing | passlib[bcrypt] |
| Architecture | Single API server, domain-modular monolith |

---

## Quick Start

### Prerequisites

- Python 3.11+
- A running MySQL 8.0 instance (remote)
- A running Redis instance (remote)

### Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Create environment configuration
cp .env.example .env

# 3. Edit .env with your database and Redis connection details
#    - DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
#    - REDIS_HOST, REDIS_PORT
#    - JWT_SECRET (change to a random string)
```

### Database Migration

```bash
alembic upgrade head
```

### Start Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Verify

```bash
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","app":"SmartAgriAI"}
```

---

## API Documentation

Once the server is running, interactive API docs are available at:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### API Overview

All endpoints are prefixed with `/api/v1/` and use a unified response format:

```json
{"code": 200, "message": "ok", "data": {...}}
```

| Module | Method | Endpoint | Description |
|--------|--------|----------|-------------|
| Auth | POST | `/api/v1/auth/register` | Register a new user (farmer / expert) |
| Auth | POST | `/api/v1/auth/login` | Login, returns JWT token |
| Auth | GET | `/api/v1/auth/me` | Get current user profile |
| Disease | POST | `/api/v1/disease/recognize` | Upload image for disease recognition |
| Disease | GET | `/api/v1/disease/records` | List my disease recognition records |
| Disease | GET | `/api/v1/disease/records/{id}` | Get disease record detail |
| Knowledge | GET | `/api/v1/knowledge/search` | Search pest catalog and knowledge docs |
| Knowledge | GET | `/api/v1/knowledge/catalog` | List pest catalog (paginated) |
| Knowledge | GET | `/api/v1/knowledge/catalog/{id}` | Get pest catalog detail |
| Agent | POST | `/api/v1/agent/chat` | AI-powered farming advice chat |
| Business | POST | `/api/v1/business/farms` | Create a farm |
| Business | GET | `/api/v1/business/farms` | List farms |
| Business | PUT | `/api/v1/business/farms/{id}` | Update a farm |
| Business | POST | `/api/v1/business/crops` | Create a crop record |
| Business | GET | `/api/v1/business/crops` | List crops |
| Business | PUT | `/api/v1/business/crops/{id}` | Update a crop |
| Business | POST | `/api/v1/business/pests` | Create pest catalog entry (admin/expert) |
| Business | GET | `/api/v1/business/pests` | List pest catalog |
| Business | PUT | `/api/v1/business/pests/{id}` | Update pest catalog (admin/expert) |
| Business | DELETE | `/api/v1/business/pests/{id}` | Delete pest catalog (admin only) |
| Business | GET | `/api/v1/business/disease-records` | List all disease records (admin/expert) |

---

## Project Structure

```
service/
├── app/
│   ├── main.py                      # FastAPI entry point, router mounting
│   ├── core/                        # Shared infrastructure layer
│   │   ├── config.py                # pydantic-settings (env-based config)
│   │   ├── database.py              # SQLAlchemy async engine & session
│   │   ├── redis.py                 # Redis async client
│   │   ├── security.py              # JWT creation/verification, password hashing
│   │   └── deps.py                  # FastAPI dependencies (auth, role check)
│   ├── common/                      # Common utilities
│   │   ├── response.py              # Unified ApiResponse + PaginatedData models
│   │   ├── exceptions.py            # Business exception classes + global handlers
│   │   └── file_storage.py          # File upload validation and saving
│   └── modules/                     # Domain modules (domain-driven)
│       ├── auth/                    # Authentication (register, login, profile)
│       │   ├── router.py
│       │   ├── service.py
│       │   ├── models.py            # User model
│       │   └── schemas.py
│       ├── disease/                 # Crop disease recognition
│       │   ├── router.py
│       │   ├── service.py
│       │   └── schemas.py
│       ├── knowledge/               # Agricultural knowledge base
│       │   ├── router.py
│       │   ├── service.py
│       │   └── schemas.py
│       ├── agent/                   # Farming decision AI agent
│       │   ├── router.py
│       │   ├── service.py
│       │   └── schemas.py
│       └── business/                # Business CRUD (farms, crops, pests, records)
│           ├── router.py
│           ├── service.py
│           ├── models.py            # Farm, Crop, PestCatalog, DiseaseRecord, KnowledgeDoc
│           └── schemas.py
├── migrations/                      # Alembic database migrations
│   ├── env.py
│   └── versions/
│       └── 0e340cb104be_init.py     # Initial migration (all tables)
├── uploads/                         # Local file storage
│   └── diseases/                    # Disease recognition images
├── tests/                           # Test directory (to be populated)
├── requirements.txt
├── .env.example                     # Environment variable template
├── alembic.ini                      # Alembic configuration
└── README.md
```

---

## Database Models

| Table | Key Fields | Description |
|-------|-----------|-------------|
| `users` | id, username, password_hash, role (admin/expert/farmer) | Unified user table |
| `farms` | id, farmer_id, name, area, location, soil_type | Farm information |
| `crops` | id, farm_id, name, variety, plant_date, status | Crop planting records |
| `pest_catalog` | id, name, category, symptoms, treatment, images | Pest/disease encyclopedia |
| `disease_records` | id, farmer_id, image_url, result_json, confidence, status | Disease recognition records |
| `knowledge_docs` | id, title, content, category, source | Agricultural knowledge documents |

---

## Development Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **P0** | Backend scaffold: project skeleton, DB models, JWT auth, business CRUD, mock disease recognition, mock agent chat, global error handling | Done |
| **P1** | Crop disease recognition model integration, real inference | Planned |
| **P2** | Milvus vector store + RAG knowledge retrieval, LLM-powered agent orchestration | Planned |

### P0 (Current) — Backend Scaffold

The current phase delivers:
- Complete project skeleton with domain-modular architecture
- JWT-based authentication (register, login, profile)
- Full CRUD for farms, crops, pest catalog, and disease records
- Mock disease recognition endpoint (returns predefined results)
- Keyword-based knowledge search
- Mock AI agent chat
- Unified `ApiResponse` format with global exception handling
- Alembic database migration support

### P1 — Real Disease Recognition

- Integrate a crop disease classification model
- Replace mock recognition with real model inference
- Enhance recognition accuracy and confidence scoring

### P2 — RAG & LLM Agent

- Integrate Milvus vector database
- Full RAG pipeline for knowledge retrieval
- LLM-powered agent for personalized farming recommendations
- Enhanced context-aware decision support

---

## Global Constraints

- **No Docker**: The service runs directly via uvicorn in development
- **Remote MySQL**: MySQL is not deployed locally; a remote instance is required
- **Remote Redis**: Redis is not deployed locally; a remote instance is required
- **No Milvus in P0**: Vector database will be introduced in P2
- **No unit tests in P0**: Tests are planned for subsequent phases

---

## License

Proprietary — Internal project.
