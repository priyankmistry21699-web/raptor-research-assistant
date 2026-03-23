# RAPTOR RAG Platform — Implementation Plan

> Exact step-by-step checklist for building the production platform. Each task has acceptance criteria.

---

## Day 1: Infrastructure + Data Model

### Task 1.1: Write `ARCHITECTURE.md`
- **Status:** ✅ Done
- **File:** `ARCHITECTURE.md`
- **Acceptance:** System diagram, data flow, DB schema, folder structure, API design documented

### Task 1.2: Refactor Project Structure
- **Status:** ⬜ Not started
- **Create new directories:**
  ```
  app/api/routes/          # Split routers into individual files
  app/api/schemas/         # Pydantic models
  app/core/config.py       # Pydantic BaseSettings
  app/core/security.py     # Clerk JWT verification
  app/core/exceptions.py   # Custom exceptions
  app/core/logging.py      # Structured logging
  app/db/                  # SQLAlchemy models + session
  app/db/models/           # One file per table
  app/db/session.py        # Engine + SessionLocal
  app/db/base.py           # Declarative base
  app/services/            # Business logic layer
  app/ai/                  # Embeddings, LLM, RAPTOR
  app/ai/raptor/           # Tree builder + index
  app/ai/finetuning/       # DPO + LoRA
  app/storage/             # S3, Qdrant, Redis clients
  app/workers/             # Celery app + tasks
  app/workers/tasks/       # Task definitions
  app/workers/pipeline/    # Ingestion pipeline stages
  alembic/                 # DB migrations
  alembic/versions/        # Migration files
  ```
- **Keep existing code working** — move files, update imports
- **Acceptance:** `python -m pytest tests/` still passes after restructure

### Task 1.3: Set Up Docker Compose
- **Status:** ⬜ Not started
- **File:** `docker-compose.yml`
- **Services:**

  | Service | Image | Ports | Volumes |
  |---------|-------|-------|---------|
  | `postgres` | `postgres:16-alpine` | `5432:5432` | `pgdata:/var/lib/postgresql/data` |
  | `qdrant` | `qdrant/qdrant:latest` | `6333:6333`, `6334:6334` | `qdrant_data:/qdrant/storage` |
  | `redis` | `redis:7-alpine` | `6379:6379` | `redis_data:/data` |
  | `minio` | `minio/minio:latest` | `9000:9000`, `9001:9001` | `minio_data:/data` |
  | `api` | Build from `Dockerfile` | `8000:8000` | App code mounted |
  | `worker` | Build from `Dockerfile.worker` | — | App code mounted |

- **File:** `Dockerfile`
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- **File:** `Dockerfile.worker`
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  CMD ["celery", "-A", "app.workers.celery_app", "worker", "--loglevel=info"]
  ```

- **Acceptance:** `docker compose up -d` starts all 6 services, `docker compose ps` shows all healthy

### Task 1.4: Set Up SQLAlchemy + Alembic
- **Status:** ⬜ Not started
- **Files to create:**
  - `app/db/session.py` — async engine, session factory
  - `app/db/base.py` — declarative base class
  - `alembic.ini` — Alembic config pointing to `DATABASE_URL`
  - `alembic/env.py` — migration environment
  - `alembic/script.py.mako` — migration template
- **Commands:**
  ```bash
  alembic init alembic
  # Edit alembic.ini → sqlalchemy.url = env:DATABASE_URL
  # Edit alembic/env.py → import all models
  ```
- **Acceptance:** `alembic current` runs without error

### Task 1.5: Create Initial DB Tables
- **Status:** ⬜ Not started
- **Files to create:**
  - `app/db/models/user.py` — `users` table
  - `app/db/models/workspace.py` — `workspaces` + `workspace_members`
  - `app/db/models/collection.py` — `collections`
  - `app/db/models/document.py` — `documents` + `document_versions`
  - `app/db/models/ingestion_job.py` — `ingestion_jobs`
  - `app/db/models/chat.py` — `chat_sessions` + `chat_messages`
  - `app/db/models/feedback.py` — `feedback` + `preference_pairs`
  - `app/db/models/training.py` — `training_runs`
- **Generate migration:**
  ```bash
  alembic revision --autogenerate -m "initial schema"
  alembic upgrade head
  ```
- **Acceptance:** All tables created in Postgres, `\dt` shows 11 tables

### Task 1.6: Add Config Management
- **Status:** ⬜ Not started
- **Files to create:**
  - `.env.example` — all env vars with placeholder values
  - `app/core/config.py` — Pydantic `BaseSettings` class
    ```python
    class Settings(BaseSettings):
        DATABASE_URL: str
        REDIS_URL: str = "redis://localhost:6379/0"
        QDRANT_URL: str = "http://localhost:6333"
        S3_ENDPOINT: str = "http://localhost:9000"
        S3_BUCKET: str = "raptor-documents"
        AWS_ACCESS_KEY_ID: str = "minioadmin"
        AWS_SECRET_ACCESS_KEY: str = "minioadmin"
        CLERK_SECRET_KEY: str = ""
        GROQ_API_KEY: str = ""
        SECRET_KEY: str = "change-me"
        DEBUG: bool = True
        LOG_LEVEL: str = "INFO"
        
        class Config:
            env_file = ".env"
    ```
- **Acceptance:** `from app.core.config import settings` works, loads from `.env`

---

## Day 2: Upload Flow + Worker + Auth + Observability

### Task 2.1: Build Upload Flow Skeleton
- **Status:** ⬜ Not started
- **File:** `app/api/routes/documents.py`
  ```python
  @router.post("/collections/{collection_id}/documents")
  async def upload_document(
      collection_id: UUID,
      file: UploadFile,
      db: AsyncSession = Depends(get_db),
      current_user: User = Depends(get_current_user),
  ):
      # 1. Validate file type
      # 2. Save to MinIO
      # 3. Create document record
      # 4. Create ingestion job
      # 5. Dispatch Celery task
      # 6. Return 202 Accepted
  ```
- **Acceptance:** POST with a PDF returns 202 + job_id

### Task 2.2: Save Files to MinIO/S3
- **Status:** ⬜ Not started
- **File:** `app/storage/s3_client.py`
  ```python
  class S3Client:
      def __init__(self, endpoint, access_key, secret_key, bucket):
          ...
      async def upload_file(self, key: str, data: bytes, content_type: str) -> str:
          ...
      async def download_file(self, key: str) -> bytes:
          ...
      async def delete_file(self, key: str):
          ...
  ```
- **Acceptance:** File uploaded via API appears in MinIO console (localhost:9001)

### Task 2.3: Create DB Records for Upload
- **Status:** ⬜ Not started
- **File:** `app/services/document_service.py`
  ```python
  class DocumentService:
      async def create_document(self, collection_id, filename, s3_key, ...) -> Document:
          ...
      async def create_ingestion_job(self, document_id) -> IngestionJob:
          ...
      async def get_document_status(self, document_id) -> dict:
          ...
  ```
- **Acceptance:** Document + job records visible in Postgres

### Task 2.4: Trigger Background Worker
- **Status:** ⬜ Not started
- **Files:**
  - `app/workers/celery_app.py` — Celery config (Redis broker)
  - `app/workers/tasks/ingest.py` — Ingestion task
  ```python
  @celery_app.task(bind=True)
  def ingest_document(self, job_id: str):
      # Calls pipeline stages sequentially
      ...
  ```
- **Acceptance:** Celery worker picks up dispatched task, logs progress

### Task 2.5: Build Worker Pipeline Skeleton
- **Status:** ⬜ Not started
- **Files:**
  - `app/workers/pipeline/extract.py` — `extract_text(file_bytes, content_type) -> str`
  - `app/workers/pipeline/chunk.py` — `chunk_text(text, max_tokens=500) -> List[str]`
  - `app/workers/pipeline/embed.py` — `embed_chunks(chunks) -> List[List[float]]`
  - `app/workers/pipeline/index.py` — `index_vectors(doc_id, chunks, embeddings)`
- **Pipeline stages with job status updates:**
  ```
  pending → validating → extracting → chunking → embedding
         → tree_building → indexing → completed
  ```
- **Acceptance:** Upload a PDF → worker processes all stages → document status = completed

### Task 2.6: Add Auth Skeleton
- **Status:** ⬜ Not started
- **File:** `app/core/security.py`
  ```python
  async def verify_clerk_token(token: str) -> dict:
      # Verify JWT with Clerk public key
      ...
  
  async def get_current_user(
      authorization: str = Header(...),
      db: AsyncSession = Depends(get_db),
  ) -> User:
      # Extract + verify token
      # Get or create user record
      ...
  ```
- **File:** `app/api/deps.py` — Dependency injection
- **Acceptance:** Protected endpoints return 401 without valid token

### Task 2.7: Add Health Endpoints
- **Status:** ⬜ Not started
- **File:** `app/api/routes/health.py`
  ```python
  @router.get("/health/live")
  async def liveness():
      return {"status": "alive"}
  
  @router.get("/health/ready")
  async def readiness(db: AsyncSession = Depends(get_db)):
      checks = {
          "postgres": await check_postgres(db),
          "qdrant": await check_qdrant(),
          "redis": await check_redis(),
          "minio": await check_minio(),
      }
      all_ok = all(checks.values())
      return {"status": "ready" if all_ok else "degraded", "checks": checks}
  ```
- **Acceptance:** `/health/ready` returns status of all 4 dependencies

### Task 2.8: Add Observability Basics
- **Status:** ⬜ Not started
- **Files:**
  - `app/core/logging.py` — JSON structured logging
  - `app/api/middleware.py` — Request ID + timing middleware
  - Sentry SDK integration in `app/main.py`
- **Acceptance:** Logs include request_id, latency_ms, user_id; errors reported to Sentry

---

## Week 1 MVP Checklist

| # | Task | Depends On | Status |
|---|------|-----------|--------|
| 1 | Clerk auth integration (sign in/up) | Day 2 auth skeleton | ⬜ |
| 2 | Workspace CRUD API | Auth | ⬜ |
| 3 | Collection CRUD API | Workspace | ⬜ |
| 4 | Document upload (full flow) | Collection + S3 + Worker | ⬜ |
| 5 | Ingestion status tracking API | Upload | ⬜ |
| 6 | Text extraction (PDF, DOCX, TXT) | Worker pipeline | ⬜ |
| 7 | Smart chunking (300-500 tokens) | Text extraction | ⬜ |
| 8 | Embedding generation (BGE/MiniLM) | Chunking | ⬜ |
| 9 | Qdrant vector indexing | Embeddings | ⬜ |
| 10 | RAPTOR tree generation | Indexing | ⬜ |
| 11 | Hybrid retrieval API (vector + tree) | Qdrant + tree | ⬜ |
| 12 | Answer generation (LiteLLM) | Retrieval | ⬜ |
| 13 | Citation formatting | Answer gen | ⬜ |
| 14 | Chat session persistence (Postgres) | DB | ⬜ |
| 15 | Next.js frontend scaffolding | Auth | ⬜ |
| 16 | Chat UI component | Frontend + Chat API | ⬜ |
| 17 | Document upload UI | Frontend + Upload API | ⬜ |

---

## Week 2–3 Checklist

| # | Task | Status |
|---|------|--------|
| 1 | BGE reranker integration | ⬜ |
| 2 | Better PDF parsing (tables, figures) | ⬜ |
| 3 | Retry logic for LLM calls | ⬜ |
| 4 | Retry logic for ingestion failures | ⬜ |
| 5 | Real-time job progress (WebSocket/SSE) | ⬜ |
| 6 | Rate limiting per user | ⬜ |
| 7 | Input validation hardening | ⬜ |
| 8 | File size limits + type restrictions | ⬜ |
| 9 | Better citation metadata (page numbers) | ⬜ |
| 10 | Collection-scoped retrieval | ⬜ |
| 11 | Session title auto-generation | ⬜ |
| 12 | Chat message streaming (WebSocket) | ⬜ |

---

## Month 1 Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Audit log table + middleware | ⬜ |
| 2 | Role-based access control (admin/member/viewer) | ⬜ |
| 3 | Production S3 configuration | ⬜ |
| 4 | Secret management (Vault/AWS) | ⬜ |
| 5 | Automated DB backup jobs | ⬜ |
| 6 | Disaster recovery plan + test | ⬜ |
| 7 | GitHub Actions CI pipeline (lint + test + build) | ⬜ |
| 8 | GitHub Actions CD pipeline (deploy) | ⬜ |
| 9 | Integration test suite | ⬜ |
| 10 | E2E test suite | ⬜ |
| 11 | Automated eval framework | ⬜ |
| 12 | Admin dashboard (models, jobs, users) | ⬜ |
| 13 | Model registry with versioning | ⬜ |

---

## Month 2 Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Prometheus metrics endpoint | ⬜ |
| 2 | Grafana dashboards | ⬜ |
| 3 | OpenTelemetry distributed tracing | ⬜ |
| 4 | Per-query cost tracking | ⬜ |
| 5 | Canary deployment setup | ⬜ |
| 6 | Feedback review + approval flow | ⬜ |
| 7 | Prompt versioning + A/B testing | ⬜ |
| 8 | Tenant data encryption at rest | ⬜ |
| 9 | Data retention + cleanup policies | ⬜ |
| 10 | Compliance controls (GDPR basics) | ⬜ |

---

## Dependency Requirements (New)

Add to `requirements.txt`:

```
# Database
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29
alembic>=1.13

# Background jobs
celery[redis]>=5.3
redis>=5.0

# Object storage
boto3>=1.34
# or minio>=7.2

# Vector DB
qdrant-client>=1.7

# Auth
clerk-backend-api>=0.5
pyjwt[crypto]>=2.8

# LLM routing
litellm>=1.30

# Reranking
# FlagEmbedding>=1.2  (for BGE reranker)

# Monitoring
sentry-sdk[fastapi]>=1.40
prometheus-fastapi-instrumentator>=6.1
python-json-logger>=2.0

# Existing (keep)
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.5
sentence-transformers>=2.3
chromadb>=0.4       # Keep for v1 compatibility during migration
torch>=2.1
transformers>=4.36
trl>=0.7
peft>=0.7
bitsandbytes>=0.41
gradio>=4.0         # Keep for v1 compatibility during migration
PyMuPDF>=1.23
scikit-learn>=1.3
networkx>=3.2
ragas>=0.4
litellm>=1.30
pyyaml>=6.0
python-dotenv>=1.0
httpx>=0.26
pytest>=7.4
pytest-asyncio>=0.23
```

---

## Accounts & Credentials Creation Guide

### Step 1: Clerk (Auth)
1. Go to https://clerk.com → Sign up
2. Create application → "RAPTOR RAG Platform"
3. Enable Email + Google sign-in
4. Copy `CLERK_SECRET_KEY` and `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`

### Step 2: Sentry (Error Tracking)
1. Go to https://sentry.io → Sign up
2. Create project → Python / FastAPI
3. Copy `SENTRY_DSN`

### Step 3: LLM Provider
- Already have: Groq API key, Ollama local
- Optional: Get OpenAI / Anthropic keys for production

### Step 4: Local Services (Docker)
- PostgreSQL, Qdrant, Redis, MinIO all run via Docker Compose
- No external accounts needed for development

### Step 5: GitHub
- Repository already exists: `priyankmistry21699-web/raptor-research-assistant`
- Set up GitHub Actions secrets for CI/CD

---

## Quick Start (After Setup)

```bash
# 1. Clone and configure
git clone https://github.com/priyankmistry21699-web/raptor-research-assistant.git
cd raptor-research-assistant
cp .env.example .env
# Edit .env with your credentials

# 2. Start infrastructure
docker compose up -d postgres qdrant redis minio

# 3. Run migrations
alembic upgrade head

# 4. Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Start worker
celery -A app.workers.celery_app worker --loglevel=info

# 6. Start frontend (separate terminal)
cd frontend && npm run dev
```
