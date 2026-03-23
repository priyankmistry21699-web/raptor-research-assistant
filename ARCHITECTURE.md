# RAPTOR RAG Platform — Architecture

> System design for the production-grade, multi-tenant RAPTOR RAG platform.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                             │
│                                                                                  │
│   ┌──────────────────────┐    ┌──────────────────────┐                          │
│   │   Next.js Frontend   │    │   API Consumers       │                          │
│   │   (Tailwind + shadcn)│    │   (REST / SDK)        │                          │
│   │   Port: 3000         │    │                       │                          │
│   └──────────┬───────────┘    └──────────┬────────────┘                          │
│              │                            │                                       │
└──────────────┼────────────────────────────┼───────────────────────────────────────┘
               │ HTTPS                       │ HTTPS
               ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AUTH LAYER (Clerk)                                        │
│                                                                                  │
│   JWT verification · User identity · Workspace membership                        │
│   Middleware on every request                                                    │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │ Authenticated request
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (FastAPI)                                    │
│                         Port: 8000                                               │
│                                                                                  │
│   ┌─────────────┐  ┌────────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐ │
│   │  /documents  │  │  /chat     │  │ /retrieve │  │ /feedback│  │ /admin    │ │
│   │  Upload      │  │  Sessions  │  │ Search    │  │ Ratings  │  │ Config    │ │
│   │  Status      │  │  Messages  │  │ Tree      │  │ Stats    │  │ Models    │ │
│   │  Versions    │  │  Stream    │  │ Paper     │  │ Export   │  │ Jobs      │ │
│   └──────┬──────┘  └─────┬──────┘  └────┬──────┘  └────┬─────┘  └─────┬─────┘ │
│          │               │               │              │               │        │
│   ┌──────┴───────────────┴───────────────┴──────────────┴───────────────┴──────┐ │
│   │                     REQUEST MIDDLEWARE                                      │ │
│   │  Request ID · Structured Logging · Rate Limiting · Error Handling          │ │
│   └────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│  BUSINESS LOGIC  │  │  BACKGROUND     │  │  AI LAYER            │
│                  │  │  WORKERS        │  │                      │
│  Session Mgmt    │  │  (Celery)       │  │  Embeddings (BGE)    │
│  Retrieval       │  │                 │  │  Reranker (BGE)      │
│  Prompt Build    │  │  Ingest Doc     │  │  LLM Router          │
│  Citation Format │  │  Extract Text   │  │    (LiteLLM)         │
│  Feedback Store  │  │  Chunk          │  │  RAPTOR Tree Builder │
│  Preference Pair │  │  Embed          │  │  LoRA Inference      │
│                  │  │  Build Tree     │  │  DPO Fine-tuning     │
│                  │  │  Index Vectors  │  │                      │
│                  │  │  Fine-tune      │  │                      │
│                  │  │  Evaluate       │  │                      │
└────────┬─────────┘  └────────┬────────┘  └───────────┬──────────┘
         │                     │                        │
         ▼                     ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                               │
│                                                                                  │
│   ┌───────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────────────┐  │
│   │  PostgreSQL   │  │  Qdrant      │  │  Redis   │  │  MinIO / S3          │  │
│   │  Port: 5432   │  │  Port: 6333  │  │  Port:   │  │  Port: 9000          │  │
│   │               │  │              │  │  6379    │  │                      │  │
│   │  • users      │  │  • vectors   │  │          │  │  • uploaded PDFs     │  │
│   │  • workspaces │  │    (384-dim) │  │  • cache │  │  • processed files   │  │
│   │  • collections│  │  • metadata  │  │  • queue │  │  • LoRA adapters     │  │
│   │  • documents  │  │  • filtered  │  │    broker│  │  • RAPTOR tree files │  │
│   │  • doc_vers.  │  │    search    │  │  • rate  │  │                      │  │
│   │  • ingest_jobs│  │              │  │    limits│  │                      │  │
│   │  • sessions   │  │              │  │  • locks │  │                      │  │
│   │  • messages   │  │              │  │          │  │                      │  │
│   │  • feedback   │  │              │  │          │  │                      │  │
│   └───────────────┘  └──────────────┘  └──────────┘  └──────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING / OPS                                          │
│                                                                                  │
│   ┌───────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │  Prometheus   │  │  Grafana     │  │  Sentry      │  │  GitHub Actions  │  │
│   │  Metrics      │  │  Dashboards  │  │  Errors      │  │  CI/CD           │  │
│   └───────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Diagrams

### 2.1 Document Upload & Ingestion

```
User                   API                    Worker                 Storage
 │                      │                       │                      │
 │── POST /documents ──▶│                       │                      │
 │   (file + metadata)  │                       │                      │
 │                      │── Save file ─────────────────────────────────▶│ MinIO
 │                      │                       │                      │
 │                      │── INSERT documents ──────────────────────────▶│ Postgres
 │                      │── INSERT ingest_jobs ────────────────────────▶│ Postgres
 │                      │                       │                      │
 │                      │── dispatch task ─────▶│                      │
 │                      │                       │                      │
 │◀── 202 Accepted ────│                       │                      │
 │    {job_id, status}  │                       │                      │
 │                      │                       │── GET file ─────────▶│ MinIO
 │                      │                       │◀── file bytes ──────│
 │                      │                       │                      │
 │                      │                       │── extract text       │
 │                      │                       │── chunk (300-500 tok)│
 │                      │                       │── embed (BGE 384-d) │
 │                      │                       │                      │
 │                      │                       │── build RAPTOR tree  │
 │                      │                       │   (cluster sections, │
 │                      │                       │    generate summaries)│
 │                      │                       │                      │
 │                      │                       │── upsert vectors ───▶│ Qdrant
 │                      │                       │── save tree file ───▶│ MinIO
 │                      │                       │── UPDATE job status ─▶│ Postgres
 │                      │                       │   (status=completed)  │
 │                      │                       │                      │
 │── GET /documents     │                       │                      │
 │   /{id}/status ─────▶│── SELECT job ────────────────────────────────▶│ Postgres
 │◀── {status: done} ──│                       │                      │
```

### 2.2 Chat / Query Flow

```
User                   API                  Retriever              LLM              Storage
 │                      │                      │                    │                 │
 │── POST /chat ───────▶│                      │                    │                 │
 │   {message, session} │                      │                    │                 │
 │                      │── get/create session ──────────────────────────────────────▶│ Postgres
 │                      │── store user msg ──────────────────────────────────────────▶│ Postgres
 │                      │                      │                    │                 │
 │                      │── retrieve(query) ──▶│                    │                 │
 │                      │                      │── embed query      │                 │
 │                      │                      │── vector search ──────────────────── ▶│ Qdrant
 │                      │                      │◀── top-k chunks ──────────────────── │
 │                      │                      │── load RAPTOR tree ────────────────── ▶│ Redis/S3
 │                      │                      │── walk up tree     │                 │
 │                      │                      │   (chunk→section   │                 │
 │                      │                      │    →topic→paper)   │                 │
 │                      │◀── enriched chunks ──│                    │                 │
 │                      │                      │                    │                 │
 │                      │── build prompt ──────────────────────────▶│                 │
 │                      │   (system + context                       │                 │
 │                      │    + history + question)                  │                 │
 │                      │                                           │                 │
 │                      │── LLM inference ─────────────────────────▶│                 │
 │                      │   (LiteLLM router:                        │                 │
 │                      │    Ollama / Groq /                        │                 │
 │                      │    LoRA adapter)                          │                 │
 │                      │◀── answer ───────────────────────────────│                 │
 │                      │                                           │                 │
 │                      │── store assistant msg ─────────────────────────────────────▶│ Postgres
 │                      │                                           │                 │
 │◀── {answer,          │                                           │                 │
 │     citations,       │                                           │                 │
 │     session_id} ────│                                           │                 │
```

### 2.3 Feedback → Fine-tuning Loop

```
User                   API                  Worker                 Storage
 │                      │                      │                     │
 │── POST /feedback ───▶│                      │                     │
 │   {msg_id, type,     │                      │                     │
 │    correction}       │── INSERT feedback ──────────────────────── ▶│ Postgres
 │                      │── build pref pair ──────────────────────── ▶│ Postgres
 │◀── {ok} ────────────│                      │                     │
 │                      │                      │                     │
 │                      │── check threshold    │                     │
 │                      │   (>= min_feedback?) │                     │
 │                      │                      │                     │
 │                      │── dispatch finetune ▶│                     │
 │                      │                      │── load pref pairs ──▶│ Postgres
 │                      │                      │── load base model   │
 │                      │                      │   (4-bit quantized) │
 │                      │                      │── apply LoRA config │
 │                      │                      │── DPO training      │
 │                      │                      │   (TRL DPOTrainer)  │
 │                      │                      │── save adapter ────▶│ MinIO
 │                      │                      │── register model ──▶│ Postgres
 │                      │                      │                     │
 │                      │                      │ Next query auto-    │
 │                      │                      │ selects best model  │
```

---

## 3. Database Schema

```sql
-- Core multi-tenancy
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_id        VARCHAR(255) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    display_name    VARCHAR(255),
    role            VARCHAR(50) DEFAULT 'user',  -- admin, user, viewer
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE workspaces (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    owner_id        UUID REFERENCES users(id) ON DELETE CASCADE,
    settings        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE workspace_members (
    workspace_id    UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    role            VARCHAR(50) DEFAULT 'member',  -- owner, admin, member, viewer
    joined_at       TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workspace_id, user_id)
);

-- Document management
CREATE TABLE collections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    settings        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id   UUID REFERENCES collections(id) ON DELETE CASCADE,
    filename        VARCHAR(500) NOT NULL,
    content_type    VARCHAR(100),
    file_size_bytes BIGINT,
    s3_key          VARCHAR(1000),
    metadata        JSONB DEFAULT '{}',
    status          VARCHAR(50) DEFAULT 'uploaded',  -- uploaded, processing, indexed, failed
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE document_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID REFERENCES documents(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL DEFAULT 1,
    s3_key          VARCHAR(1000),
    chunk_count     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Ingestion pipeline
CREATE TABLE ingestion_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID REFERENCES documents(id) ON DELETE CASCADE,
    status          VARCHAR(50) DEFAULT 'pending',
    -- pending → validating → extracting → chunking → embedding → tree_building → indexing → completed / failed
    current_stage   VARCHAR(50),
    progress_pct    SMALLINT DEFAULT 0,
    chunk_count     INTEGER,
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Chat
CREATE TABLE chat_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    collection_id   UUID REFERENCES collections(id),  -- optional: scope to a collection
    title           VARCHAR(500),
    settings        JSONB DEFAULT '{}',  -- model, top_k, task, etc.
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role            VARCHAR(20) NOT NULL,  -- user, assistant, system
    content         TEXT NOT NULL,
    citations       JSONB DEFAULT '[]',
    model_used      VARCHAR(255),
    latency_ms      INTEGER,
    token_count     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Feedback & learning
CREATE TABLE feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id      UUID REFERENCES chat_messages(id) ON DELETE CASCADE,
    user_id         UUID REFERENCES users(id),
    feedback_type   VARCHAR(50) NOT NULL,
    -- helpful, incorrect, hallucination, correction
    correction_text TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE preference_pairs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feedback_id     UUID REFERENCES feedback(id),
    prompt          TEXT NOT NULL,
    chosen          TEXT NOT NULL,
    rejected        TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE training_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id),
    model_name      VARCHAR(255),
    base_model      VARCHAR(255),
    method          VARCHAR(50) DEFAULT 'dpo',
    status          VARCHAR(50) DEFAULT 'pending',
    pair_count      INTEGER,
    metrics         JSONB DEFAULT '{}',
    adapter_s3_key  VARCHAR(1000),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_documents_collection ON documents(collection_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_ingestion_jobs_document ON ingestion_jobs(document_id);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs(status);
CREATE INDEX idx_chat_sessions_workspace ON chat_sessions(workspace_id);
CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX idx_feedback_message ON feedback(message_id);
CREATE INDEX idx_feedback_type ON feedback(feedback_type);
CREATE INDEX idx_preference_pairs_feedback ON preference_pairs(feedback_id);
CREATE INDEX idx_training_runs_workspace ON training_runs(workspace_id);
```

---

## 4. New Project Structure

```
raptor-research-assistant/
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.worker
├── .env.example
├── .env
├── .gitignore
├── pyproject.toml
├── alembic.ini
├── README.md
├── ARCHITECTURE.md
├── PROJECT_ROADMAP.md
├── IMPLEMENTATION_PLAN.md
│
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 001_initial_schema.py
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                # Dependency injection (db session, current_user)
│   │   ├── middleware.py           # Request ID, logging, rate limit
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py            # Clerk webhook, user sync
│   │   │   ├── chat.py            # Chat sessions + messages
│   │   │   ├── documents.py       # Upload, list, status
│   │   │   ├── retrieve.py        # Search + tree browsing
│   │   │   ├── feedback.py        # Submit + query feedback
│   │   │   ├── train.py           # Fine-tuning + learning loop
│   │   │   ├── eval.py            # RAGAS evaluation
│   │   │   ├── admin.py           # System config, models, jobs
│   │   │   └── health.py          # /health/live, /health/ready
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── chat.py
│   │       ├── documents.py
│   │       ├── feedback.py
│   │       └── common.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Settings from env vars (Pydantic BaseSettings)
│   │   ├── security.py            # Clerk JWT verification
│   │   ├── exceptions.py          # Custom exception classes
│   │   └── logging.py             # Structured JSON logging setup
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py             # SQLAlchemy engine + session factory
│   │   ├── base.py                # Declarative base
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── user.py
│   │       ├── workspace.py
│   │       ├── collection.py
│   │       ├── document.py
│   │       ├── ingestion_job.py
│   │       ├── chat.py
│   │       ├── feedback.py
│   │       └── training.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_service.py    # Upload, versioning, status
│   │   ├── chat_service.py        # Session + message management
│   │   ├── retrieval_service.py   # Hybrid retrieval orchestration
│   │   ├── feedback_service.py    # Feedback + preference pairs
│   │   ├── training_service.py    # Fine-tuning orchestration
│   │   └── eval_service.py        # Evaluation orchestration
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── embeddings.py          # SentenceTransformers / BGE
│   │   ├── reranker.py            # BGE reranker
│   │   ├── llm_router.py          # LiteLLM routing
│   │   ├── prompt_builder.py      # Prompt construction
│   │   ├── raptor/
│   │   │   ├── __init__.py
│   │   │   ├── tree_builder.py    # Build RAPTOR trees
│   │   │   ├── tree_index.py      # Load/traverse trees
│   │   │   └── clustering.py      # Topic clustering
│   │   └── finetuning/
│   │       ├── __init__.py
│   │       ├── dpo_trainer.py     # DPO training logic
│   │       └── lora_inference.py  # LoRA adapter inference
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── s3_client.py           # MinIO / S3 operations
│   │   ├── vector_store.py        # Qdrant operations
│   │   └── cache.py               # Redis caching
│   │
│   └── workers/
│       ├── __init__.py
│       ├── celery_app.py          # Celery configuration
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── ingest.py          # Document ingestion pipeline
│       │   ├── finetune.py        # Model fine-tuning
│       │   └── evaluate.py        # Background evaluation
│       └── pipeline/
│           ├── __init__.py
│           ├── extract.py         # Text extraction (PDF, DOCX, TXT)
│           ├── chunk.py           # Text chunking
│           ├── embed.py           # Embedding generation
│           └── index.py           # Vector indexing
│
├── frontend/                       # Next.js app (separate)
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   ├── chat/
│   │   │   ├── documents/
│   │   │   ├── collections/
│   │   │   └── admin/
│   │   ├── components/
│   │   │   ├── ui/               # shadcn components
│   │   │   ├── chat/
│   │   │   ├── documents/
│   │   │   └── layout/
│   │   ├── lib/
│   │   │   ├── api.ts            # API client
│   │   │   └── utils.ts
│   │   └── hooks/
│   │       ├── useChat.ts
│   │       └── useDocuments.ts
│   └── public/
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_services/
│   │   ├── test_ai/
│   │   └── test_workers/
│   ├── integration/
│   │   ├── test_api/
│   │   ├── test_db/
│   │   └── test_storage/
│   └── e2e/
│       └── test_full_pipeline.py
│
├── scripts/
│   ├── seed_db.py
│   ├── migrate_from_v1.py        # Migrate existing ChromaDB data
│   └── benchmark.py
│
└── docs/
    ├── api.md
    ├── deployment.md
    └── development.md
```

---

## 5. Component Specifications

### 5.1 FastAPI Backend

| Component | Technology | Notes |
|-----------|-----------|-------|
| Framework | FastAPI 0.110+ | Async, OpenAPI docs |
| Validation | Pydantic v2 | Request/response models |
| Auth middleware | Clerk SDK | JWT verification on every request |
| Rate limiting | Redis + `slowapi` | Per-user, per-endpoint limits |
| Request ID | UUID4 middleware | Propagated in logs and responses |
| Error handling | Custom exception handlers | Consistent error response format |
| CORS | FastAPI CORS middleware | Allow frontend origin |

### 5.2 Database (PostgreSQL)

| Aspect | Detail |
|--------|--------|
| ORM | SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| Connection pool | `asyncpg` driver, pool size 20 |
| Naming convention | snake_case tables, UUID primary keys |
| Soft deletes | `deleted_at` column where needed |
| Audit | `created_at`, `updated_at` on all tables |

### 5.3 Vector Database (Qdrant)

| Aspect | Detail |
|--------|--------|
| Dimensions | 384 (all-MiniLM-L6-v2) or 768 (BGE-base) |
| Distance metric | Cosine |
| Collections | One per workspace-collection pair |
| Payload filtering | `document_id`, `section`, `topic` |
| Replication | Single node for dev, replicated for prod |

### 5.4 Object Storage (MinIO / S3)

| Bucket | Contents |
|--------|----------|
| `raptor-documents` | Uploaded files (PDFs, DOCX, TXT) |
| `raptor-trees` | Serialized RAPTOR tree files |
| `raptor-models` | LoRA adapter weights |

### 5.5 Background Workers (Celery)

| Queue | Tasks |
|-------|-------|
| `ingest` | Document ingestion pipeline |
| `train` | DPO fine-tuning runs |
| `eval` | Background evaluation jobs |
| `default` | Misc tasks |

### 5.6 Cache (Redis)

| Purpose | TTL | Key Pattern |
|---------|-----|-------------|
| Session cache | 24h | `session:{id}` |
| RAPTOR tree cache | 1h | `tree:{doc_id}` |
| Rate limit counters | 1min | `rl:{user_id}:{endpoint}` |
| Job status | 30min | `job:{id}:status` |
| Celery broker | N/A | Celery internal |

---

## 6. API Design

### Base URL: `/api/v1`

| Group | Endpoint | Method | Auth | Description |
|-------|---------|--------|------|-------------|
| **Health** | `/health/live` | GET | No | Process alive |
| | `/health/ready` | GET | No | Dependencies ready |
| **Auth** | `/auth/webhook` | POST | Clerk | User sync from Clerk |
| **Workspaces** | `/workspaces` | GET | Yes | List user's workspaces |
| | `/workspaces` | POST | Yes | Create workspace |
| | `/workspaces/{id}` | GET | Yes | Get workspace |
| **Collections** | `/workspaces/{wid}/collections` | GET | Yes | List collections |
| | `/workspaces/{wid}/collections` | POST | Yes | Create collection |
| **Documents** | `/collections/{cid}/documents` | POST | Yes | Upload document |
| | `/collections/{cid}/documents` | GET | Yes | List documents |
| | `/documents/{id}` | GET | Yes | Document details + status |
| | `/documents/{id}/status` | GET | Yes | Ingestion job status |
| **Chat** | `/chat/sessions` | POST | Yes | Create session |
| | `/chat/sessions/{id}` | GET | Yes | Get session + messages |
| | `/chat/sessions/{id}/messages` | POST | Yes | Send message |
| | `/chat/sessions/{id}/stream` | WS | Yes | Stream response |
| **Retrieve** | `/retrieve/search` | POST | Yes | Hybrid search |
| | `/retrieve/tree/{doc_id}` | GET | Yes | RAPTOR tree structure |
| **Feedback** | `/feedback` | POST | Yes | Submit feedback |
| | `/feedback/stats` | GET | Yes | Feedback statistics |
| **Training** | `/training/trigger` | POST | Admin | Trigger fine-tuning |
| | `/training/status` | GET | Admin | Training status |
| | `/training/models` | GET | Yes | List available models |
| **Eval** | `/eval/run` | POST | Admin | Run evaluation |
| | `/eval/history` | GET | Admin | Evaluation history |
| **Admin** | `/admin/config` | GET | Admin | System configuration |
| | `/admin/jobs` | GET | Admin | Active background jobs |

---

## 7. Security Architecture

```
┌──────────────────────────────────────────────┐
│               Security Layers                 │
├──────────────────────────────────────────────┤
│  1. HTTPS (TLS termination at load balancer) │
│  2. Clerk JWT verification (every request)   │
│  3. Workspace membership check               │
│  4. Role-based access (admin/member/viewer)  │
│  5. Rate limiting (Redis-backed)             │
│  6. Input validation (Pydantic)              │
│  7. SQL injection prevention (SQLAlchemy ORM)│
│  8. File type validation (upload)            │
│  9. Request ID + audit logging               │
│ 10. Secrets in env vars (not in code)        │
└──────────────────────────────────────────────┘
```

---

## 8. Deployment Architecture

### Development (Docker Compose)

```yaml
services:
  api:        FastAPI (port 8000)
  worker:     Celery worker
  postgres:   PostgreSQL 16 (port 5432)
  qdrant:     Qdrant (port 6333)
  redis:      Redis 7 (port 6379)
  minio:      MinIO (port 9000, console 9001)
  frontend:   Next.js (port 3000)
```

### Production (Future)

```
Load Balancer (Nginx / Cloud LB)
    ├── Frontend (Vercel / Cloud Run)
    ├── API (2+ replicas behind LB)
    ├── Workers (auto-scaled Celery)
    ├── PostgreSQL (managed: RDS / Cloud SQL)
    ├── Qdrant (managed or self-hosted)
    ├── Redis (managed: ElastiCache / Memorystore)
    └── S3 (managed object storage)
```

---

## 9. Migration Strategy (v1 → v2)

| v1 Component | v2 Replacement | Migration Path |
|-------------|---------------|----------------|
| ChromaDB (SQLite) | Qdrant | Export embeddings → reimport to Qdrant |
| In-memory sessions | PostgreSQL `chat_sessions` | Script to migrate any saved sessions |
| JSONL feedback | PostgreSQL `feedback` | Parse JSONL → INSERT into Postgres |
| JSONL preferences | PostgreSQL `preference_pairs` | Parse JSONL → INSERT into Postgres |
| File-based trees (.gpickle) | MinIO + Redis cache | Upload to MinIO, load on demand |
| Gradio UI | Next.js frontend | Complete rewrite |
| Direct Ollama/Groq calls | LiteLLM router | Wrap existing logic in LiteLLM |
| No auth | Clerk | Add middleware, user table |
| No background jobs | Celery | Move ingestion/training to tasks |

A migration script (`scripts/migrate_from_v1.py`) will handle data transfer.
