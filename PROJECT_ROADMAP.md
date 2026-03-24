# RAPTOR RAG Platform — Project Roadmap

> Transform `raptor-research-assistant` from a research demo into an **industry-standard generic RAPTOR RAG platform** where users upload their own data and receive grounded answers with citations.

---

## Current State (v1 — Research Demo)

| Aspect                                 | Status                        |
| -------------------------------------- | ----------------------------- |
| 18-step RAPTOR pipeline                | ✅ Complete                   |
| 204 ML/DL papers indexed               | ✅ 148,986 chunks in ChromaDB |
| Hybrid retrieval (vector + tree)       | ✅ Working                    |
| Multi-model LLM (Ollama + Groq + LoRA) | ✅ Working                    |
| Feedback → DPO fine-tuning loop        | ✅ Working                    |
| RAGAS evaluation                       | ✅ Working                    |
| Gradio UI (4 tabs)                     | ✅ Working                    |
| FastAPI backend (50+ endpoints)        | ✅ Working                    |
| 208 tests                              | ✅ Passing                    |
| Auth / multi-tenant                    | ❌ None                       |
| Docker / CI-CD                         | ❌ None                       |
| Persistent DB                          | ❌ In-memory + JSONL files    |
| Monitoring                             | ❌ None                       |

---

## Target State (v2 — Production Platform)

A multi-tenant, containerized, observable RAG platform with:

- User authentication (Clerk)
- Workspace/collection isolation
- Upload any document (PDF, DOCX, TXT)
- Background ingestion (parse → chunk → embed → RAPTOR tree → index)
- Retrieval + generation with citations
- Chat persistence in PostgreSQL
- Feedback + preference-based fine-tuning
- Job progress tracking
- Health checks + monitoring
- CI/CD + Docker deployment

---

## Recommended Stack

### Frontend

| Technology   | Purpose                       |
| ------------ | ----------------------------- |
| Next.js      | React framework, SSR, routing |
| Tailwind CSS | Utility-first styling         |
| shadcn/ui    | Accessible component library  |

### Backend

| Technology | Purpose                     |
| ---------- | --------------------------- |
| FastAPI    | Async API framework         |
| Pydantic   | Request/response validation |
| SQLAlchemy | ORM for PostgreSQL          |
| Alembic    | Database migrations         |

### Data / Infrastructure

| Technology | Purpose                                        |
| ---------- | ---------------------------------------------- |
| PostgreSQL | Persistent relational DB                       |
| Qdrant     | Production vector database (replaces ChromaDB) |
| Redis      | Caching, session store, job queue broker       |
| S3 / MinIO | File storage (uploaded documents)              |

### Background Jobs

| Technology | Purpose                |
| ---------- | ---------------------- |
| Celery     | Distributed task queue |

### AI Layer

| Technology                 | Purpose                         |
| -------------------------- | ------------------------------- |
| SentenceTransformers / BGE | Embeddings                      |
| BGE reranker               | Re-ranking retrieved results    |
| LiteLLM                    | Unified LLM routing             |
| vLLM (optional)            | High-throughput local inference |
| Ollama                     | Local dev only                  |

### Auth

| Technology        | Purpose                          |
| ----------------- | -------------------------------- |
| Clerk (preferred) | Authentication + user management |

### Monitoring / Ops

| Technology              | Purpose            |
| ----------------------- | ------------------ |
| Prometheus              | Metrics collection |
| Grafana                 | Dashboards         |
| Sentry                  | Error tracking     |
| GitHub Actions          | CI/CD              |
| Docker / Docker Compose | Containerization   |

---

## Implementation Order

| Phase | Focus                                 | Priority    |
| ----- | ------------------------------------- | ----------- |
| 1     | Infrastructure (Docker, DB, services) | 🔴 Critical |
| 2     | Data model (SQLAlchemy + Alembic)     | 🔴 Critical |
| 3     | Authentication (Clerk)                | 🔴 Critical |
| 4     | Upload / storage (MinIO)              | 🟠 High     |
| 5     | Background indexing (Celery)          | 🟠 High     |
| 6     | Retrieval (Qdrant)                    | 🟠 High     |
| 7     | Generation (LiteLLM)                  | 🟠 High     |
| 8     | Citations                             | 🟡 Medium   |
| 9     | Feedback                              | 🟡 Medium   |
| 10    | Observability (Prometheus, Sentry)    | 🟡 Medium   |
| 11    | Backups                               | 🟢 Low      |
| 12    | CI/CD                                 | 🟢 Low      |
| 13    | Security hardening                    | 🟢 Low      |

---

## 2-Day Foundation Goal

By end of 2 days, have:

- [x] Architecture finalized (`ARCHITECTURE.md`)
- [ ] Project structure refactored
- [ ] Docker Compose running (postgres, qdrant, redis, minio, api, worker)
- [ ] PostgreSQL running
- [ ] Qdrant running
- [ ] Redis running
- [ ] MinIO running
- [ ] FastAPI connected to Postgres
- [ ] Alembic initialized
- [ ] Auth strategy selected (Clerk)
- [ ] Upload API stub created
- [ ] Worker skeleton created
- [ ] Env/secrets prepared (`.env.example`)
- [ ] Health checks added (`/health/live`, `/health/ready`)

---

## Day 1 Tasks

1. **Write `ARCHITECTURE.md`** — System design, component diagram, data flow
2. **Refactor project structure** — New folder layout for production
3. **Set up Docker Compose** with services:
   - `postgres` (port 5432)
   - `qdrant` (port 6333)
   - `redis` (port 6379)
   - `minio` (port 9000/9001)
   - `api` (port 8000)
   - `worker` (Celery)
4. **Set up SQLAlchemy + Alembic**
5. **Create initial DB tables:**
   - `users`
   - `workspaces`
   - `collections`
   - `documents`
   - `document_versions`
   - `ingestion_jobs`
   - `chat_sessions`
   - `chat_messages`
   - `feedback`
6. **Add config management:**
   - `.env.example` with all required variables
   - Settings loader (`app/core/config.py`)

---

## Day 2 Tasks

1. **Build upload flow skeleton** — POST endpoint accepting files
2. **Save files to MinIO/S3** — Store raw uploaded documents
3. **Create DB record** for uploaded document
4. **Create ingestion job record** — Track processing state
5. **Trigger background worker** — Celery task dispatch
6. **Build worker skeleton stages:**
   - Validate file
   - Extract text (PDF, DOCX, TXT)
   - Chunk text
   - Generate embeddings
   - Build RAPTOR tree
   - Save vectors to Qdrant
   - Update job state (pending → processing → done/failed)
7. **Add auth skeleton** — Clerk middleware stub
8. **Add health endpoints:**
   - `GET /health/live` — Is the API process alive?
   - `GET /health/ready` — Are all dependencies connected?
9. **Add observability basics:**
   - Sentry integration
   - Structured JSON logging
   - Request ID middleware

---

## Week 1 MVP Goals

| Feature                | Description                         |
| ---------------------- | ----------------------------------- |
| Sign in / auth         | Clerk-based login                   |
| Workspace support      | Isolated data per workspace         |
| Document upload        | PDF/DOCX/TXT via UI or API          |
| Ingestion status       | Job progress tracking               |
| Parsing + chunking     | PyMuPDF + smart chunking            |
| Embeddings             | BGE or SentenceTransformers         |
| Qdrant indexing        | Vector storage with metadata        |
| RAPTOR tree generation | Hierarchical clustering + summaries |
| Retrieval API          | Hybrid vector + tree search         |
| Answer generation      | LiteLLM-routed LLM calls            |
| Citations              | Source attribution in responses     |
| Chat persistence       | PostgreSQL-backed sessions          |

---

## Week 2–3 Goals

| Feature                     | Description                          |
| --------------------------- | ------------------------------------ |
| Reranking                   | BGE reranker for better precision    |
| Better parsing              | Handle tables, figures, multi-column |
| Retries / error handling    | Resilient ingestion + LLM calls      |
| Job progress tracking       | Real-time status updates             |
| Security basics             | Input validation, rate limiting      |
| Better citation metadata    | Page numbers, section references     |
| Collection-scoped retrieval | Search within specific collections   |

---

## Month 1 Goals

| Feature           | Description                        |
| ----------------- | ---------------------------------- |
| Audit logs        | Track all user actions             |
| Role-based access | Admin / editor / viewer roles      |
| Production S3     | AWS S3 or managed MinIO            |
| Secret management | Vault or AWS Secrets Manager       |
| Backup jobs       | Automated DB + vector backups      |
| Restore plan      | Tested disaster recovery           |
| CI/CD             | GitHub Actions pipeline            |
| Tests             | Integration + E2E test suite       |
| Eval framework    | Automated quality regression       |
| Admin dashboard   | System metrics + user management   |
| Model registry    | Track model versions + performance |

---

## Month 2 Goals

| Feature                   | Description                     |
| ------------------------- | ------------------------------- |
| Prometheus + Grafana      | Full observability stack        |
| OpenTelemetry             | Distributed tracing             |
| Model cost tracking       | Per-query cost attribution      |
| Canary deployment         | Gradual rollout of new models   |
| Feedback review flow      | Admin approval of training data |
| Prompt versioning         | Track and A/B test prompts      |
| Stronger tenant isolation | Data encryption at rest         |
| Governance/compliance     | Data retention policies         |

---

## What NOT to Focus on First

- Advanced fine-tuning pipelines
- Fancy UI animations
- Agent/tool-use features
- Kubernetes (start with Docker Compose)
- Complex eval dashboards
- Over-optimizing RAPTOR internals

---

## Required Credentials / Accounts

### Must Have to Start

| Service           | Credential                                                            |
| ----------------- | --------------------------------------------------------------------- |
| GitHub            | Repository access                                                     |
| PostgreSQL        | `DATABASE_URL`                                                        |
| Redis             | `REDIS_URL`                                                           |
| MinIO or S3       | `S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`           |
| Qdrant            | `QDRANT_URL`, `QDRANT_API_KEY` (if managed)                           |
| LLM provider      | At least one: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GROQ_API_KEY` |
| Auth provider     | `CLERK_SECRET_KEY`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`               |
| Sentry (optional) | `SENTRY_DSN`                                                          |

### Full `.env` Template

```bash
# Database
DATABASE_URL=postgresql://raptor:raptor@localhost:5432/raptor

# Redis
REDIS_URL=redis://localhost:6379/0

# Vector DB
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Object Storage
S3_ENDPOINT=http://localhost:9000
S3_BUCKET=raptor-documents
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# LLM Providers
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=
LLM_API_URL=http://localhost:11435/v1/chat/completions

# Auth
CLERK_SECRET_KEY=
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=

# Monitoring
SENTRY_DSN=

# App
SECRET_KEY=change-me-in-production
DEBUG=true
LOG_LEVEL=INFO
```

---

## Next Artifacts to Build

1. ✅ `PROJECT_ROADMAP.md` — This file
2. ✅ `ARCHITECTURE.md` — System design
3. ✅ `IMPLEMENTATION_PLAN.md` — Exact 2-day checklist
4. `docker-compose.yml` — Infrastructure setup
5. `alembic/` — DB migration framework
6. Refactored folder structure
