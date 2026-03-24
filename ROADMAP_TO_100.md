# ROADMAP: 48/100 → 100/100

## RAPTOR Research Assistant — Complete Production Readiness Plan

**Initial Score: 48/100 | Current Score: 78/100 | Target: 100/100 | Gap: 22 points**
**Last Audit: Phase 1+2 Complete (commit b5e0caf)**

---

## Scoring Reference (20 Categories × 5 points each)

| #   | Category            | Initial | Phase1+2 | Target | Gap | Phase |
| --- | ------------------- | ------- | -------- | ------ | --- | ----- |
| 1   | Frontend            | 2       | 2        | 5      | +3  | 5     |
| 2   | Authentication      | 2       | 5        | 5      | 0   | ✅    |
| 3   | Authorization/RBAC  | 1       | 4        | 5      | +1  | 5     |
| 4   | API Design          | 3       | 5        | 5      | 0   | ✅    |
| 5   | Data Models         | 4       | 5        | 5      | 0   | ✅    |
| 6   | Database/Migrations | 2       | 5        | 5      | 0   | ✅    |
| 7   | Object Storage      | 4       | 4        | 5      | +1  | 3     |
| 8   | Vector Database     | 4       | 4        | 5      | +1  | 3     |
| 9   | Async Job Pipeline  | 4       | 5        | 5      | 0   | ✅    |
| 10  | Ingestion Pipeline  | 3       | 5        | 5      | 0   | ✅    |
| 11  | Retrieval/Reranker  | 3       | 4        | 5      | +1  | 3     |
| 12  | RAPTOR Hierarchy    | 1       | 5        | 5      | 0   | ✅    |
| 13  | LLM/Generation      | 3       | 4        | 5      | +1  | 3     |
| 14  | Citations           | 3       | 3        | 5      | +2  | 3     |
| 15  | Feedback/Evaluation | 3       | 4        | 5      | +1  | 3     |
| 16  | Security            | 1       | 4        | 5      | +1  | 3     |
| 17  | Observability       | 1       | 4        | 5      | +1  | 4     |
| 18  | CI/CD               | 0       | 4        | 5      | +1  | 5     |
| 19  | Config/Secrets      | 4       | 4        | 5      | +1  | 3     |
| 20  | Backup/DR           | 0       | 0        | 5      | +5  | 4     |

### Phase 1+2 Audit Evidence

#### ✅ Authentication (2→5)

- Auth bypass fixed: dev mode restricted to `ENVIRONMENT=development` + viewer role only
- Clerk JWT verification for all protected routes
- `/api/v1/*` removed from PUBLIC_PREFIXES (requires auth)
- Startup validation blocks production without CLERK_SECRET_KEY
- `get_current_user` dependency on ALL v2 routes

#### ✅ Authorization/RBAC (1→4)

- Role hierarchy: admin(3) > editor(2) > viewer(1)
- `require_role()` and `require_roles()` dependency factories
- Admin routes protected with `require_role("admin")`
- Training creation requires `require_role("editor")`
- All read routes require authenticated user
- Missing: frontend RBAC rendering, workspace-level RBAC

#### ✅ API Design (3→5)

- Auth on ALL v2 routes (workspaces, collections, documents, feedback, retrieve, training, chat, generate, eval, admin)
- Pagination via `PaginatedResponse(items, total, page, page_size)` on all list endpoints
- v1 routes marked `deprecated=True` in OpenAPI
- Consistent response schemas with Pydantic models
- Proper OpenAPI tags on all routers

#### ✅ Data Models (4→5)

- CHECK constraints: chat_message.role, document.status, ingestion_job.status, eval_run.status, training_run.status, training_run.run_type, feedback.rating, tree_node.node_type, tree_node.level, user.role, workspace_member.role
- Composite indexes: tree_nodes(collection_id, node_type, level), chunks_metadata(document_id, chunk_index), chunks_metadata(collection_id, document_id), chat_messages(session_id, created_at)
- ON DELETE CASCADE / SET NULL on all foreign keys

#### ✅ Database/Migrations (2→5)

- Alembic initial migration covers all 16+ tables
- All CHECK constraints and composite indexes in migration
- Alembic env.py reads DB URL from app settings
- Migration script has proper upgrade() and downgrade()

#### ✅ Async Job Pipeline (4→5)

- Celery tasks: autoretry_for, retry_backoff, retry_backoff_max
- acks_late=True for at-least-once delivery
- MaxRetriesExceededError handling with status update to "failed"
- Async eval dispatch via run_evaluation.delay()

#### ✅ Ingestion Pipeline (3→5)

- RAPTOR tree builder integrated into ingest pipeline
- `_build_raptor_tree()` calls `build_raptor_tree()` (real implementation)
- Summary node embeddings indexed in vector store
- Progress logging and error handling

#### ✅ RAPTOR Hierarchy (1→5)

- Full algorithm: KMeans clustering → LLM summarization → embedding → recursive until root
- Configurable max_depth, max_topics from settings
- Nodes stored with parent_id chains in tree_nodes table
- Summary embeddings indexed in Qdrant alongside leaf chunks
- Fallback to extractive summarization if LLM fails

#### Security (1→4)

- CORS lockdown: specific methods/headers instead of wildcards
- Rate limiting: Redis-backed per-user sliding window (120/min default, 20/min generate)
- Security headers: CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy
- Input sanitization: prompt injection detection, control char stripping, HTML escaping
- Audit logging wired to admin, generate, eval routes
- Missing: ChromaDB/pickle removal (legacy code)

#### Observability (1→4)

- Sentry SDK with FastAPI + SQLAlchemy integrations
- Prometheus metrics at /metrics
- OpenTelemetry with OTLP exporter + FastAPIInstrumentor
- Structured logging with request ID
- Missing: custom business metric alerts, advanced alerting

#### CI/CD (0→4)

- GitHub Actions CI: lint(ruff), security(bandit), test(pytest), docker build
- Deploy pipeline: staging → production with approval gate
- Dependabot for weekly dependency updates
- Missing: E2E tests, Playwright integration

#### LLM/Generation (3→4)

- Chat.py v2 consolidated to use generation.py (LiteLLM)
- Fallback chain: primary → Groq → OpenAI
- Chat history included in generation
- Retrieve route uses retrieval_orchestrator
- Missing: streaming support, token counting

#### Retrieval (3→4)

- Tree traversal enhanced with fallback to summary nodes
- Retrieve route uses orchestrator instead of direct Qdrant
- Reranker exists
- Missing: caching for tree traversal, weighted tree results

#### Feedback/Evaluation (3→4)

- Async RAGAS evaluation via Celery task
- Auth on feedback submission
- Feedback route with pagination
- Missing: automated feedback → preference pair pipeline, A/B testing

---

## PHASE 1 — CRITICAL FIXES & SECURITY (48 → 70)

> **Goal:** Fix every security vulnerability, add observability, CI/CD, and migrations.
> **Expected Score After Phase 1: ~70/100**
> **Categories Affected:** Auth (2→5), RBAC (1→4), DB (2→5), Security (1→4), Observability (1→4), CI/CD (0→4)

### Task 1.1 — Fix Authentication Bypass (Auth 2→4)

**Files:** `app/core/security.py`

- [ ] Remove the dev-mode stub that returns a fake admin user when `CLERK_SECRET_KEY` is unset
- [ ] Replace with explicit `DEV_AUTH_ENABLED=true` env var gated check that ONLY works when `ENVIRONMENT=development`
- [ ] Dev mode returns `role="viewer"` (not admin) to prevent accidental privilege escalation
- [ ] Add `Depends(get_current_user)` to ALL v2 routes that currently lack it
- [ ] Ensure `/api/v1/*` routes also require auth (remove from `PUBLIC_PREFIXES`) or deprecation-gate them

### Task 1.2 — Protect Admin Routes (Auth 4→5, RBAC 1→3)

**Files:** `app/api/v2/routes/admin.py`, `app/core/security.py`

- [ ] Create `require_role(role: str)` dependency in `security.py`
- [ ] Add `Depends(require_role("admin"))` to ALL admin endpoints (`/stats`, `/models`, `/audit`)
- [ ] Create `require_roles(*roles)` for flexible multi-role checks
- [ ] Add role check tests

### Task 1.3 — RBAC Middleware (RBAC 3→4)

**Files:** NEW `app/core/rbac.py`, `app/core/security.py`

- [ ] Define permission model: `admin`, `editor`, `viewer` roles
- [ ] Map roles → allowed actions (admin: all, editor: read+write, viewer: read-only)
- [ ] Create `RBACMiddleware` or dependency that checks `user.role` against route requirements
- [ ] Apply RBAC to: admin routes (admin only), write routes (editor+), read routes (all authenticated)
- [ ] Add RBAC tests with each role

### Task 1.4 — CORS Lockdown (Security 1→2)

**Files:** `app/main.py`, `app/core/config.py`

- [ ] Replace `allow_origins=["*"]` with `settings.cors_origins` (list from config)
- [ ] Set default to `["http://localhost:3000", "http://localhost:7860"]` for dev
- [ ] Remove `allow_methods=["*"]`, specify `["GET", "POST", "PUT", "DELETE", "OPTIONS"]`
- [ ] Remove `allow_headers=["*"]`, specify `["Authorization", "Content-Type"]`

### Task 1.5 — Rate Limiting (Security 2→3)

**Files:** NEW `app/core/rate_limit.py`, `app/main.py`

- [ ] Create Redis-backed rate limiter: per-user sliding window (e.g., 100 req/min for API, 10 req/min for generation)
- [ ] Create `RateLimitMiddleware` using Redis INCR + TTL
- [ ] Apply to all v2 routes via middleware
- [ ] Add configurable limits in `config.py`: `rate_limit_default`, `rate_limit_generation`
- [ ] Return proper `429 Too Many Requests` with `Retry-After` header

### Task 1.6 — Security Headers (Security 3→4)

**Files:** `app/main.py` or NEW `app/core/headers.py`

- [ ] Add security headers middleware:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 0` (modern approach: rely on CSP)
  - `Strict-Transport-Security: max-age=31536000; includeSubDomains` (when HTTPS)
  - `Content-Security-Policy: default-src 'self'`
  - `Referrer-Policy: strict-origin-when-cross-origin`

### Task 1.7 — Wire Audit Logging (Security 4→4.5)

**Files:** All v2 route files, `app/core/audit.py`

- [ ] Add `log_audit_from_request()` calls to ALL mutation endpoints:
  - `POST /documents` → action: "document.upload"
  - `POST /collections` → action: "collection.create"
  - `POST /generate` → action: "generation.create"
  - `POST /admin/models` → action: "model.register"
  - `POST /feedback` → action: "feedback.submit"
  - All DELETE/PUT endpoints
- [ ] Ensure audit log includes user_id, IP, resource_id, action, timestamp

### Task 1.8 — Initialize Sentry (Observability 1→2)

**Files:** `app/main.py`

- [ ] Add `sentry_sdk.init()` in `create_app()` startup:
  ```python
  if settings.sentry_dsn:
      import sentry_sdk
      sentry_sdk.init(
          dsn=settings.sentry_dsn,
          environment=settings.environment,
          traces_sample_rate=0.1,
          profiles_sample_rate=0.1,
      )
  ```
- [ ] Add Sentry FastAPI integration

### Task 1.9 — Initialize Prometheus (Observability 2→3)

**Files:** `app/main.py`

- [ ] Add Prometheus instrumentator:
  ```python
  from prometheus_fastapi_instrumentator import Instrumentator
  Instrumentator().instrument(app).expose(app, endpoint="/metrics")
  ```
- [ ] Add custom metrics: `generation_latency_seconds`, `ingestion_jobs_total`, `vector_search_duration`

### Task 1.10 — OpenTelemetry Tracing (Observability 3→4)

**Files:** NEW `app/core/telemetry.py`, `app/main.py`, `requirements.txt`

- [ ] Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-exporter-otlp` to requirements
- [ ] Initialize OTLP tracer in startup
- [ ] Instrument FastAPI, SQLAlchemy, Redis, HTTP clients
- [ ] Add custom spans for: retrieval pipeline, generation, tree building

### Task 1.11 — Generate Alembic Migrations (DB 2→5)

**Files:** `alembic/versions/`, `alembic/env.py`

- [ ] Run `alembic revision --autogenerate -m "initial_schema"` to capture all 16 tables
- [ ] Review generated migration for correctness
- [ ] Run `alembic upgrade head` to verify
- [ ] Add `alembic upgrade head` to Docker entrypoint / startup script
- [ ] Document migration workflow in README

### Task 1.12 — CI/CD Pipeline (CI/CD 0→4)

**Files:** NEW `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`

- [ ] **CI workflow** (on push + PR):
  - Lint: `ruff check .` + `ruff format --check .`
  - Type check: `mypy app/ --ignore-missing-imports`
  - Unit tests: `pytest tests/ -x --tb=short`
  - Security scan: `bandit -r app/` + `safety check`
  - Build Docker image (but don't push on PR)
- [ ] **Deploy workflow** (on push to main):
  - Build + push Docker image to GCR/Artifact Registry
  - Run migrations
  - Deploy to Cloud Run (staging → prod with approval gate)
- [ ] Add branch protection rules documentation
- [ ] Create `.github/dependabot.yml` for dependency updates

### Task 1.13 — Input Sanitization (Security 4.5→5)

**Files:** NEW `app/core/sanitize.py`, generation routes

- [ ] Sanitize user prompts before sending to LLM (strip injection patterns)
- [ ] Validate and sanitize all user string inputs (collection names, document titles)
- [ ] Add max-length checks on all text fields
- [ ] HTML-escape any user content returned in API responses

---

## PHASE 2 — CORE ARCHITECTURE ALIGNMENT (70 → 83)

> **Goal:** Implement RAPTOR tree building, improve ingestion/retrieval, clean up API design.
> **Expected Score After Phase 2: ~83/100**
> **Categories Affected:** RAPTOR (1→5), Ingestion (3→5), Retrieval (3→4), API (3→5), Models (4→5), Async (4→5)

### Task 2.1 — RAPTOR Tree Building Algorithm (RAPTOR 1→5, Ingestion 3→4)

**Files:** `app/workers/tasks/ingest.py`, NEW `app/core/raptor_tree.py`

- [ ] Create `app/core/raptor_tree.py` — the core algorithm:
  ```
  build_raptor_tree(chunks, embeddings) -> list[TreeNode]:
    1. Cluster leaf chunks using KMeans/GMM (scikit-learn)
    2. For each cluster, generate a summary using LLM
    3. Embed the cluster summaries
    4. Recursively cluster+summarize until single root
    5. Return list of TreeNode objects with parent_id links
  ```
- [ ] Configurable parameters from `settings.raptor_*`:
  - `raptor_max_topics` — max clusters per level
  - `raptor_chunk_size` — summary chunk size
  - `raptor_summary_model` — LLM model for summarization
- [ ] Wire into `_build_raptor_tree()` in `ingest.py` (replace stub)
- [ ] Persist all TreeNode levels (leaf, section, topic, root) with proper `parent_id` chains
- [ ] Index summary embeddings in Qdrant alongside leaf chunks
- [ ] Add progress logging for long-running tree builds
- [ ] Unit tests with mock LLM + small document

### Task 2.2 — Retrieval with Real Tree Traversal (Retrieval 3→4)

**Files:** `app/core/retrieval_orchestrator.py`

- [ ] Update `_traverse_tree_nodes()` to actually query TreeNode table for parent chain
- [ ] Include section/topic summaries in context when leaf matches
- [ ] Weight tree-level results properly (leaf > section > topic in relevance)
- [ ] Add caching for tree traversal results (Redis, TTL-based)
- [ ] Test with real tree data end-to-end

### Task 2.3 — Consolidate Dual LLM Clients (LLM 3→4)

**Files:** `app/core/generation.py`, `app/core/llm_client.py`

- [ ] Audit all callers of legacy `llm_client.py`
- [ ] Migrate all usage to `generation.py` (LiteLLM-based)
- [ ] Keep `llm_client.py` only for fine-tuned model inference (if needed) or deprecate entirely
- [ ] Ensure tree building summarization uses generation.py

### Task 2.4 — Async Eval Dispatch (Async 4→5, Feedback 3→4)

**Files:** `app/api/v2/routes/eval.py`, NEW `app/workers/tasks/evaluate.py`

- [ ] Create Celery task `run_evaluation` in `evaluate.py`
- [ ] Dispatches via `.delay()` from eval route
- [ ] Runs RAGAS evaluation asynchronously
- [ ] Updates `EvalRun` status (`pending → running → completed/failed`)
- [ ] Stores results in `EvalResult` table
- [ ] Sends webhook/notification on completion (optional)

### Task 2.5 — API Design Polish (API 3→5)

**Files:** All route files, `app/main.py`

- [ ] Add consistent response schemas for ALL endpoints (Pydantic response_model)
- [ ] Add proper OpenAPI tags + descriptions to all routers
- [ ] Add pagination support (`skip`, `limit`, `total`) to all list endpoints
- [ ] Add API versioning headers (`X-API-Version: v2`)
- [ ] Deprecation headers for v1 routes (`Deprecation: true`, `Sunset: <date>`)
- [ ] Add `OPTIONS` preflight handling
- [ ] Comprehensive error response schema (error code, message, details)

### Task 2.6 — Data Model Improvements (Models 4→5)

**Files:** Various model files

- [ ] Add `CHECK` constraints where appropriate (e.g., `score BETWEEN 0 AND 1`)
- [ ] Add composite indexes for common query patterns (e.g., `(collection_id, created_at)`)
- [ ] Add `ON DELETE CASCADE` / `SET NULL` for all foreign keys explicitly
- [ ] Add model-level validation (Pydantic validators on schemas)
- [ ] Generate new Alembic migration for any schema changes

### Task 2.7 — Celery Error Handling & Retries (Async 4.5→5)

**Files:** `app/workers/tasks/ingest.py`, `app/workers/celery_app.py`

- [ ] Add `autoretry_for=(Exception,)` with exponential backoff
- [ ] Add `max_retries=3` on ingest task
- [ ] Add dead letter queue for permanently failed tasks
- [ ] Add task status callbacks (update DocumentUpload.status on each stage)
- [ ] Add Celery Flower for task monitoring (add to docker-compose)

---

## PHASE 3 — PRODUCTION HARDENING (83 → 93)

> **Goal:** Polish storage, vector DB, LLM, citations, feedback, config.
> **Expected Score After Phase 3: ~93/100**
> **Categories Affected:** Storage (4→5), Vector (4→5), LLM (4→5), Citations (3→5), Feedback (4→5), Config (4→5)

### Task 3.1 — Legacy ChromaDB Cleanup (Vector 4→5)

**Files:** Multiple scripts, `app/frontend/ui.py`, `app/core/raptor_index.py`

- [ ] Remove all 23 ChromaDB references from application code
- [ ] Remove `chromadb` from `requirements.txt`
- [ ] Migrate any remaining ChromaDB data to Qdrant (one-time script)
- [ ] Remove `chroma_db/` directory
- [ ] Update `app/frontend/ui.py` to use Qdrant-backed retrieval
- [ ] Remove `raptor_index.py` (legacy pickle-based)

### Task 3.2 — Legacy Pickle Removal (Security improvement)

**Files:** Scripts, `app/core/raptor_index.py`, `app/frontend/ui.py`

- [ ] Replace `pickle.load()` with safe alternatives (JSON, MessagePack, or DB queries)
- [ ] Remove all `pickle.load()` from application code (keep only in offline scripts if unavoidable)
- [ ] Add safety note in README for any remaining pickle usage

### Task 3.3 — Object Storage Migration (Storage 4→5)

**Files:** `app/storage/object_store.py`, ingestion pipeline

- [ ] Migrate `data/raw/`, `data/processed/` files to S3/GCS buckets
- [ ] Store embeddings in Qdrant (not filesystem)
- [ ] Store feedback exports in object storage (not `data/feedback/`)
- [ ] Create migration script for existing local data
- [ ] Remove local filesystem dependencies from v2 code paths

### Task 3.4 — Enhanced Citation System (Citations 3→5)

**Files:** `app/core/retrieval_orchestrator.py`, `app/api/v2/routes/generate.py`

- [ ] Add page numbers to citations (from PDF extraction metadata)
- [ ] Add section headings to citations
- [ ] Return structured citation objects in generation response
- [ ] Add citation verification (check cited chunk actually supports the claim)
- [ ] API response includes `citations[]` array with `{doc_title, page, section, score, snippet}`

### Task 3.5 — Feedback Loop Completion (Feedback 4→5)

**Files:** `app/core/feedback.py`, `app/core/learning_loop.py`

- [ ] Ensure feedback → preference pair conversion is automated
- [ ] Add feedback aggregation dashboard endpoint (`GET /feedback/stats`)
- [ ] Schedule periodic DPO training trigger based on feedback volume
- [ ] Add A/B testing support (serve from base vs fine-tuned model, track performance)
- [ ] Wire RAGAS evaluation to run automatically after fine-tuning

### Task 3.6 — LLM Routing Polish (LLM 4→5)

**Files:** `app/core/generation.py`

- [ ] Add streaming support (SSE/WebSocket for real-time token generation)
- [ ] Add token counting before sending (prevent exceeding context window)
- [ ] Add prompt template management system
- [ ] Add model-specific prompt formatting (chat vs completion)
- [ ] Add cost tracking per request (estimate tokens × price)

### Task 3.7 — Config & Secrets Hardening (Config 4→5)

**Files:** `app/core/config.py`, `.env.example`

- [ ] Add config validation on startup (fail fast if required secrets missing in production)
- [ ] Add secret rotation support (reload secrets without restart)
- [ ] Add per-environment config profiles (dev, staging, prod)
- [ ] Document all env vars in `.env.example` with descriptions
- [ ] Add `ENVIRONMENT` check that blocks dangerous settings in production (e.g., `DEV_AUTH_ENABLED`)

---

## PHASE 4 — DISASTER RECOVERY & OPERATIONAL EXCELLENCE (93 → 97)

> **Goal:** Backup, DR, advanced monitoring, and operational procedures.
> **Expected Score After Phase 4: ~97/100**
> **Categories Affected:** Backup/DR (0→5), Observability (4→5), Async (5→5)

### Task 4.1 — Database Backup System (Backup 0→3)

**Files:** NEW `scripts/backup_db.sh`, NEW `scripts/restore_db.sh`, `docker-compose.yml`

- [ ] Create `pg_dump` backup script with:
  - Daily full backups to GCS/S3
  - Hourly WAL archiving for PITR
  - Retention policy (30 days full, 7 days WAL)
- [ ] Create restore script with verification
- [ ] Add backup service to docker-compose (cron-based)
- [ ] Test backup/restore cycle end-to-end

### Task 4.2 — Object Storage Backup (Backup 3→4)

**Files:** NEW `scripts/backup_storage.sh`

- [ ] Enable S3/GCS versioning on all buckets
- [ ] Cross-region replication for critical buckets (documents, models)
- [ ] Lifecycle policies (move old versions to cold storage after 90 days)

### Task 4.3 — Vector DB Backup (Backup 4→4.5)

**Files:** NEW `scripts/backup_qdrant.sh`

- [ ] Qdrant snapshot automation (periodic snapshots to object storage)
- [ ] Restore procedure documented and tested

### Task 4.4 — Disaster Recovery Runbook (Backup 4.5→5)

**Files:** NEW `docs/disaster-recovery.md`

- [ ] Document RTO (Recovery Time Objective) and RPO (Recovery Point Objective)
- [ ] Step-by-step recovery procedures for each service
- [ ] Runbook for: database corruption, vector DB failure, object storage loss, full region outage
- [ ] Quarterly DR drill checklist

### Task 4.5 — Advanced Alerting (Observability 4→5)

**Files:** NEW `infrastructure/alerting/`, `app/core/telemetry.py`

- [ ] Error rate alerts (Sentry → PagerDuty/Slack)
- [ ] Latency alerts (p99 > threshold)
- [ ] Queue depth alerts (Celery backlog > threshold)
- [ ] Disk/memory usage alerts
- [ ] Custom business metric alerts (e.g., generation failure rate > 5%)

### Task 4.6 — Celery Monitoring Dashboard (Async 5→5)

**Files:** `docker-compose.yml`

- [ ] Add Flower to docker-compose for Celery monitoring
- [ ] Configure Flower authentication
- [ ] Add task success/failure metrics to Prometheus

---

## PHASE 5 — FRONTEND & FINAL POLISH (97 → 100)

> **Goal:** Modern frontend, final integrations, documentation.
> **Expected Score After Phase 5: 100/100**
> **Categories Affected:** Frontend (2→5), RBAC (4→5), Security (4.5→5)

### Task 5.1 — Next.js Frontend Foundation (Frontend 2→4)

**Files:** NEW `frontend/` directory (separate project or monorepo)

- [ ] Create Next.js 14+ app with App Router
- [ ] Clerk authentication integration (sign-in, sign-up, user management)
- [ ] Layout: sidebar navigation, main content area, chat interface
- [ ] Pages: Dashboard, Collections, Documents, Chat, Search, Admin
- [ ] Tailwind CSS + shadcn/ui component library

### Task 5.2 — Frontend Feature Pages (Frontend 4→5)

- [ ] **Chat page:** Streaming responses, citation display, conversation history
- [ ] **Collections page:** CRUD collections, view documents, upload PDFs
- [ ] **Search page:** Semantic search with filters, reranking toggle
- [ ] **Admin page:** Platform stats, model registry, audit log viewer
- [ ] **Feedback page:** Thumbs up/down on responses, preference comparisons
- [ ] **Dashboard:** Usage stats, recent activity, system health

### Task 5.3 — RBAC in Frontend (RBAC 4→5)

- [ ] Role-based UI rendering (admin sees admin panel; viewers can't edit)
- [ ] API error handling for 403 Forbidden responses
- [ ] Role management UI for admins

### Task 5.4 — End-to-End Testing (CI/CD 4→5)

**Files:** `.github/workflows/e2e.yml`, NEW `tests/e2e/`

- [ ] Playwright E2E tests for critical paths:
  - User sign-up → create collection → upload document → chat
  - Admin: view stats, manage models
  - Search with citations
- [ ] Add E2E tests to CI pipeline
- [ ] Test against staging environment before production deploy

### Task 5.5 — Final Documentation (All categories)

**Files:** `README.md`, NEW `docs/`

- [ ] Complete API documentation (all endpoints with examples)
- [ ] Deployment guide (GCP Cloud Run step-by-step)
- [ ] Architecture decision records (ADRs) for key decisions
- [ ] Contributing guide
- [ ] Security policy (SECURITY.md)
- [ ] Changelog (CHANGELOG.md)

---

## PHASE SUMMARY

| Phase       | Focus                            | Score Change  | Tasks    | Effort    |
| ----------- | -------------------------------- | ------------- | -------- | --------- |
| **Phase 1** | Security + Observability + CI/CD | 48 → 70 (+22) | 13 tasks | ~2-3 days |
| **Phase 2** | RAPTOR Core + API Polish         | 70 → 83 (+13) | 7 tasks  | ~3-4 days |
| **Phase 3** | Production Hardening             | 83 → 93 (+10) | 7 tasks  | ~2-3 days |
| **Phase 4** | Backup/DR + Operations           | 93 → 97 (+4)  | 6 tasks  | ~1-2 days |
| **Phase 5** | Frontend + Final Polish          | 97 → 100 (+3) | 5 tasks  | ~3-4 days |

---

## EXECUTION ORDER WITHIN EACH PHASE

**Phase 1 (do in this order):**
1.1 → 1.2 → 1.3 (auth chain)
1.4 → 1.5 → 1.6 → 1.13 (security hardening chain)
1.7 (audit wiring — independent)
1.8 → 1.9 → 1.10 (observability chain)
1.11 (migrations — independent)
1.12 (CI/CD — independent, but do after tests pass)

**Phase 2 (do in this order):**
2.1 → 2.2 (RAPTOR → retrieval, tightly coupled)
2.3 (LLM consolidation — independent)
2.4 (eval async — independent)
2.5 → 2.6 → 2.7 (API + models + async polish)

**Phase 3-5:** Tasks within each phase are mostly independent.

---

## HOW TO START

```
Ready? Say "Start Phase 1" and we'll begin with Task 1.1 (Fix Authentication Bypass).
Each task will be implemented, tested, and committed before moving to the next.
```
