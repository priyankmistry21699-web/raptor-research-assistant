# RAPTOR RAG Platform

Production-oriented RAPTOR retrieval-augmented generation platform for document ingestion, hierarchical retrieval, chat, feedback, evaluation, and model iteration.

The repository has moved beyond the original research-demo shape. The active platform now centers on FastAPI v2 APIs, Celery workers, PostgreSQL, Redis, Qdrant, MinIO, LiteLLM-based generation, and Clerk-backed authentication.

## Current Status

- Backend platform is running on FastAPI with v2 routes under `/api/v2`
- Async ingestion and evaluation run through Celery workers
- Primary local infrastructure runs through Docker Compose
- Local generation is configured for Ollama by default
- Cloud API keys can be configured for Anthropic, Groq, and OpenAI
- Clerk auth is wired for protected routes, with development bypass available when explicitly in development and Clerk secrets are unset
- Production-readiness score is tracked in `ROADMAP_TO_100.md`

## What This Platform Does

- Upload and track documents per workspace and collection
- Parse documents into chunks and build RAPTOR trees
- Store vectors in Qdrant and metadata in PostgreSQL
- Retrieve context with vector search plus hierarchical RAPTOR traversal
- Generate grounded answers with citations
- Persist chat sessions and messages
- Collect feedback and run asynchronous evaluation / training workflows
- Expose admin, health, metrics, and operational endpoints

## Runtime Architecture

### Core Services

| Layer | Service | Role |
| --- | --- | --- |
| API | FastAPI | Public HTTP API, auth, orchestration, generation, retrieval |
| Worker | Celery | Ingestion, evaluation, background processing |
| Database | PostgreSQL | Users, workspaces, documents, chat, feedback, audit, training metadata |
| Cache / Queue | Redis | Celery broker, caching, rate limiting |
| Vector Store | Qdrant | Chunk and summary embeddings |
| Object Storage | MinIO | S3-compatible local storage for uploaded files and artifacts |
| LLM | Ollama / cloud providers | Local inference plus optional cloud fallback |
| Auth | Clerk | User identity, JWT verification, webhook-based user sync |

### Local Service Map

These are the currently configured local ports from `docker-compose.yml`.

| Service | URL / Port | Notes |
| --- | --- | --- |
| API | `http://localhost:8000` | OpenAPI docs at `/docs` |
| PostgreSQL | `localhost:5432` | Database `raptor` |
| Redis | `localhost:6379` | Celery broker / cache |
| Qdrant | `http://localhost:6335` | Host port remapped from 6333 |
| MinIO API | `http://localhost:9000` | S3-compatible endpoint |
| MinIO Console | `http://localhost:9002` | Host port remapped from 9001 |
| Ollama | `http://localhost:11435` | Expected by the API container via `host.docker.internal` |

## API Surface

The active v2 route groups are:

- `health`
- `auth`
- `workspaces`
- `collections`
- `documents`
- `chat`
- `retrieve`
- `generate`
- `feedback`
- `eval`
- `training`
- `admin`

v1 routes remain mounted for backward compatibility but are deprecated. The application entrypoint is defined in `app/main.py`.

## Quick Start

### 1. Prerequisites

- Docker Desktop
- Python 3.11+
- Ollama running locally if you want the default local model path
- Optional provider keys for Anthropic, Groq, or OpenAI

### 2. Create `.env`

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

Minimum fields for local development are documented in `.env.example`. The most important ones are:

- `SECRET_KEY`
- `DATABASE_URL`
- `REDIS_URL`
- `QDRANT_URL`
- `S3_ENDPOINT`
- `LLM_PROVIDER`
- `OLLAMA_BASE_URL`

Optional cloud keys:

- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `OPENAI_API_KEY`
- `CLERK_SECRET_KEY`
- `CLERK_PUBLISHABLE_KEY`
- `CLERK_WEBHOOK_SECRET`

### 3. Start the stack

```powershell
docker compose up -d --build
```

### 4. Verify services

```powershell
docker compose ps
docker compose logs -f api
```

### 5. Open the platform

- API docs: `http://localhost:8000/docs`
- Live health: `http://localhost:8000/api/v2/health/live`
- MinIO console: `http://localhost:9002`
- Qdrant dashboard: `http://localhost:6335/dashboard`

## Authentication Notes

Clerk is the intended auth provider for protected v2 endpoints.

Required Clerk settings:

- `CLERK_SECRET_KEY`
- `CLERK_PUBLISHABLE_KEY`
- `CLERK_WEBHOOK_SECRET`

For local development only, auth bypass is available when:

- `ENVIRONMENT=development`
- Clerk secret key is unset

That bypass is intentionally limited and should not be used as a production configuration.

## LLM Routing Notes

The main generation path lives in `app/core/generation.py`.

Current behavior:

- Primary provider is controlled by `LLM_PROVIDER` and `LLM_MODEL`
- Local Docker defaults use Ollama with `mistral:latest`
- Anthropic credentials are supported by configuration
- Automatic fallback in the current generation layer is not yet fully symmetric across all providers

If you want production-grade multi-provider failover, track the remaining work in `ROADMAP_TO_100.md`.

## Project Layout

```text
app/
  api/
    v2/routes/         FastAPI v2 route groups
  core/                Config, security, middleware, generation, retrieval
  db/                  Session management and SQLAlchemy models
  storage/             Object storage integrations
  workers/             Celery app and background tasks
alembic/               Database migration environment
scripts/               Operational and data pipeline scripts
tests/                 API, retrieval, ingestion, evaluation, and worker tests
```

## Documentation Index

- `ARCHITECTURE.md`: system design, data flows, deployment profiles, API and storage responsibilities
- `ROADMAP_TO_100.md`: production-readiness scoring and remaining work
- `PROJECT_ROADMAP.md`: broader implementation roadmap
- `IMPLEMENTATION_PLAN.md`: phase-by-phase build plan

## Known Gaps

The platform is materially stronger than the original demo, but it is not yet finished. The biggest open areas are:

- modern frontend replacement for the legacy UI path
- backup and disaster recovery runbooks
- citation enrichment and response streaming
- cleanup of remaining legacy ChromaDB-era code paths
- stronger cloud-provider fallback and secret-hardening

Those gaps are tracked in `ROADMAP_TO_100.md`.

## Development Commands

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f api
docker compose logs -f worker
pytest
```

## Versioning Note

The application currently identifies itself as `2.0.0-alpha` in `app/main.py`. The repository contains both legacy/demo-era code and the newer production-oriented v2 platform, so documentation should be interpreted in favor of the v2 stack unless explicitly marked otherwise.
