"""Run a live end-to-end smoke test against the local RAPTOR stack."""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.generation import generate_with_retrieval
from app.core.retrieval_orchestrator import build_context_text, retrieve
from app.db.models.chat import ChatMessage, ChatSession
from app.db.models.chunk import ChunkMetadata
from app.db.models.collection import Collection
from app.db.models.document import Document, DocumentVersion
from app.db.models.ingestion_job import IngestionJob
from app.db.models.tree_node import TreeNode
from app.db.models.user import User
from app.db.models.workspace import Workspace
from app.db.session import SyncSessionLocal
from app.storage.s3_client import ensure_bucket, upload_file
from app.storage.vector_store import collection_info
from app.workers.tasks.ingest import run_ingestion_pipeline


QUERY = "How does this RAPTOR stack ingest documents and answer questions locally?"
TIMEOUT_SECONDS = int(os.environ.get("SMOKE_TIMEOUT_SECONDS", "900"))
POLL_SECONDS = float(os.environ.get("SMOKE_POLL_SECONDS", "5"))


def _build_smoke_text(run_label: str) -> str:
    return f"""
RAPTOR local smoke run {run_label}

This synthetic document describes the local research assistant stack.
The FastAPI API receives uploads and chat requests.
Documents are stored in MinIO and their metadata is stored in PostgreSQL.
Redis brokers background Celery ingestion jobs.
The ingestion worker validates the file, extracts text, normalizes it, chunks it,
creates embeddings with the configured sentence-transformer model, builds RAPTOR
summary nodes, and indexes vectors in Qdrant.
The retrieval pipeline embeds the user query, searches Qdrant for candidate chunks,
optionally reranks them, traverses RAPTOR tree summaries, and assembles context.
The generation layer sends the assembled context to Ollama using the configured
model so answers can be produced fully locally.
This smoke test verifies upload, storage, ingestion, retrieval, generation,
and chat persistence in one end-to-end flow.
""".strip()


def _create_test_records() -> dict[str, str]:
    run_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    file_name = f"smoke-{run_id}.txt"
    s3_key = f"smoke/{file_name}"
    content = _build_smoke_text(run_id).encode("utf-8")

    ensure_bucket()
    upload_file(s3_key, io.BytesIO(content), content_type="text/plain")

    with SyncSessionLocal() as session:
        user = User(
            clerk_id=f"smoke-{run_id}",
            email=f"smoke-{run_id}@local.test",
            display_name="Pipeline Smoke",
        )
        workspace = Workspace(
            name=f"Smoke Workspace {run_id}",
            owner=user,
            settings={"created_by": "pipeline_smoke"},
        )
        collection = Collection(
            workspace=workspace,
            name=f"Smoke Collection {run_id}",
            description="Synthetic end-to-end validation collection",
            settings={"created_by": "pipeline_smoke"},
        )
        document = Document(
            collection=collection,
            filename=file_name,
            content_type="text/plain",
            file_size_bytes=len(content),
            s3_key=s3_key,
            metadata_={"smoke": True, "run_id": run_id, "created_at": now.isoformat()},
            status="uploaded",
        )
        version = DocumentVersion(document=document, version=1, s3_key=s3_key)
        job = IngestionJob(document=document, status="pending", progress_pct=0)
        session.add_all([user, workspace, collection, document, version, job])
        session.commit()

        return {
            "run_id": run_id,
            "user_id": str(user.id),
            "workspace_id": str(workspace.id),
            "collection_id": str(collection.id),
            "document_id": str(document.id),
            "job_id": str(job.id),
            "s3_key": s3_key,
            "filename": file_name,
        }


def _wait_for_ingestion(job_id: str, document_id: str) -> dict[str, object]:
    deadline = time.time() + TIMEOUT_SECONDS
    timeline: list[dict[str, object]] = []

    while time.time() < deadline:
        with SyncSessionLocal() as session:
            job = session.get(IngestionJob, uuid.UUID(job_id))
            document = session.get(Document, uuid.UUID(document_id))
            if job is None or document is None:
                raise RuntimeError("Smoke test records disappeared during ingestion")

            snapshot = {
                "status": job.status,
                "stage": job.current_stage,
                "progress_pct": job.progress_pct,
                "error_message": job.error_message,
                "document_status": document.status,
            }
            if not timeline or timeline[-1] != snapshot:
                timeline.append(snapshot)

            if job.status == "completed":
                return {
                    "timeline": timeline,
                    "job_status": job.status,
                    "job_stage": job.current_stage,
                    "job_progress_pct": job.progress_pct,
                    "job_chunk_count": job.chunk_count,
                    "document_status": document.status,
                }

            if job.status == "failed":
                raise RuntimeError(job.error_message or "Ingestion job failed")

        time.sleep(POLL_SECONDS)

    raise TimeoutError(f"Ingestion did not complete within {TIMEOUT_SECONDS} seconds")


def _collect_storage_stats(collection_id: str, document_id: str) -> dict[str, object]:
    with SyncSessionLocal() as session:
        chunk_count = (
            session.query(ChunkMetadata)
            .filter(ChunkMetadata.document_id == uuid.UUID(document_id))
            .count()
        )
        tree_node_count = (
            session.query(TreeNode)
            .filter(TreeNode.document_id == uuid.UUID(document_id))
            .count()
        )

    return {
        "chunk_metadata_count": chunk_count,
        "tree_node_count": tree_node_count,
        "qdrant": collection_info(uuid.UUID(collection_id)),
    }


def _run_retrieval_and_generation(user_id: str, collection_id: str) -> dict[str, object]:
    collection_uuid = uuid.UUID(collection_id)
    user_uuid = uuid.UUID(user_id)

    with SyncSessionLocal() as session:
        retrieval = retrieve(
            query=QUERY,
            collection_id=collection_uuid,
            top_k=5,
            session=session,
        )
        context = build_context_text(retrieval)
        generation = generate_with_retrieval(
            question=QUERY,
            collection_id=collection_uuid,
            top_k=5,
            session=session,
        )

        chat_session = ChatSession(
            user_id=user_uuid,
            collection_id=collection_uuid,
            title="Smoke test session",
        )
        user_message = ChatMessage(
            session=chat_session,
            role="user",
            content=QUERY,
        )
        assistant_message = ChatMessage(
            session=chat_session,
            role="assistant",
            content=generation["content"],
            citations=generation["citations"],
            model_used=generation["model_used"],
            latency_ms=generation["latency_ms"],
            token_count=generation["token_count"],
        )
        session.add_all([chat_session, user_message, assistant_message])
        session.commit()

        return {
            "retrieved_chunks": len(retrieval["chunks"]),
            "tree_context_items": len(retrieval["tree_context"]),
            "citations": len(generation["citations"]),
            "context_chars": len(context),
            "model_used": generation["model_used"],
            "latency_ms": generation["latency_ms"],
            "token_count": generation["token_count"],
            "answer_preview": generation["content"][:600],
            "chat_session_id": str(chat_session.id),
            "assistant_message_id": str(assistant_message.id),
        }


def main() -> int:
    seed = _create_test_records()

    # Use the Celery task body directly inside the API container so the live stack
    # is exercised even if the broker queue is slow or unobserved from the host.
    run_ingestion_pipeline.run(
        job_id=seed["job_id"],
        document_id=seed["document_id"],
        collection_id=seed["collection_id"],
    )

    ingestion = _wait_for_ingestion(seed["job_id"], seed["document_id"])
    storage = _collect_storage_stats(seed["collection_id"], seed["document_id"])
    rag = _run_retrieval_and_generation(seed["user_id"], seed["collection_id"])

    summary = {
        "seed": seed,
        "ingestion": ingestion,
        "storage": storage,
        "rag": rag,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        raise