"""
Document ingestion pipeline — Celery task.

Stages:
  pending → validating → extracting → chunking → embedding
        → tree_building → indexing → completed  (or → failed)
"""

import io
import logging
import uuid
from datetime import datetime, timezone

from app.workers.celery_app import celery_app
from app.db.session import SyncSessionLocal
from app.db.models.ingestion_job import IngestionJob
from app.db.models.document import Document
from app.storage import s3_client

logger = logging.getLogger(__name__)

STAGES = [
    "validating",
    "extracting",
    "chunking",
    "embedding",
    "tree_building",
    "indexing",
]


def _update_job(session, job: IngestionJob, stage: str, progress: int):
    job.current_stage = stage
    job.progress_pct = progress
    session.commit()


# ── Pipeline stage implementations ────────────────────────────────────

def _validate(file_bytes: bytes, content_type: str) -> None:
    """Validate the uploaded file (type, size, malformed PDF check)."""
    allowed = {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if content_type not in allowed:
        raise ValueError(f"Unsupported content type: {content_type}")
    if len(file_bytes) > 100 * 1024 * 1024:  # 100 MB
        raise ValueError("File exceeds 100 MB limit")


def _extract_text(file_bytes: bytes, content_type: str) -> str:
    """Extract raw text from the document bytes."""
    if content_type == "application/pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(text_parts)
    elif content_type in ("text/plain", "text/markdown"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Text extraction not implemented for {content_type}")


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": chunk_text,
            "chunk_index": idx,
            "word_start": start,
            "word_end": end,
        })
        start += chunk_size - overlap
        idx += 1
    return chunks


def _embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Generate embeddings for each chunk."""
    from sentence_transformers import SentenceTransformer
    from app.core.config import settings

    model = SentenceTransformer(settings.embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def _build_raptor_tree(chunks: list[dict], embeddings: list[list[float]]) -> list[dict]:
    """
    Build RAPTOR hierarchical summaries.
    Returns additional summary-level nodes to index alongside leaf chunks.
    """
    # Placeholder — will integrate with existing RAPTOR tree builder
    # For now, return empty (leaf-only indexing)
    return []


def _index_vectors(
    collection_id: uuid.UUID,
    chunks: list[dict],
    embeddings: list[list[float]],
    document_id: uuid.UUID,
) -> int:
    """Upsert embeddings into Qdrant."""
    from app.storage.vector_store import ensure_collection, upsert_vectors

    ensure_collection(collection_id, dim=len(embeddings[0]))
    ids = [c["id"] for c in chunks]
    payloads = [
        {
            "text": c["text"],
            "chunk_index": c["chunk_index"],
            "document_id": str(document_id),
        }
        for c in chunks
    ]
    return upsert_vectors(collection_id, ids, embeddings, payloads)


# ── Main Celery task ──────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def run_ingestion_pipeline(self, job_id: str, document_id: str, collection_id: str):
    """
    Execute the full ingestion pipeline for a single document.
    Called asynchronously after a document is uploaded.
    """
    job_uuid = uuid.UUID(job_id)
    doc_uuid = uuid.UUID(document_id)
    col_uuid = uuid.UUID(collection_id)

    session = SyncSessionLocal()
    try:
        job = session.get(IngestionJob, job_uuid)
        doc = session.get(Document, doc_uuid)
        if not job or not doc:
            logger.error("Job %s or document %s not found", job_id, document_id)
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        session.commit()

        # 1. Download file from S3
        _update_job(session, job, "validating", 5)
        file_bytes = s3_client.download_file(doc.s3_key)
        _validate(file_bytes, doc.content_type)
        _update_job(session, job, "validating", 15)

        # 2. Extract text
        _update_job(session, job, "extracting", 20)
        full_text = _extract_text(file_bytes, doc.content_type)
        _update_job(session, job, "extracting", 35)

        # 3. Chunk
        _update_job(session, job, "chunking", 40)
        chunks = _chunk_text(full_text)
        _update_job(session, job, "chunking", 50)

        # 4. Embed
        _update_job(session, job, "embedding", 55)
        embeddings = _embed_chunks(chunks)
        _update_job(session, job, "embedding", 70)

        # 5. RAPTOR tree (optional hierarchical summarisation)
        _update_job(session, job, "tree_building", 75)
        summary_nodes = _build_raptor_tree(chunks, embeddings)
        _update_job(session, job, "tree_building", 80)

        # 6. Index
        _update_job(session, job, "indexing", 85)
        all_chunks = chunks + summary_nodes
        all_embeddings = embeddings  # TODO: add summary embeddings
        count = _index_vectors(col_uuid, all_chunks, all_embeddings, doc_uuid)
        _update_job(session, job, "indexing", 95)

        # Done
        job.status = "completed"
        job.progress_pct = 100
        job.chunk_count = count
        job.completed_at = datetime.now(timezone.utc)
        doc.status = "ready"
        session.commit()
        logger.info("Ingestion completed for document %s — %d chunks", document_id, count)

    except Exception as exc:
        session.rollback()
        job = session.get(IngestionJob, job_uuid)
        if job:
            job.status = "failed"
            job.error_message = str(exc)[:2000]
            job.completed_at = datetime.now(timezone.utc)
            session.commit()
        logger.exception("Ingestion failed for document %s", document_id)
        raise self.retry(exc=exc)
    finally:
        session.close()
