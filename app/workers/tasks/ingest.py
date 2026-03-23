"""
Document ingestion pipeline — Celery task.

Stages:
  pending → validating → extracting → normalizing → chunking → embedding
        → tree_building → indexing → completed  (or → failed)
"""

import hashlib
import io
import logging
import re
import uuid
from datetime import datetime, timezone

from celery.exceptions import MaxRetriesExceededError

from app.workers.celery_app import celery_app
from app.db.session import SyncSessionLocal
from app.db.models.ingestion_job import IngestionJob
from app.db.models.document import Document
from app.db.models.chunk import ChunkMetadata
from app.db.models.tree_node import TreeNode
from app.storage import s3_client

logger = logging.getLogger(__name__)

STAGES = [
    "validating",
    "extracting",
    "normalizing",
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


def _normalize_text(text: str) -> str:
    """
    Normalize / clean extracted text:
      - Collapse excessive whitespace and newlines
      - Remove control characters except newlines/tabs
      - Strip leading/trailing whitespace per line
      - Merge hyphenated line breaks (e.g. "computa-\\ntion" → "computation")
    """
    # Remove control chars (keep newline, tab, carriage return)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Merge hyphenated line breaks
    text = re.sub(r'-\s*\n\s*', '', text)
    # Collapse multiple blank lines into two newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing whitespace per line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()


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
            "text_hash": hashlib.sha256(chunk_text.encode()).hexdigest(),
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


def _build_raptor_tree(chunks: list[dict], embeddings: list[list[float]],
                       collection_id: uuid.UUID, document_id: uuid.UUID) -> list[dict]:
    """
    Build RAPTOR hierarchical summaries using the real tree builder.
    Returns additional summary-level nodes to index alongside leaf chunks.
    """
    try:
        from app.core.raptor_tree_builder import build_raptor_tree
        return build_raptor_tree(
            chunks=chunks,
            embeddings=embeddings,
            collection_id=collection_id,
            document_id=document_id,
        )
    except Exception as e:
        logger.warning("RAPTOR tree building failed, continuing with leaf-only: %s", e)
        return []


def _persist_chunk_metadata(
    session,
    chunks: list[dict],
    document_id: uuid.UUID,
    collection_id: uuid.UUID,
) -> None:
    """Save chunk metadata rows to PostgreSQL."""
    for c in chunks:
        cm = ChunkMetadata(
            id=uuid.UUID(c["id"]),
            document_id=document_id,
            collection_id=collection_id,
            chunk_index=c["chunk_index"],
            text_hash=c["text_hash"],
            word_start=c["word_start"],
            word_end=c["word_end"],
            vector_id=c["id"],
        )
        session.add(cm)
    session.flush()


def _persist_tree_nodes(
    session,
    chunks: list[dict],
    document_id: uuid.UUID,
    collection_id: uuid.UUID,
    summary_nodes: list[dict] | None = None,
) -> None:
    """Create TreeNode entries for both leaf chunks and summary nodes."""
    # Leaf nodes
    for c in chunks:
        node = TreeNode(
            id=uuid.uuid4(),
            collection_id=collection_id,
            document_id=document_id,
            node_type="chunk",
            level=0,
            label=f"chunk-{c['chunk_index']}",
            vector_id=c["id"],
        )
        session.add(node)

    # Summary nodes from RAPTOR tree
    if summary_nodes:
        for sn in summary_nodes:
            node = TreeNode(
                id=uuid.UUID(sn["id"]),
                collection_id=collection_id,
                document_id=document_id,
                node_type=sn.get("node_type", "section"),
                level=sn.get("level", 1),
                label=sn.get("label"),
                summary=sn.get("summary"),
                vector_id=sn["id"],
                children_count=len(sn.get("children_ids", [])),
            )
            session.add(node)

    session.flush()


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

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
)
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
        raw_text = _extract_text(file_bytes, doc.content_type)
        _update_job(session, job, "extracting", 30)

        # 3. Normalize / clean
        _update_job(session, job, "normalizing", 32)
        full_text = _normalize_text(raw_text)
        _update_job(session, job, "normalizing", 38)

        # 4. Chunk
        _update_job(session, job, "chunking", 40)
        chunks = _chunk_text(full_text)
        _update_job(session, job, "chunking", 48)

        # 4b. Persist chunk metadata to PostgreSQL
        _persist_chunk_metadata(session, chunks, doc_uuid, col_uuid)
        _update_job(session, job, "chunking", 50)

        # 5. Embed
        _update_job(session, job, "embedding", 55)
        embeddings = _embed_chunks(chunks)
        _update_job(session, job, "embedding", 70)

        # 6. RAPTOR tree (hierarchical summarisation)
        _update_job(session, job, "tree_building", 75)
        summary_nodes = _build_raptor_tree(chunks, embeddings, col_uuid, doc_uuid)
        _persist_tree_nodes(session, chunks, doc_uuid, col_uuid, summary_nodes=summary_nodes)
        _update_job(session, job, "tree_building", 80)

        # 7. Index
        _update_job(session, job, "indexing", 85)
        # Combine leaf chunks and summary nodes for vector indexing
        all_chunks = list(chunks)
        all_embeddings = list(embeddings)
        for sn in summary_nodes:
            if sn.get("embedding"):
                all_chunks.append({"id": sn["id"], "text": sn["text"], "chunk_index": -1})
                all_embeddings.append(sn["embedding"])
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
        try:
            raise self.retry(exc=exc)
        except MaxRetriesExceededError:
            logger.error("Max retries exceeded for document %s", document_id)
    finally:
        session.close()
