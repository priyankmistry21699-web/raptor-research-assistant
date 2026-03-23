"""
Qdrant vector store client.

Wraps the Qdrant REST client for collection management, upsert, and search.
Each workspace/collection pair maps to a Qdrant collection named
``{prefix}_{collection_id}``.
"""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
    )


def _collection_name(collection_id: uuid.UUID) -> str:
    return f"{settings.qdrant.collection_prefix}_{collection_id.hex}"


# ── Collection lifecycle ─────────────────────────────────────────────

def ensure_collection(collection_id: uuid.UUID, dim: int | None = None) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    client = _get_client()
    name = _collection_name(collection_id)
    dim = dim or settings.embedding_dim
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection %s (dim=%d)", name, dim)


def delete_collection(collection_id: uuid.UUID) -> None:
    """Delete a Qdrant collection."""
    client = _get_client()
    name = _collection_name(collection_id)
    client.delete_collection(name)
    logger.info("Deleted Qdrant collection %s", name)


# ── Write ─────────────────────────────────────────────────────────────

def upsert_vectors(
    collection_id: uuid.UUID,
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    batch_size: int = 256,
) -> int:
    """Upsert vectors into the collection in batches. Returns count upserted."""
    client = _get_client()
    name = _collection_name(collection_id)
    total = 0
    for i in range(0, len(ids), batch_size):
        batch = [
            PointStruct(id=ids[j], vector=vectors[j], payload=payloads[j])
            for j in range(i, min(i + batch_size, len(ids)))
        ]
        client.upsert(collection_name=name, points=batch)
        total += len(batch)
    logger.info("Upserted %d vectors into %s", total, name)
    return total


def delete_vectors(collection_id: uuid.UUID, ids: list[str]) -> None:
    """Delete vectors by their point IDs."""
    client = _get_client()
    name = _collection_name(collection_id)
    client.delete(collection_name=name, points_selector=ids)


# ── Search ────────────────────────────────────────────────────────────

def search(
    collection_id: uuid.UUID,
    query_vector: list[float],
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Search for the nearest vectors.
    Returns list of dicts with keys: id, score, payload.
    """
    client = _get_client()
    name = _collection_name(collection_id)

    qdrant_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = client.search(
        collection_name=name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
    )
    return [
        {"id": str(r.id), "score": r.score, "payload": r.payload}
        for r in results
    ]


def collection_info(collection_id: uuid.UUID) -> dict | None:
    """Return collection info, or None if it doesn't exist."""
    client = _get_client()
    name = _collection_name(collection_id)
    if not client.collection_exists(name):
        return None
    info = client.get_collection(name)
    return {
        "name": name,
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
    }
