"""
Retrieval routes — /api/v2/retrieve

Standalone semantic search endpoint (no LLM generation).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.v2.schemas import RetrieveRequest, RetrieveResponse, RetrieveChunk

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post("", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Semantic search against a collection's vector index."""
    from app.storage.vector_store import search as qdrant_search
    from app.core.config import settings
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    query_vec = model.encode(body.query, normalize_embeddings=True).tolist()
    results = qdrant_search(body.collection_id, query_vec, top_k=body.top_k)

    chunks = [
        RetrieveChunk(
            id=r["id"],
            text=r["payload"].get("text", ""),
            score=r["score"],
            document_id=r["payload"].get("document_id"),
            chunk_index=r["payload"].get("chunk_index"),
        )
        for r in results
    ]
    return RetrieveResponse(query=body.query, chunks=chunks)
