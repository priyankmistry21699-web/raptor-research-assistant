"""
Retrieval routes — /api/v2/retrieve

Standalone semantic search endpoint (no LLM generation).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.security import get_current_user
from app.api.v2.schemas import RetrieveRequest, RetrieveResponse, RetrieveChunk

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post("", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Semantic search against a collection's vector index."""
    from app.core.retrieval_orchestrator import retrieve as orchestrator_retrieve

    result = orchestrator_retrieve(
        query=body.query,
        collection_id=body.collection_id,
        top_k=body.top_k,
    )

    chunks = [
        RetrieveChunk(
            id=c.get("id", ""),
            text=c.get("text", ""),
            score=c.get("score", 0.0),
            document_id=c.get("document_id"),
            chunk_index=c.get("chunk_index"),
        )
        for c in result.get("chunks", [])
    ]
    return RetrieveResponse(query=body.query, chunks=chunks)
