"""
Chat routes — /api/v2/chat

Session-based chat with RAG retrieval and citation tracking.
"""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.chat import ChatSession, ChatMessage
from app.api.v2.schemas import (
    ChatSessionCreate, ChatSessionOut,
    ChatMessageIn, ChatMessageOut,
)

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Sessions ──────────────────────────────────────────────────────────

@router.post("/sessions", response_model=ChatSessionOut, status_code=201)
async def create_session(
    body: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
):
    session = ChatSession(
        collection_id=body.collection_id,
        title=body.title,
        # TODO: user_id from auth middleware
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).order_by(ChatSession.updated_at.desc())
    )
    return result.scalars().all()


@router.get("/sessions/{session_id}", response_model=list[ChatMessageOut])
async def get_session_messages(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    return result.scalars().all()


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)


# ── Messages (RAG chat) ──────────────────────────────────────────────

@router.post("/sessions/{session_id}/messages", response_model=ChatMessageOut)
async def send_message(
    session_id: uuid.UUID,
    body: ChatMessageIn,
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a user message, run retrieval + LLM generation, return
    assistant response with citations.
    """
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=body.content,
    )
    db.add(user_msg)
    await db.flush()

    # Retrieve context from Qdrant
    start = time.perf_counter()
    from app.storage.vector_store import search as qdrant_search
    from app.core.config import settings
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    query_vec = model.encode(body.content, normalize_embeddings=True).tolist()
    chunks = qdrant_search(session.collection_id, query_vec, top_k=8)

    # Build prompt with retrieved context
    context_text = "\n\n".join(
        f"[{i+1}] {c['payload'].get('text', '')}" for i, c in enumerate(chunks)
    )
    citations = {
        str(i+1): {
            "chunk_id": c["id"],
            "document_id": c["payload"].get("document_id"),
            "score": c["score"],
        }
        for i, c in enumerate(chunks)
    }

    system_prompt = (
        "You are a helpful research assistant. Answer the user's question based on the "
        "provided context. Cite sources using [1], [2], etc. If the context is insufficient, "
        "say so clearly.\n\n"
        f"Context:\n{context_text}"
    )

    # Call LLM
    from app.core.llm_client import run_llm_messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": body.content},
    ]
    answer = run_llm_messages(messages)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Save assistant message
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=answer,
        citations=citations,
        model_used=settings.llm.model,
        latency_ms=elapsed_ms,
    )
    db.add(assistant_msg)
    await db.flush()
    await db.refresh(assistant_msg)

    return assistant_msg
