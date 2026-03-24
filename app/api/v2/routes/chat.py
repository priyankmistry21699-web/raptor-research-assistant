"""
Chat routes — /api/v2/chat

Session-based chat with RAG retrieval and citation tracking.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.chat import ChatSession, ChatMessage
from app.core.security import get_current_user
from app.api.v2.schemas import (
    ChatSessionCreate,
    ChatSessionOut,
    ChatMessageIn,
    ChatMessageOut,
)

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Sessions ──────────────────────────────────────────────────────────


@router.post("/sessions", response_model=ChatSessionOut, status_code=201)
async def create_session(
    body: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
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
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    result = await db.execute(
        select(ChatSession).order_by(ChatSession.updated_at.desc())
    )
    return result.scalars().all()


@router.get("/sessions/{session_id}", response_model=list[ChatMessageOut])
async def get_session_messages(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
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
    user: dict = Depends(get_current_user),
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
    user: dict = Depends(get_current_user),
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

    # Build chat history from previous messages
    history_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .where(ChatMessage.role.in_(["user", "assistant"]))
        .order_by(ChatMessage.created_at)
    )
    history_msgs = history_result.scalars().all()
    chat_history = [{"role": m.role, "content": m.content} for m in history_msgs]

    # Generate answer using unified generation layer (LiteLLM + retrieval orchestrator)
    from app.core.generation import generate_with_retrieval

    result = generate_with_retrieval(
        question=body.content,
        collection_id=str(session.collection_id),
        chat_history=chat_history,
        top_k=8,
    )

    # Save assistant message
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=result["content"],
        citations=result.get("citations"),
        model_used=result.get("model_used", "unknown"),
        latency_ms=result.get("latency_ms", 0),
    )
    db.add(assistant_msg)
    await db.flush()
    await db.refresh(assistant_msg)

    return assistant_msg
