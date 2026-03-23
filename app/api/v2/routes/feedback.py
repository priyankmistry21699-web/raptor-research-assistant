"""
Feedback routes — /api/v2/feedback

Captures user ratings and comments on assistant messages.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.feedback import Feedback
from app.db.models.chat import ChatMessage
from app.api.v2.schemas import FeedbackCreate, FeedbackOut

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut, status_code=201)
async def submit_feedback(
    body: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    # Verify the message exists
    msg = await db.get(ChatMessage, body.message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    if msg.role != "assistant":
        raise HTTPException(status_code=400, detail="Feedback must be on assistant messages")

    # Check for existing feedback
    existing = await db.execute(
        select(Feedback).where(Feedback.message_id == body.message_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Feedback already submitted for this message")

    fb = Feedback(
        message_id=body.message_id,
        rating=body.rating,
        comment=body.comment,
    )
    db.add(fb)
    await db.flush()
    await db.refresh(fb)
    return fb


@router.get("", response_model=list[FeedbackOut])
async def list_feedback(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Feedback)
        .order_by(Feedback.created_at.desc())
        .offset(offset)
        .limit(min(limit, 200))
    )
    return result.scalars().all()
