"""
Feedback API — Endpoints for collecting and querying user feedback.

Endpoints:
  POST /feedback              — Submit feedback on an answer
  GET  /feedback              — Get all feedback entries
  GET  /feedback/stats        — Get feedback summary statistics
  GET  /feedback/session/{id} — Get feedback for a specific session
  GET  /feedback/type/{type}  — Get feedback by type (helpful/incorrect/hallucination/correction)
"""
import os
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.feedback import feedback_store, FEEDBACK_TYPES

router = APIRouter(prefix="/feedback", tags=["feedback"])


# --- Request / Response models ---

class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    feedback_type: str  # helpful | incorrect | hallucination | correction
    correction: str = ""
    model_used: str = ""
    task: str = "qa"
    citations: Optional[List[Dict[str, str]]] = None

class FeedbackResponse(BaseModel):
    status: str
    feedback_type: str
    timestamp: str

class FeedbackStats(BaseModel):
    total: int
    by_type: Dict[str, int]
    unique_sessions: int


# --- Endpoints ---

@router.post("", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    """
    Submit feedback on an LLM answer.

    Feedback types:
      - helpful       : Answer was accurate and useful
      - incorrect     : Answer had factual errors
      - hallucination : Answer fabricated info not in sources
      - correction    : User provides corrected text (include in 'correction' field)
    """
    if req.feedback_type not in FEEDBACK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback type '{req.feedback_type}'. Must be one of: {sorted(FEEDBACK_TYPES)}"
        )
    if req.feedback_type == "correction" and not req.correction.strip():
        raise HTTPException(
            status_code=400,
            detail="Correction text is required when feedback_type is 'correction'"
        )

    record = feedback_store.submit(
        session_id=req.session_id,
        question=req.question,
        answer=req.answer,
        feedback_type=req.feedback_type,
        correction=req.correction,
        model_used=req.model_used,
        task=req.task,
        citations=req.citations,
    )

    return FeedbackResponse(
        status="recorded",
        feedback_type=req.feedback_type,
        timestamp=record["timestamp"],
    )


@router.get("", response_model=List[Dict[str, Any]])
def get_all_feedback():
    """Get all feedback entries."""
    return feedback_store.get_all()


@router.get("/stats", response_model=FeedbackStats)
def get_feedback_stats():
    """Get summary statistics of all collected feedback."""
    return FeedbackStats(**feedback_store.get_stats())


@router.get("/session/{session_id}", response_model=List[Dict[str, Any]])
def get_feedback_by_session(session_id: str):
    """Get all feedback entries for a specific chat session."""
    entries = feedback_store.get_by_session(session_id)
    if not entries:
        raise HTTPException(status_code=404, detail="No feedback found for this session")
    return entries


@router.get("/type/{feedback_type}", response_model=List[Dict[str, Any]])
def get_feedback_by_type(feedback_type: str):
    """Get all feedback entries of a specific type."""
    if feedback_type not in FEEDBACK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid type '{feedback_type}'. Must be one of: {sorted(FEEDBACK_TYPES)}"
        )
    entries = feedback_store.get_by_type(feedback_type)
    return entries
