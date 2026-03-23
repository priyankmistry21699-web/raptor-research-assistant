"""
Generate routes — Standalone LLM generation + RAG generation.

POST /generate           — Generate with retrieved context (RAG)
POST /generate/raw       — Direct LLM generation without retrieval
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generate"])


class GenerateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    collection_id: uuid.UUID
    top_k: int = Field(default=10, ge=1, le=100)
    temperature: float | None = None
    max_tokens: int | None = None


class GenerateRawRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class GenerateResponse(BaseModel):
    content: str
    model_used: str | None
    latency_ms: int | None
    token_count: int | None
    citations: list[dict] | None = None


@router.post("", response_model=GenerateResponse)
def generate_with_rag(req: GenerateRequest, db: Session = Depends(get_db_sync)):
    """RAG generation: retrieve context from collection then generate."""
    from app.core.generation import generate_with_retrieval

    result = generate_with_retrieval(
        question=req.question,
        collection_id=req.collection_id,
        top_k=req.top_k,
        session=db,
    )
    return GenerateResponse(**result)


@router.post("/raw", response_model=GenerateResponse)
def generate_raw(req: GenerateRawRequest):
    """Direct LLM generation without retrieval."""
    from app.core.generation import generate

    result = generate(
        question=req.prompt,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return GenerateResponse(**result)
