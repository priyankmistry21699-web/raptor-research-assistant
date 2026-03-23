"""Pydantic schemas for the v2 API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Generic ───────────────────────────────────────────────────────────

class Message(BaseModel):
    detail: str


class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int
    page_size: int


# ── Auth / User ───────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str | None
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Workspaces ────────────────────────────────────────────────────────

class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class WorkspaceOut(BaseModel):
    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Collections ───────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None


class CollectionOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Documents ─────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    filename: str
    content_type: str
    file_size_bytes: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class IngestionJobOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    status: str
    current_stage: str | None
    progress_pct: int
    chunk_count: int | None
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Chat ──────────────────────────────────────────────────────────────

class ChatSessionCreate(BaseModel):
    collection_id: uuid.UUID
    title: str | None = None


class ChatSessionOut(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=10000)


class ChatMessageOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    citations: dict | None
    model_used: str | None
    latency_ms: int | None
    token_count: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Retrieve ──────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    collection_id: uuid.UUID
    top_k: int = Field(default=10, ge=1, le=100)


class RetrieveChunk(BaseModel):
    id: str
    text: str
    score: float
    document_id: str | None
    chunk_index: int | None


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrieveChunk]


# ── Feedback ──────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    message_id: uuid.UUID
    rating: int = Field(ge=1, le=5)
    comment: str | None = None


class FeedbackOut(BaseModel):
    id: uuid.UUID
    message_id: uuid.UUID
    rating: int
    comment: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Training ──────────────────────────────────────────────────────────

class TrainingRunCreate(BaseModel):
    run_type: str = Field(default="dpo")
    base_model: str = Field(min_length=1)
    epochs: int | None = Field(default=1, ge=1)


class TrainingRunOut(BaseModel):
    id: uuid.UUID
    run_type: str
    status: str
    base_model: str
    pair_count: int | None
    epochs: int | None
    metrics: dict | None
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────────

class HealthCheck(BaseModel):
    status: str
    version: str


class ReadinessCheck(BaseModel):
    status: str
    database: str
    redis: str
    qdrant: str
    s3: str
