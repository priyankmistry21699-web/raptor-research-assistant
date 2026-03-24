"""
Admin routes — Platform administration (admin-only).

GET  /admin/stats                — Platform statistics
GET  /admin/models               — List registered models
POST /admin/models               — Register a new model
GET  /admin/audit                — Query audit logs
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Any

from app.db.session import get_db_sync
from app.db.models.user import User
from app.db.models.document import Document
from app.db.models.collection import Collection
from app.db.models.ingestion_job import IngestionJob
from app.db.models.model_registry import ModelRegistry
from app.db.models.audit_log import AuditLog
from app.core.security import require_role
from app.core.audit import log_audit_from_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Stats ─────────────────────────────────────────────────────────────


class PlatformStats(BaseModel):
    total_users: int
    total_collections: int
    total_documents: int
    total_ingestion_jobs: int
    jobs_by_status: dict[str, int]


@router.get("/stats", response_model=PlatformStats)
def get_stats(
    db: Session = Depends(get_db_sync),
    _admin: dict = Depends(require_role("admin")),
):
    """Platform-wide statistics."""
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_collections = db.query(func.count(Collection.id)).scalar() or 0
    total_documents = db.query(func.count(Document.id)).scalar() or 0
    total_jobs = db.query(func.count(IngestionJob.id)).scalar() or 0

    status_counts = (
        db.query(IngestionJob.status, func.count(IngestionJob.id))
        .group_by(IngestionJob.status)
        .all()
    )
    jobs_by_status = {status: count for status, count in status_counts}

    return PlatformStats(
        total_users=total_users,
        total_collections=total_collections,
        total_documents=total_documents,
        total_ingestion_jobs=total_jobs,
        jobs_by_status=jobs_by_status,
    )


# ── Model Registry ───────────────────────────────────────────────────


class ModelRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=50)
    model_type: str = Field(min_length=1, max_length=50)
    storage_key: str | None = None
    metrics: dict[str, Any] | None = None
    is_active: bool = False


class ModelRegistryOut(BaseModel):
    id: uuid.UUID
    name: str
    version: str
    model_type: str
    storage_key: str | None
    metrics: dict | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/models", response_model=list[ModelRegistryOut])
def list_models(
    model_type: str | None = Query(default=None),
    active_only: bool = Query(default=False),
    db: Session = Depends(get_db_sync),
    _admin: dict = Depends(require_role("admin")),
):
    """List registered models."""
    q = db.query(ModelRegistry)
    if model_type:
        q = q.filter(ModelRegistry.model_type == model_type)
    if active_only:
        q = q.filter(ModelRegistry.is_active.is_(True))
    return q.order_by(ModelRegistry.created_at.desc()).all()


@router.post("/models", response_model=ModelRegistryOut, status_code=201)
def register_model(
    req: ModelRegisterRequest,
    request: Request,
    db: Session = Depends(get_db_sync),
    _admin: dict = Depends(require_role("admin")),
):
    """Register a new model version."""
    existing = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.name == req.name, ModelRegistry.version == req.version)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Model name+version already exists")

    model = ModelRegistry(
        id=uuid.uuid4(),
        name=req.name,
        version=req.version,
        model_type=req.model_type,
        storage_key=req.storage_key,
        metrics=req.metrics,
        is_active=req.is_active,
    )
    db.add(model)
    log_audit_from_request(
        db, request, action="model.register", resource="model", resource_id=model.id
    )
    db.commit()
    db.refresh(model)
    return model


# ── Audit Logs ────────────────────────────────────────────────────────


class AuditLogOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None
    action: str
    resource: str | None
    resource_id: uuid.UUID | None
    details: dict | None
    ip_address: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/audit", response_model=list[AuditLogOut])
def list_audit_logs(
    user_id: uuid.UUID | None = Query(default=None),
    action: str | None = Query(default=None),
    _admin: dict = Depends(require_role("admin")),
    resource: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_sync),
):
    """Query audit logs with optional filters."""
    q = db.query(AuditLog)
    if user_id:
        q = q.filter(AuditLog.user_id == user_id)
    if action:
        q = q.filter(AuditLog.action == action)
    if resource:
        q = q.filter(AuditLog.resource == resource)
    q = q.order_by(AuditLog.created_at.desc())
    return q.offset(offset).limit(limit).all()
