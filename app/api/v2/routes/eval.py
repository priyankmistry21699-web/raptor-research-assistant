"""
Eval routes — Evaluation run CRUD.

POST /eval/runs                  — Start new eval run
GET  /eval/runs                  — List eval runs
GET  /eval/runs/{run_id}         — Get eval run details + results
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Any

from app.db.session import get_db_sync
from app.db.models.eval_run import EvalRun

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRunCreate(BaseModel):
    collection_id: uuid.UUID | None = None
    eval_type: str = Field(default="ragas", min_length=1)
    config: dict[str, Any] | None = None


class EvalRunOut(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID | None
    eval_type: str
    status: str
    config: dict | None
    results: dict | None
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


@router.post("/runs", response_model=EvalRunOut, status_code=201)
def create_eval_run(req: EvalRunCreate, db: Session = Depends(get_db_sync)):
    """Start a new evaluation run."""
    run = EvalRun(
        id=uuid.uuid4(),
        collection_id=req.collection_id,
        eval_type=req.eval_type,
        config=req.config,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # TODO: dispatch Celery task for async eval execution
    logger.info("Created eval run %s (type=%s)", run.id, run.eval_type)
    return run


@router.get("/runs", response_model=list[EvalRunOut])
def list_eval_runs(
    collection_id: uuid.UUID | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_sync),
):
    """List evaluation runs with optional filters."""
    q = db.query(EvalRun)
    if collection_id:
        q = q.filter(EvalRun.collection_id == collection_id)
    if status:
        q = q.filter(EvalRun.status == status)
    q = q.order_by(EvalRun.created_at.desc())
    return q.offset(offset).limit(limit).all()


@router.get("/runs/{run_id}", response_model=EvalRunOut)
def get_eval_run(run_id: uuid.UUID, db: Session = Depends(get_db_sync)):
    """Get a specific evaluation run with results."""
    run = db.get(EvalRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return run
