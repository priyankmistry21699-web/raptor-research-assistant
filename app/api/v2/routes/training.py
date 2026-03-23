"""
Training routes — /api/v2/training

Manage DPO / fine-tuning runs.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.training import TrainingRun
from app.core.security import get_current_user, require_role
from app.api.v2.schemas import TrainingRunCreate, TrainingRunOut, PaginatedResponse

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/runs", response_model=TrainingRunOut, status_code=201)
async def start_training_run(
    body: TrainingRunCreate,
    db: AsyncSession = Depends(get_db),
    _editor: dict = Depends(require_role("editor")),
):
    run = TrainingRun(
        run_type=body.run_type,
        base_model=body.base_model,
        epochs=body.epochs,
        status="pending",
    )
    db.add(run)
    await db.flush()
    await db.refresh(run)
    # TODO: dispatch Celery training task
    return run


@router.get("/runs", response_model=PaginatedResponse)
async def list_training_runs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    offset = (page - 1) * page_size
    total_result = await db.execute(select(func.count()).select_from(TrainingRun))
    total = total_result.scalar() or 0
    result = await db.execute(
        select(TrainingRun)
        .order_by(TrainingRun.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    items = result.scalars().all()
    return PaginatedResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/runs/{run_id}", response_model=TrainingRunOut)
async def get_training_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    run = await db.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run
