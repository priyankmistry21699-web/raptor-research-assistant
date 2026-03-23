"""
Training routes — /api/v2/training

Manage DPO / fine-tuning runs.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.training import TrainingRun
from app.api.v2.schemas import TrainingRunCreate, TrainingRunOut

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/runs", response_model=TrainingRunOut, status_code=201)
async def start_training_run(
    body: TrainingRunCreate,
    db: AsyncSession = Depends(get_db),
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


@router.get("/runs", response_model=list[TrainingRunOut])
async def list_training_runs(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrainingRun)
        .order_by(TrainingRun.created_at.desc())
        .limit(min(limit, 100))
    )
    return result.scalars().all()


@router.get("/runs/{run_id}", response_model=TrainingRunOut)
async def get_training_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    run = await db.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run
