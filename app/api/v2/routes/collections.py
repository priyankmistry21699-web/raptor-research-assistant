"""
Collection routes — /api/v2/workspaces/{workspace_id}/collections
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.collection import Collection
from app.db.models.workspace import Workspace
from app.core.security import get_current_user
from app.api.v2.schemas import CollectionCreate, CollectionOut, PaginatedResponse

router = APIRouter(
    prefix="/workspaces/{workspace_id}/collections",
    tags=["collections"],
)


@router.post("", response_model=CollectionOut, status_code=201)
async def create_collection(
    workspace_id: uuid.UUID,
    body: CollectionCreate,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    ws = await db.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    coll = Collection(
        workspace_id=workspace_id,
        name=body.name,
        description=body.description,
    )
    db.add(coll)
    await db.flush()
    await db.refresh(coll)
    return coll


@router.get("", response_model=PaginatedResponse)
async def list_collections(
    workspace_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    base = select(Collection).where(Collection.workspace_id == workspace_id)
    total_result = await db.execute(select(func.count()).select_from(base.subquery()))
    total = total_result.scalar() or 0
    offset = (page - 1) * page_size
    result = await db.execute(
        base.order_by(Collection.created_at.desc()).offset(offset).limit(page_size)
    )
    items = result.scalars().all()
    return PaginatedResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{collection_id}", response_model=CollectionOut)
async def get_collection(
    workspace_id: uuid.UUID,
    collection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    coll = await db.get(Collection, collection_id)
    if not coll or coll.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Collection not found")
    return coll


@router.delete("/{collection_id}", status_code=204)
async def delete_collection(
    workspace_id: uuid.UUID,
    collection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    coll = await db.get(Collection, collection_id)
    if not coll or coll.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Collection not found")
    await db.delete(coll)
