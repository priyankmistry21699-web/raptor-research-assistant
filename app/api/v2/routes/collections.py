"""
Collection routes — /api/v2/workspaces/{workspace_id}/collections
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.collection import Collection
from app.db.models.workspace import Workspace
from app.api.v2.schemas import CollectionCreate, CollectionOut

router = APIRouter(
    prefix="/workspaces/{workspace_id}/collections",
    tags=["collections"],
)


@router.post("", response_model=CollectionOut, status_code=201)
async def create_collection(
    workspace_id: uuid.UUID,
    body: CollectionCreate,
    db: AsyncSession = Depends(get_db),
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


@router.get("", response_model=list[CollectionOut])
async def list_collections(
    workspace_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Collection)
        .where(Collection.workspace_id == workspace_id)
        .order_by(Collection.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{collection_id}", response_model=CollectionOut)
async def get_collection(
    workspace_id: uuid.UUID,
    collection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
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
):
    coll = await db.get(Collection, collection_id)
    if not coll or coll.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Collection not found")
    await db.delete(coll)
