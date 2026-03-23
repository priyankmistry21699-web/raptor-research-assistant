"""
Document routes — /api/v2/collections/{collection_id}/documents

Handles file upload → S3 → DB record → kicks off async ingestion pipeline.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.document import Document, DocumentVersion
from app.db.models.collection import Collection
from app.db.models.ingestion_job import IngestionJob
from app.core.security import get_current_user
from app.api.v2.schemas import DocumentOut, IngestionJobOut, PaginatedResponse
from app.storage import s3_client

router = APIRouter(
    prefix="/collections/{collection_id}/documents",
    tags=["documents"],
)


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    collection_id: uuid.UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Upload a document, store in S3, and start the ingestion pipeline."""
    coll = await db.get(Collection, collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Read file
    content = await file.read()
    content_type = file.content_type or "application/octet-stream"

    # Store in S3
    s3_key = f"documents/{collection_id}/{uuid.uuid4()}/{file.filename}"
    s3_client.upload_file(s3_key, __import__("io").BytesIO(content), content_type)

    # Create DB records
    doc = Document(
        collection_id=collection_id,
        filename=file.filename,
        content_type=content_type,
        file_size_bytes=len(content),
        s3_key=s3_key,
        status="processing",
    )
    db.add(doc)
    await db.flush()

    version = DocumentVersion(
        document_id=doc.id,
        version=1,
        s3_key=s3_key,
    )
    db.add(version)

    job = IngestionJob(document_id=doc.id, status="pending")
    db.add(job)
    await db.flush()
    await db.refresh(doc)

    # Dispatch Celery task
    from app.workers.tasks.ingest import run_ingestion_pipeline
    run_ingestion_pipeline.delay(str(job.id), str(doc.id), str(collection_id))

    return doc


@router.get("", response_model=PaginatedResponse)
async def list_documents(
    collection_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    base = select(Document).where(Document.collection_id == collection_id)
    total_result = await db.execute(select(func.count()).select_from(base.subquery()))
    total = total_result.scalar() or 0
    offset = (page - 1) * page_size
    result = await db.execute(
        base.order_by(Document.created_at.desc()).offset(offset).limit(page_size)
    )
    items = result.scalars().all()
    return PaginatedResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    doc = await db.get(Document, document_id)
    if not doc or doc.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/{document_id}/status", response_model=IngestionJobOut)
async def get_ingestion_status(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Get the latest ingestion job status for a document."""
    result = await db.execute(
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="No ingestion job found")
    return job


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    doc = await db.get(Document, document_id)
    if not doc or doc.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Document not found")
    # Clean up S3
    try:
        s3_client.delete_file(doc.s3_key)
    except Exception:
        pass  # Best-effort S3 cleanup
    await db.delete(doc)
