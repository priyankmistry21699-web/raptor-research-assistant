"""Document and DocumentVersion models — uploaded files and their versions."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Text, BigInteger, Integer, DateTime, ForeignKey, JSON, CheckConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(100))
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    s3_key: Mapped[str | None] = mapped_column(String(1000))
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, server_default="{}")
    status: Mapped[str] = mapped_column(
        String(50), default="uploaded", server_default="uploaded", index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"), onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    collection = relationship("Collection", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    ingestion_jobs = relationship("IngestionJob", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(
            "status IN ('uploaded', 'processing', 'ready', 'failed', 'archived')",
            name="ck_document_status",
        ),
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    s3_key: Mapped[str | None] = mapped_column(String(1000))
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
    )

    # Relationships
    document = relationship("Document", back_populates="versions")
