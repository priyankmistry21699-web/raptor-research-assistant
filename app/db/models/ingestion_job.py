"""IngestionJob model — tracks document processing pipeline state."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Text,
    SmallInteger,
    Integer,
    DateTime,
    ForeignKey,
    CheckConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        server_default="pending",
        index=True,
    )
    # Stages: pending → validating → extracting → chunking → embedding
    #       → tree_building → indexing → completed / failed
    current_stage: Mapped[str | None] = mapped_column(String(50))
    progress_pct: Mapped[int] = mapped_column(
        SmallInteger, default=0, server_default="0"
    )
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
    )

    # Relationships
    document = relationship("Document", back_populates="ingestion_jobs")

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_ingestion_job_status",
        ),
        CheckConstraint(
            "progress_pct >= 0 AND progress_pct <= 100", name="ck_ingestion_progress"
        ),
    )
