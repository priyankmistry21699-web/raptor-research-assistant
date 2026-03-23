"""Training models — TrainingRun."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Integer, Float, Text, DateTime, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    run_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "dpo", "sft", "rlhf"
    status: Mapped[str] = mapped_column(
        String(50), default="pending", server_default="pending", index=True,
    )
    base_model: Mapped[str] = mapped_column(String(200), nullable=False)
    pair_count: Mapped[int | None] = mapped_column(Integer)
    epochs: Mapped[int | None] = mapped_column(Integer)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    adapter_s3_key: Mapped[str | None] = mapped_column(String(500))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
    )
