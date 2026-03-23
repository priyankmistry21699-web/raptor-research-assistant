"""TreeNode model — RAPTOR hierarchical tree structure stored in Postgres."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TreeNode(Base):
    __tablename__ = "tree_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tree_nodes.id", ondelete="SET NULL"),
    )
    node_type: Mapped[str] = mapped_column(
        String(50), nullable=False,
    )  # 'document', 'topic', 'section', 'chunk'
    level: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    label: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    vector_id: Mapped[str | None] = mapped_column(String(200))
    children_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
    )

    # Relationships
    collection = relationship("Collection", backref="tree_nodes")
    document = relationship("Document", backref="tree_nodes")
    parent = relationship("TreeNode", remote_side=[id], backref="children")
