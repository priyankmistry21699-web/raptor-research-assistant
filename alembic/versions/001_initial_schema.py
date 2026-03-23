"""Initial schema - all models.

Revision ID: 001_initial
Revises: None
Create Date: 2025-01-20

All tables for the RAPTOR RAG Platform:
  users, workspaces, workspace_members, collections, documents,
  document_versions, ingestion_jobs, chunks_metadata, tree_nodes,
  chat_sessions, chat_messages, feedback, preference_pairs,
  training_runs, eval_runs, model_registry, audit_logs
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Users ─────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("clerk_id", sa.String(255), unique=True, nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("display_name", sa.String(255)),
        sa.Column("role", sa.String(50), server_default="user", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("role IN ('admin', 'editor', 'member', 'viewer', 'user')", name="ck_user_role"),
    )

    # ── Workspaces ────────────────────────────────────────────────
    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("settings", postgresql.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    op.create_table(
        "workspace_members",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.String(50), server_default="member"),
        sa.Column("joined_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("role IN ('owner', 'admin', 'editor', 'member', 'viewer')", name="ck_workspace_member_role"),
    )

    # ── Collections ───────────────────────────────────────────────
    op.create_table(
        "collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("settings", postgresql.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # ── Documents ─────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("content_type", sa.String(100)),
        sa.Column("file_size_bytes", sa.BigInteger),
        sa.Column("s3_key", sa.String(1000)),
        sa.Column("metadata", postgresql.JSON, server_default="{}"),
        sa.Column("status", sa.String(50), server_default="uploaded", index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("status IN ('uploaded', 'processing', 'ready', 'failed', 'archived')", name="ck_document_status"),
    )

    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("s3_key", sa.String(1000)),
        sa.Column("chunk_count", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # ── Ingestion Jobs ────────────────────────────────────────────
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("status", sa.String(50), server_default="pending", index=True),
        sa.Column("current_stage", sa.String(50)),
        sa.Column("progress_pct", sa.SmallInteger, server_default="0"),
        sa.Column("chunk_count", sa.Integer),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name="ck_ingestion_job_status"),
        sa.CheckConstraint("progress_pct >= 0 AND progress_pct <= 100", name="ck_ingestion_progress"),
    )

    # ── Chunks Metadata ───────────────────────────────────────────
    op.create_table(
        "chunks_metadata",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("word_start", sa.Integer),
        sa.Column("word_end", sa.Integer),
        sa.Column("page_number", sa.Integer),
        sa.Column("section_title", sa.Text),
        sa.Column("vector_id", sa.String(200)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_chunks_document_index", "chunks_metadata", ["document_id", "chunk_index"])
    op.create_index("ix_chunks_collection_document", "chunks_metadata", ["collection_id", "document_id"])

    # ── Tree Nodes ────────────────────────────────────────────────
    op.create_table(
        "tree_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), index=True),
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tree_nodes.id", ondelete="SET NULL")),
        sa.Column("node_type", sa.String(50), nullable=False),
        sa.Column("level", sa.Integer, server_default="0"),
        sa.Column("label", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("vector_id", sa.String(200)),
        sa.Column("children_count", sa.Integer, server_default="0"),
        sa.Column("metadata", postgresql.JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("node_type IN ('document', 'topic', 'section', 'chunk', 'root')", name="ck_tree_node_type"),
        sa.CheckConstraint("level >= 0", name="ck_tree_node_level"),
    )
    op.create_index("ix_tree_nodes_collection_type_level", "tree_nodes", ["collection_id", "node_type", "level"])

    # ── Chat Sessions ─────────────────────────────────────────────
    op.create_table(
        "chat_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(255)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # ── Chat Messages ─────────────────────────────────────────────
    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("citations", postgresql.JSONB),
        sa.Column("model_used", sa.String(100)),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("token_count", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_chat_message_role"),
    )
    op.create_index("ix_chat_messages_session_created", "chat_messages", ["session_id", "created_at"])

    # ── Feedback ──────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("rating", sa.SmallInteger, nullable=False),
        sa.Column("comment", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("rating >= 1 AND rating <= 5", name="ck_feedback_rating"),
    )

    # ── Preference Pairs ──────────────────────────────────────────
    op.create_table(
        "preference_pairs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("chosen", sa.Text, nullable=False),
        sa.Column("rejected", sa.Text, nullable=False),
        sa.Column("source", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # ── Training Runs ─────────────────────────────────────────────
    op.create_table(
        "training_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("run_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), server_default="pending", index=True),
        sa.Column("base_model", sa.String(200), nullable=False),
        sa.Column("pair_count", sa.Integer),
        sa.Column("epochs", sa.Integer),
        sa.Column("metrics", postgresql.JSONB),
        sa.Column("adapter_s3_key", sa.String(500)),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name="ck_training_run_status"),
        sa.CheckConstraint("run_type IN ('dpo', 'sft', 'rlhf')", name="ck_training_run_type"),
    )

    # ── Eval Runs ─────────────────────────────────────────────────
    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="SET NULL"), index=True),
        sa.Column("eval_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), server_default="pending", index=True),
        sa.Column("config", postgresql.JSONB),
        sa.Column("results", postgresql.JSONB),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name="ck_eval_run_status"),
    )

    # ── Model Registry ────────────────────────────────────────────
    op.create_table(
        "model_registry",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("storage_key", sa.String(500)),
        sa.Column("metrics", postgresql.JSONB),
        sa.Column("is_active", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("name", "version", name="uq_model_name_version"),
    )

    # ── Audit Logs ────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), index=True),
        sa.Column("action", sa.String(100), nullable=False, index=True),
        sa.Column("resource", sa.String(100), nullable=False),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True)),
        sa.Column("details", postgresql.JSONB),
        sa.Column("ip_address", sa.String(45)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("model_registry")
    op.drop_table("eval_runs")
    op.drop_table("training_runs")
    op.drop_table("preference_pairs")
    op.drop_table("feedback")
    op.drop_index("ix_chat_messages_session_created", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_index("ix_tree_nodes_collection_type_level", table_name="tree_nodes")
    op.drop_table("tree_nodes")
    op.drop_index("ix_chunks_collection_document", table_name="chunks_metadata")
    op.drop_index("ix_chunks_document_index", table_name="chunks_metadata")
    op.drop_table("chunks_metadata")
    op.drop_table("ingestion_jobs")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("collections")
    op.drop_table("workspace_members")
    op.drop_table("workspaces")
    op.drop_table("users")
