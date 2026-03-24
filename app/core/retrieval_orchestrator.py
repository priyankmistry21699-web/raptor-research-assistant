"""
Retrieval Orchestrator — Full query-to-context pipeline.

Stages:
  1. Embed query
  2. Qdrant vector search (initial candidates)
  3. BGE cross-encoder reranking
  4. RAPTOR tree traversal (chunk → section → topic → document)
  5. Context assembly with deduplication
  6. Citation preparation
"""

import logging
import uuid
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Embedding ─────────────────────────────────────────────────────────

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(settings.embedding_model)
    return _embed_model


def embed_query(query: str) -> list[float]:
    """Encode a query string into a dense vector."""
    model = _get_embed_model()
    vec = model.encode(query, normalize_embeddings=True)
    return vec.tolist()


# ── Tree traversal ────────────────────────────────────────────────────


def _traverse_tree_nodes(
    session,
    chunk_doc_ids: list[str],
    collection_id: uuid.UUID,
) -> list[dict]:
    """
    Walk up the RAPTOR tree from chunk-level nodes to get
    section and topic summaries for richer context.
    Also fetch direct summary nodes that match the collection.
    """
    from app.db.models.tree_node import TreeNode

    extra_context = []
    seen_ids: set[uuid.UUID] = set()

    # Find tree nodes linked to these documents
    doc_nodes = (
        session.query(TreeNode)
        .filter(
            TreeNode.collection_id == collection_id,
            TreeNode.document_id.in_(chunk_doc_ids),
            TreeNode.node_type == "chunk",
        )
        .all()
    )

    # Walk up to parent (section/topic) nodes for summaries
    for node in doc_nodes:
        parent = node.parent
        while parent and parent.id not in seen_ids:
            seen_ids.add(parent.id)
            if parent.summary:
                extra_context.append(
                    {
                        "id": str(parent.id),
                        "text": parent.summary,
                        "node_type": parent.node_type,
                        "level": parent.level,
                        "label": parent.label,
                    }
                )
            parent = parent.parent

    # Also fetch any summary nodes (section/topic/root) for the collection
    # that may not be in the parent chain of matched chunks
    if not extra_context:
        summary_nodes = (
            session.query(TreeNode)
            .filter(
                TreeNode.collection_id == collection_id,
                TreeNode.node_type.in_(["section", "topic", "root"]),
                TreeNode.summary.isnot(None),
            )
            .order_by(TreeNode.level.desc())
            .limit(5)
            .all()
        )
        for sn in summary_nodes:
            if sn.id not in seen_ids:
                extra_context.append(
                    {
                        "id": str(sn.id),
                        "text": sn.summary,
                        "node_type": sn.node_type,
                        "level": sn.level,
                        "label": sn.label,
                    }
                )

    return extra_context


# ── Citation builder ──────────────────────────────────────────────────


def _build_citations(chunks: list[dict]) -> list[dict]:
    """
    Convert retrieved chunks into structured citation objects.
    Each citation includes the document id, chunk index, score, and snippet.
    """
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk.get("payload", {})
        citations.append(
            {
                "index": i,
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "score": chunk.get("rerank_score", chunk.get("score", 0)),
                "snippet": (chunk.get("text", "") or payload.get("text", ""))[:200],
            }
        )
    return citations


# ── Main orchestration ────────────────────────────────────────────────


def retrieve(
    query: str,
    collection_id: uuid.UUID,
    top_k: int = 10,
    with_tree_context: bool = True,
    filters: dict[str, Any] | None = None,
    session=None,
) -> dict:
    """
    Full retrieval pipeline.

    Returns:
        {
            "chunks": [...],      # ranked context chunks
            "citations": [...],   # structured citation objects
            "tree_context": [...] # additional RAPTOR tree summaries (if enabled)
        }
    """
    from app.storage.vector_store import search as qdrant_search
    from app.core.reranker import rerank

    # 1. Embed query
    query_vector = embed_query(query)

    # 2. Qdrant vector search — fetch more candidates than final top_k for reranking
    candidate_k = top_k * 3 if settings.reranker.enabled else top_k
    raw_results = qdrant_search(
        collection_id=collection_id,
        query_vector=query_vector,
        top_k=candidate_k,
        filters=filters,
    )

    # Normalize: ensure each result has a "text" key at top level
    for r in raw_results:
        if "text" not in r and r.get("payload"):
            r["text"] = r["payload"].get("text", "")

    # 3. Rerank
    ranked = rerank(query, raw_results, top_k=top_k)

    # 4. RAPTOR tree traversal (optional)
    tree_context: list[dict] = []
    if with_tree_context and session:
        doc_ids = list(
            {
                r.get("payload", {}).get("document_id")
                for r in ranked
                if r.get("payload", {}).get("document_id")
            }
        )
        if doc_ids:
            tree_context = _traverse_tree_nodes(session, doc_ids, collection_id)

    # 5. Build citations
    citations = _build_citations(ranked)

    return {
        "chunks": ranked,
        "citations": citations,
        "tree_context": tree_context,
    }


def build_context_text(retrieval_result: dict, max_tokens: int = 4000) -> str:
    """
    Assemble the final context string from retrieval results for LLM prompting.
    Includes both ranked chunks and tree-level summaries.
    Applies rough token budget to avoid exceeding LLM context window.
    """
    parts: list[str] = []
    budget = max_tokens * 4  # rough chars-to-tokens
    used = 0

    # Add tree-level context first (higher-level summaries)
    for tc in retrieval_result.get("tree_context", []):
        entry = f"[{tc['node_type'].upper()} — {tc.get('label', '')}]\n{tc['text']}\n"
        if used + len(entry) > budget:
            break
        parts.append(entry)
        used += len(entry)

    # Add retrieved chunks
    for i, chunk in enumerate(retrieval_result["chunks"], start=1):
        text = chunk.get("text", "") or chunk.get("payload", {}).get("text", "")
        entry = f"[Source {i}]\n{text}\n"
        if used + len(entry) > budget:
            break
        parts.append(entry)
        used += len(entry)

    return "\n".join(parts)
