"""
RAPTOR Tree Builder — Hierarchical recursive summarization.

Implements the core RAPTOR algorithm:
  1. Cluster leaf chunks using KMeans
  2. Summarize each cluster with LLM
  3. Embed summaries
  4. Recurse: cluster summaries → summarize → embed → ...
  5. Stop when single root or max depth reached

Produces a list of TreeNode dicts with parent_id links for persistence.
"""

import logging
import uuid

import numpy as np
from sklearn.cluster import KMeans

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_raptor_tree(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    max_depth: int = 3,
) -> list[dict]:
    """
    Build a RAPTOR hierarchical tree from leaf chunks.

    Args:
        chunks: List of chunk dicts with 'id', 'text', 'chunk_index'.
        embeddings: Corresponding embedding vectors.
        collection_id: Optional collection UUID for metadata.
        document_id: Optional document UUID for metadata.
        max_depth: Maximum tree depth (1 = one level of clustering above leaves).

    Returns:
        List of summary node dicts suitable for indexing:
        [{"id", "text", "chunk_index", "node_type", "level", "label",
          "parent_id", "children_ids", "summary", "embedding"}]
    """
    if not chunks or not embeddings:
        return []

    emb_array = np.array(embeddings, dtype=np.float32)
    summary_nodes: list[dict] = []

    # Current level items to cluster
    current_items = [
        {
            "id": c["id"],
            "text": c["text"],
            "embedding": emb_array[i],
            "level": 0,
            "node_type": "chunk",
        }
        for i, c in enumerate(chunks)
    ]

    for depth in range(1, max_depth + 1):
        if len(current_items) <= 1:
            break

        # Determine number of clusters
        n_clusters = _compute_n_clusters(len(current_items))
        if n_clusters <= 1:
            # Single cluster → create root summary
            root_node = _create_summary_node(
                items=current_items,
                level=depth,
                node_type="root" if depth == max_depth else "topic",
                cluster_idx=0,
                collection_id=collection_id,
                document_id=document_id,
            )
            if root_node:
                summary_nodes.append(root_node)
            break

        # Cluster current items
        item_embeddings = np.array(
            [it["embedding"] for it in current_items], dtype=np.float32
        )
        labels = _cluster_embeddings(item_embeddings, n_clusters)

        # Group items by cluster
        clusters: dict[int, list[dict]] = {}
        for item, label in zip(current_items, labels):
            clusters.setdefault(label, []).append(item)

        # Summarize each cluster
        next_level_items = []
        for cluster_idx, cluster_items in sorted(clusters.items()):
            if len(cluster_items) == 1:
                # Single item cluster — promote without summarizing
                item = cluster_items[0]
                item["level"] = depth
                next_level_items.append(item)
                continue

            node_type = "section" if depth == 1 else "topic"
            summary_node = _create_summary_node(
                items=cluster_items,
                level=depth,
                node_type=node_type,
                cluster_idx=cluster_idx,
                collection_id=collection_id,
                document_id=document_id,
            )
            if summary_node:
                summary_nodes.append(summary_node)
                next_level_items.append(summary_node)

        current_items = next_level_items

    # Final root if multiple items remain
    if len(current_items) > 1:
        root = _create_summary_node(
            items=current_items,
            level=max_depth + 1,
            node_type="root",
            cluster_idx=0,
            collection_id=collection_id,
            document_id=document_id,
        )
        if root:
            summary_nodes.append(root)

    logger.info(
        "RAPTOR tree built: %d leaf chunks → %d summary nodes",
        len(chunks),
        len(summary_nodes),
    )
    return summary_nodes


def _compute_n_clusters(n_items: int) -> int:
    """Determine number of clusters based on item count."""
    max_topics = settings.raptor_max_topics
    if n_items <= max_topics:
        return max(1, n_items // 2)
    return min(max_topics, max(2, int(np.sqrt(n_items))))


def _cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> list[int]:
    """Cluster embeddings using KMeans. Returns list of cluster labels."""
    n_clusters = min(n_clusters, len(embeddings))
    if n_clusters <= 1:
        return [0] * len(embeddings)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    labels = kmeans.fit_predict(embeddings)
    return labels.tolist()


def _create_summary_node(
    items: list[dict],
    level: int,
    node_type: str,
    cluster_idx: int,
    collection_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
) -> dict | None:
    """
    Create a summary node by combining and summarizing cluster items.
    Returns a node dict with an embedded summary, or None on failure.
    """
    # Combine texts for summarization
    combined_text = "\n\n".join(
        it["text"][:1000] for it in items[:20]
    )  # cap input size
    children_ids = [it["id"] for it in items]

    # Generate summary via LLM
    summary = _generate_summary(combined_text, node_type, level)
    if not summary:
        return None

    # Embed the summary
    embedding = _embed_text(summary)
    if embedding is None:
        return None

    node_id = str(uuid.uuid4())
    label = f"{node_type}-L{level}-C{cluster_idx}"

    return {
        "id": node_id,
        "text": summary,
        "chunk_index": -1,  # not a leaf chunk
        "node_type": node_type,
        "level": level,
        "label": label,
        "parent_id": None,  # set during persistence
        "children_ids": children_ids,
        "summary": summary,
        "embedding": embedding,
        "collection_id": str(collection_id) if collection_id else None,
        "document_id": str(document_id) if document_id else None,
    }


def _generate_summary(text: str, node_type: str, level: int) -> str | None:
    """Use LLM to generate a summary of the combined text."""
    prompt_map = {
        "section": "Summarize the following text passages into a coherent section summary. Focus on key findings, methods, and conclusions:",
        "topic": "Synthesize the following section summaries into a high-level topic overview. Identify common themes and key insights:",
        "root": "Create a comprehensive executive summary of the following topic summaries. Capture the main contributions, methodology, and conclusions:",
    }
    system_prompt = prompt_map.get(node_type, prompt_map["section"])

    try:
        from app.core.generation import generate

        result = generate(
            question=text[:8000],
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=512,
        )
        return result.get("content", "").strip()
    except Exception as e:
        logger.warning("LLM summary generation failed (level=%d): %s", level, e)
        # Fallback: extractive summary (first N sentences)
        sentences = text.split(". ")
        return ". ".join(sentences[:5]) + "." if sentences else None


def _embed_text(text: str) -> list[float] | None:
    """Embed a text string using the configured embedding model."""
    try:
        from app.core.retrieval_orchestrator import embed_query

        return embed_query(text)
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None
