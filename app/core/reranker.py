"""
BGE Reranker — Cross-encoder reranking for retrieved chunks.

Loads a cross-encoder model (default: BAAI/bge-reranker-base) to re-score
candidate chunks against the original query, producing higher-precision
top-k results before context assembly.
"""

import logging
import threading
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                model_name = settings.reranker.model
                logger.info("Loading reranker model: %s", model_name)
                _model = CrossEncoder(model_name)
                logger.info("Reranker model loaded")
    return _model


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int | None = None,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """
    Rerank chunks using cross-encoder scores.

    Args:
        query: The user query.
        chunks: List of dicts, each containing at least ``text_key``.
        top_k: Number of top results to return (default from settings).
        text_key: Key in each chunk dict containing the passage text.

    Returns:
        Sorted list of top-k chunks, each augmented with a ``rerank_score`` field.
    """
    if not settings.reranker.enabled or not chunks:
        return chunks

    top_k = top_k or settings.reranker.top_k

    model = _get_model()
    pairs = [(query, c[text_key]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
