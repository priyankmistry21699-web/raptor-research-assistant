"""
Tests for app/core/embedding.py — EmbeddingModel.
"""

import pytest
from app.core.embedding import EmbeddingModel


class TestEmbeddingModel:
    """Tests for the SentenceTransformer embedding wrapper."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create a shared EmbeddingModel instance (model loading is slow)."""
        return EmbeddingModel()

    def test_init_default_model(self, embedder):
        assert embedder.model is not None

    def test_encode_returns_list(self, embedder):
        result = embedder.encode("hello world")
        assert isinstance(result, list)

    def test_encode_dimension(self, embedder):
        result = embedder.encode("test sentence")
        assert len(result) == 384  # all-MiniLM-L6-v2 produces 384-dim

    def test_encode_float_values(self, embedder):
        result = embedder.encode("test")
        assert all(isinstance(v, float) for v in result)

    def test_similar_texts_closer(self, embedder):
        """Semantically similar texts should have higher cosine similarity."""
        import math

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x**2 for x in a))
            norm_b = math.sqrt(sum(x**2 for x in b))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0

        v1 = embedder.encode("neural network deep learning")
        v2 = embedder.encode("machine learning neural nets")
        v3 = embedder.encode("cooking recipe for pasta")

        sim_related = cosine_sim(v1, v2)
        sim_unrelated = cosine_sim(v1, v3)
        assert sim_related > sim_unrelated, (
            f"Related texts should be closer: {sim_related:.3f} vs {sim_unrelated:.3f}"
        )

    def test_encode_empty_string(self, embedder):
        result = embedder.encode("")
        assert isinstance(result, list)
        assert len(result) == 384
