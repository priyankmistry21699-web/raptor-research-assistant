"""
Tests for app/core/retrieval.py — RaptorRetriever.
"""
import pytest
from app.core.retrieval import RaptorRetriever


@pytest.fixture(scope="module")
def retriever():
    return RaptorRetriever()


class TestRaptorRetriever:
    """Tests for the hybrid retrieval engine."""

    def test_init(self, retriever):
        assert retriever is not None

    def test_list_available_papers(self, retriever):
        papers = retriever.list_available_papers()
        assert isinstance(papers, list)
        assert len(papers) > 0

    def test_get_paper_overview(self, retriever):
        overview = retriever.get_paper_overview("1706.03762")
        if overview is not None:
            assert "title" in overview or "tree_levels" in overview

    def test_get_paper_overview_nonexistent(self, retriever):
        overview = retriever.get_paper_overview("NONEXISTENT")
        assert overview is None

    def test_retrieve_by_tree(self, retriever):
        results = retriever.retrieve_by_tree("1706.03762")
        assert isinstance(results, list)

    def test_retrieve_by_tree_with_section(self, retriever):
        results = retriever.retrieve_by_tree("1706.03762", section="Introduction")
        assert isinstance(results, list)

    def test_retrieve_hybrid(self, retriever):
        """Test hybrid retrieval (vector + tree). Requires ChromaDB data."""
        from app.core.vector_db import VectorDB
        db = VectorDB()
        if db.count() > 0:
            results = retriever.retrieve("What is the transformer architecture?", top_k=3)
            assert isinstance(results, list)
            assert len(results) > 0

    def test_retrieve_with_tree_context(self, retriever):
        from app.core.vector_db import VectorDB
        db = VectorDB()
        if db.count() > 0:
            results = retriever.retrieve(
                "attention mechanism", top_k=2, include_tree_context=True,
            )
            if results:
                assert "tree_context" in results[0]

    def test_retrieve_without_tree_context(self, retriever):
        from app.core.vector_db import VectorDB
        db = VectorDB()
        if db.count() > 0:
            results = retriever.retrieve(
                "neural network", top_k=2, include_tree_context=False,
            )
            if results:
                assert "tree_context" not in results[0]
