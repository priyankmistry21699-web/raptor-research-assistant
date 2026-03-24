"""
Tests for app/core/vector_db.py — VectorDB (ChromaDB wrapper).
"""

import pytest
from app.core.vector_db import VectorDB


class TestVectorDB:
    """Tests for the ChromaDB-backed VectorDB."""

    @pytest.fixture(scope="class")
    def db(self):
        return VectorDB()

    def test_init(self, db):
        assert db is not None

    def test_count(self, db):
        count = db.count()
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.skipif(
        not VectorDB().count(), reason="ChromaDB is empty — skip search tests"
    )
    class TestWithData:
        """Tests that require data in ChromaDB."""

        @pytest.fixture(scope="class")
        def db(self):
            return VectorDB()

        @pytest.fixture(scope="class")
        def query_vec(self):
            from app.core.embedding import EmbeddingModel

            return EmbeddingModel().encode("transformer attention mechanism")

        def test_search(self, db, query_vec):
            results = db.search(query_vec, top_k=3)
            assert len(results) > 0
            assert len(results) <= 3

        def test_search_result_fields(self, db, query_vec):
            results = db.search(query_vec, top_k=1)
            r = results[0]
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "distance" in r
            assert "arxiv_id" in r

        def test_search_by_paper(self, db, query_vec):
            results = db.search_by_paper(query_vec, "1706.03762", top_k=3)
            assert all(r["arxiv_id"] == "1706.03762" for r in results)

        def test_get_by_id(self, db, query_vec):
            results = db.search(query_vec, top_k=1)
            doc = db.get_by_id(results[0]["id"])
            assert doc is not None
            assert "text" in doc
            assert "id" in doc

        def test_get_by_id_nonexistent(self, db):
            doc = db.get_by_id("nonexistent_id_12345")
            assert doc is None

    def test_upsert_chunks_method_exists(self, db):
        assert hasattr(db, "upsert_chunks")
        assert callable(db.upsert_chunks)
