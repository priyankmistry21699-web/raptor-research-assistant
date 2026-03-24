"""
Tests for app/core/ingestion.py — arXiv paper fetching and PDF download.

Note: Network tests (actual arXiv calls) are marked with @pytest.mark.network
and skipped by default. Run with: pytest -m network
"""

import os
import pytest
from app.core.ingestion import (
    CATEGORIES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    fetch_arxiv_papers,
    save_metadata,
)


class TestIngestionConstants:
    """Tests for module-level constants."""

    def test_categories_defined(self):
        assert len(CATEGORIES) > 0
        assert all(isinstance(c, str) for c in CATEGORIES)

    def test_directories_are_strings(self):
        assert isinstance(RAW_DATA_DIR, str)
        assert isinstance(PROCESSED_DATA_DIR, str)


class TestSaveMetadata:
    """Tests for save_metadata()."""

    def test_save_and_read(self, tmp_dir):
        papers = [
            {"title": "Test Paper", "arxiv_id": "0000.00001", "authors": ["Author A"]},
        ]
        filepath = os.path.join(str(tmp_dir), "test_meta.json")
        save_metadata(papers, filepath)

        import json

        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test Paper"

    def test_save_empty_list(self, tmp_dir):
        filepath = os.path.join(str(tmp_dir), "empty_meta.json")
        save_metadata([], filepath)

        import json

        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert loaded == []


@pytest.mark.skipif(True, reason="Requires network access to arXiv API")
class TestFetchArxiv:
    """Tests for fetch_arxiv_papers() — requires network access."""

    def test_fetch_small_batch(self):
        papers = fetch_arxiv_papers(
            categories=["cs.AI"],
            max_results=2,
            date_from="2024-01-01",
            date_to="2024-12-31",
        )
        assert isinstance(papers, list)
        # May be empty if no papers match, but should not error
