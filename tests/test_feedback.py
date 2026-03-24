"""
Tests for app/core/feedback.py — FeedbackEntry and FeedbackStore.
"""

import os
import pytest
from app.core.feedback import FeedbackEntry, FeedbackStore, FEEDBACK_TYPES


class TestFeedbackEntry:
    """Tests for FeedbackEntry construction and serialization."""

    def test_create_valid_entry(self):
        entry = FeedbackEntry(
            session_id="s1",
            question="What is attention?",
            answer="A mechanism for weighting inputs.",
            feedback_type="helpful",
        )
        assert entry.session_id == "s1"
        assert entry.feedback_type == "helpful"
        assert entry.timestamp is not None

    def test_invalid_feedback_type(self):
        with pytest.raises(ValueError, match="Invalid feedback type"):
            FeedbackEntry(
                session_id="s1",
                question="Q",
                answer="A",
                feedback_type="invalid_type",
            )

    def test_all_feedback_types_valid(self):
        for ftype in FEEDBACK_TYPES:
            entry = FeedbackEntry(
                session_id="s1",
                question="Q",
                answer="A",
                feedback_type=ftype,
            )
            assert entry.feedback_type == ftype

    def test_to_dict(self):
        entry = FeedbackEntry(
            session_id="s1",
            question="Q",
            answer="A",
            feedback_type="correction",
            correction="Better answer",
            model_used="mistral",
            task="qa",
            citations=[{"arxiv_id": "123"}],
        )
        d = entry.to_dict()
        assert d["session_id"] == "s1"
        assert d["feedback_type"] == "correction"
        assert d["correction"] == "Better answer"
        assert d["model_used"] == "mistral"
        assert len(d["citations"]) == 1
        assert "timestamp" in d

    def test_defaults(self):
        entry = FeedbackEntry(
            session_id="s1",
            question="Q",
            answer="A",
            feedback_type="helpful",
        )
        assert entry.correction == ""
        assert entry.model_used == ""
        assert entry.task == "qa"
        assert entry.citations == []
        assert entry.context_chunks == []


class TestFeedbackStore:
    """Tests for FeedbackStore persistence and querying."""

    def test_empty_store(self, feedback_store):
        assert feedback_store.count() == 0
        assert feedback_store.get_all() == []

    def test_add_entry(self, feedback_store):
        entry = FeedbackEntry(
            session_id="s1",
            question="Q",
            answer="A",
            feedback_type="helpful",
        )
        result = feedback_store.add(entry)
        assert result["session_id"] == "s1"
        assert result["feedback_type"] == "helpful"
        assert feedback_store.count() == 1

    def test_submit_shortcut(self, feedback_store):
        result = feedback_store.submit(
            session_id="s1",
            question="Q",
            answer="A",
            feedback_type="incorrect",
            model_used="mistral",
        )
        assert result["session_id"] == "s1"
        assert result["feedback_type"] == "incorrect"
        assert feedback_store.count() == 1

    def test_get_all(self, feedback_store):
        feedback_store.submit(
            session_id="s1", question="Q1", answer="A1", feedback_type="helpful"
        )
        feedback_store.submit(
            session_id="s2", question="Q2", answer="A2", feedback_type="incorrect"
        )
        entries = feedback_store.get_all()
        assert len(entries) == 2

    def test_get_by_session(self, feedback_store):
        feedback_store.submit(
            session_id="s1", question="Q1", answer="A1", feedback_type="helpful"
        )
        feedback_store.submit(
            session_id="s2", question="Q2", answer="A2", feedback_type="incorrect"
        )
        feedback_store.submit(
            session_id="s1",
            question="Q3",
            answer="A3",
            feedback_type="correction",
            correction="Fixed",
        )
        entries = feedback_store.get_by_session("s1")
        assert len(entries) == 2
        assert all(e["session_id"] == "s1" for e in entries)

    def test_get_by_type(self, feedback_store):
        feedback_store.submit(
            session_id="s1", question="Q1", answer="A1", feedback_type="helpful"
        )
        feedback_store.submit(
            session_id="s2", question="Q2", answer="A2", feedback_type="helpful"
        )
        feedback_store.submit(
            session_id="s3", question="Q3", answer="A3", feedback_type="incorrect"
        )
        entries = feedback_store.get_by_type("helpful")
        assert len(entries) == 2

    def test_get_stats(self, feedback_store):
        feedback_store.submit(
            session_id="s1", question="Q1", answer="A1", feedback_type="helpful"
        )
        feedback_store.submit(
            session_id="s1", question="Q2", answer="A2", feedback_type="incorrect"
        )
        feedback_store.submit(
            session_id="s2", question="Q3", answer="A3", feedback_type="helpful"
        )
        stats = feedback_store.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["helpful"] == 2
        assert stats["by_type"]["incorrect"] == 1
        assert stats["unique_sessions"] == 2

    def test_persistence(self, tmp_dir):
        filepath = os.path.join(str(tmp_dir), "persist_test.jsonl")
        store1 = FeedbackStore(filepath=filepath)
        store1.submit(
            session_id="s1", question="Q", answer="A", feedback_type="helpful"
        )

        # New store reading same file
        store2 = FeedbackStore(filepath=filepath)
        assert store2.count() == 1
        assert store2.get_all()[0]["question"] == "Q"
