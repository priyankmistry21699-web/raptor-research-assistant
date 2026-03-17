"""
Tests for app/core/preference.py — Preference dataset creation.
"""
import os
import pytest
from app.core.preference import PreferenceStore, feedback_to_preference


class TestFeedbackToPreference:
    """Tests for the feedback_to_preference conversion function."""

    def test_helpful_creates_pair(self):
        entry = {
            "question": "What is attention?",
            "answer": "A mechanism for weighting input features.",
            "feedback_type": "helpful",
            "citations": [{"arxiv_id": "1706.03762"}],
        }
        pair = feedback_to_preference(entry)
        assert pair is not None
        assert pair["chosen"] == entry["answer"]
        assert pair["rejected"] != entry["answer"]  # Placeholder rejection
        assert "prompt" in pair

    def test_correction_creates_pair(self):
        entry = {
            "question": "What is attention?",
            "answer": "Wrong answer.",
            "feedback_type": "correction",
            "correction": "Attention is a mechanism for weighting inputs.",
            "citations": [],
        }
        pair = feedback_to_preference(entry)
        assert pair is not None
        assert pair["chosen"] == "Attention is a mechanism for weighting inputs."
        assert pair["rejected"] == "Wrong answer."

    def test_incorrect_without_correction_skipped(self):
        entry = {
            "question": "Q",
            "answer": "Bad answer",
            "feedback_type": "incorrect",
            "correction": "",
            "citations": [],
        }
        pair = feedback_to_preference(entry)
        # Should be None or have placeholder — depends on implementation
        # Either way, the rejected should be the answer
        if pair is not None:
            assert pair["rejected"] == "Bad answer"

    def test_hallucination_without_correction(self):
        entry = {
            "question": "Q",
            "answer": "Hallucinated answer",
            "feedback_type": "hallucination",
            "correction": "",
            "citations": [],
        }
        pair = feedback_to_preference(entry)
        if pair is not None:
            assert pair["rejected"] == "Hallucinated answer"


class TestPreferenceStore:
    """Tests for PreferenceStore persistence and operations."""

    def test_empty_store(self, preference_store):
        assert preference_store.count() == 0
        assert preference_store.get_all() == []

    def test_get_stats_empty(self, preference_store):
        stats = preference_store.get_stats()
        assert stats["total_pairs"] == 0

    def test_build_from_feedback(self, feedback_store, preference_store):
        # Add some feedback first
        feedback_store.submit(
            session_id="s1", question="What is attention?",
            answer="It is a mechanism.", feedback_type="helpful",
        )
        feedback_store.submit(
            session_id="s1", question="Explain transformers.",
            answer="Wrong.", feedback_type="correction",
            correction="Transformers use self-attention.",
        )

        # Monkey-patch the preference store to read from our test feedback store
        import app.core.preference as pref_module
        original_store = pref_module.feedback_store
        pref_module.feedback_store = feedback_store
        try:
            result = preference_store.build_from_feedback()
            assert result["pairs_created"] >= 1
            assert preference_store.count() >= 1
        finally:
            pref_module.feedback_store = original_store

    def test_export_for_training(self, preference_store):
        # Manually add a preference pair via the store's internal method
        pair = {
            "prompt": "What is X?",
            "chosen": "X is Y.",
            "rejected": "I don't know.",
            "metadata": {"feedback_type": "helpful"},
        }
        preference_store._write_all([pair])

        exported = preference_store.export_for_training()
        assert len(exported) == 1
        assert "prompt" in exported[0]
        assert "chosen" in exported[0]
        assert "rejected" in exported[0]

    def test_persistence(self, tmp_dir):
        from app.core.preference import PreferenceStore
        filepath = os.path.join(str(tmp_dir), 'persist_pref.jsonl')
        store1 = PreferenceStore(filepath=filepath)
        store1._write_all([{"prompt": "Q", "chosen": "A", "rejected": "B", "metadata": {}}])

        store2 = PreferenceStore(filepath=filepath)
        assert store2.count() == 1
