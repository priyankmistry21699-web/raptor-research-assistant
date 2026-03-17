"""
Pytest configuration & shared fixtures for RAPTOR Research Assistant tests.

Fixtures provide isolated instances of core components (sessions, feedback, preferences)
using temporary directories so tests don't modify production data.
"""
import os
import sys
import json
import tempfile
import shutil

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up automatically."""
    return tmp_path


@pytest.fixture
def config():
    """Load project config.yaml as a dict."""
    import yaml
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def session_manager():
    """Provide a fresh SessionManager instance (isolated from global singleton)."""
    from app.core.session import SessionManager
    return SessionManager(max_sessions=10)


@pytest.fixture
def feedback_store(tmp_dir):
    """Provide a FeedbackStore writing to a temp file."""
    from app.core.feedback import FeedbackStore
    filepath = os.path.join(str(tmp_dir), 'test_feedback.jsonl')
    return FeedbackStore(filepath=filepath)


@pytest.fixture
def preference_store(tmp_dir, feedback_store):
    """Provide a PreferenceStore writing to a temp file."""
    from app.core.preference import PreferenceStore
    filepath = os.path.join(str(tmp_dir), 'test_preferences.jsonl')
    return PreferenceStore(filepath=filepath)


@pytest.fixture
def sample_chunks():
    """Sample retrieved chunks for prompt building and chat tests."""
    return [
        {
            "arxiv_id": "1706.03762",
            "chunk_index": 0,
            "chunk_text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            "section_num": "1",
            "section_title": "Introduction",
            "section_summary": "Introduces the Transformer architecture.",
            "topic": "Sequence Modeling",
            "topic_summary": "Models for sequence-to-sequence tasks.",
            "paper_title": "Attention Is All You Need",
        },
        {
            "arxiv_id": "1706.03762",
            "chunk_index": 1,
            "chunk_text": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "section_num": "1",
            "section_title": "Introduction",
            "section_summary": "Introduces the Transformer architecture.",
            "topic": "Sequence Modeling",
            "topic_summary": "Models for sequence-to-sequence tasks.",
            "paper_title": "Attention Is All You Need",
        },
    ]


@pytest.fixture
def sample_feedback_entry():
    """A sample feedback submission dict."""
    return {
        "session_id": "test_session_001",
        "question": "What is the Transformer?",
        "answer": "The Transformer is a neural network architecture based on attention.",
        "feedback_type": "helpful",
        "correction": "",
        "model_used": "mistral",
        "task": "qa",
        "citations": [{"arxiv_id": "1706.03762", "paper_title": "Attention Is All You Need"}],
    }
