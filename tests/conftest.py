"""
Pytest configuration & shared fixtures for RAPTOR Research Assistant tests.

Fixtures provide isolated instances of core components (sessions, feedback, preferences)
using temporary directories so tests don't modify production data.
"""

import os
import sys

import networkx as nx
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up automatically."""
    return tmp_path


@pytest.fixture
def config():
    """Load project config.yaml as a dict."""
    import yaml

    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path, "r") as f:
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

    filepath = os.path.join(str(tmp_dir), "test_feedback.jsonl")
    return FeedbackStore(filepath=filepath)


@pytest.fixture
def preference_store(tmp_dir, feedback_store):
    """Provide a PreferenceStore writing to a temp file."""
    from app.core.preference import PreferenceStore

    filepath = os.path.join(str(tmp_dir), "test_preferences.jsonl")
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
        "citations": [
            {"arxiv_id": "1706.03762", "paper_title": "Attention Is All You Need"}
        ],
    }


@pytest.fixture
def sample_raptor_tree_dir(tmp_path, monkeypatch):
    """Provide a temporary RAPTOR tree directory populated with sample trees."""
    import app.core.raptor_index as raptor_index
    from app.core.raptor_index import save_tree

    tree_dir = tmp_path / "paper_trees"
    tree_dir.mkdir()
    monkeypatch.setattr(raptor_index, "TREE_DIR", str(tree_dir))

    graph = nx.DiGraph()
    graph.add_node(
        "root",
        type="paper",
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        metadata={"year": 2017},
    )

    topic_nodes = [
        (
            "topic_1",
            "Sequence Modeling",
            "Models for sequence-to-sequence tasks.",
        ),
        ("topic_2", "Attention Architecture", "Attention-only network design."),
    ]
    for topic_id, title, summary in topic_nodes:
        graph.add_node(topic_id, type="topic", title=title, summary=summary)
        graph.add_edge("root", topic_id)

    section_nodes = [
        (
            "section_1",
            "topic_1",
            "1",
            "Introduction",
            "Introduces the Transformer architecture.",
        ),
        (
            "section_2",
            "topic_1",
            "2",
            "Background",
            "Reviews prior sequence models.",
        ),
        (
            "section_3",
            "topic_2",
            "3",
            "Model Architecture",
            "Details the encoder-decoder attention stack.",
        ),
    ]
    for section_id, topic_id, section_num, title, summary in section_nodes:
        graph.add_node(
            section_id,
            type="section",
            section_num=section_num,
            title=title,
            summary=summary,
        )
        graph.add_edge(topic_id, section_id)

    chunk_texts = [
        (
            "section_1",
            "The dominant sequence transduction models use recurrence or convolution.",
        ),
        (
            "section_1",
            "The Transformer relies entirely on attention mechanisms.",
        ),
        ("section_1", "This improves parallelization during training."),
        ("section_2", "Recurrent models process tokens sequentially."),
        ("section_2", "Convolutional models shorten dependency paths."),
        ("section_2", "Attention provides direct pairwise interactions."),
        (
            "section_3",
            "The encoder is composed of stacked self-attention blocks.",
        ),
        (
            "section_3",
            "The decoder attends to encoder outputs autoregressively.",
        ),
    ]
    for chunk_index, (section_id, text) in enumerate(chunk_texts):
        chunk_id = f"chunk_{chunk_index}"
        graph.add_node(
            chunk_id,
            type="chunk",
            chunk_index=chunk_index,
            text=text,
        )
        graph.add_edge(section_id, chunk_id)

    save_tree("1706.03762", graph)
    save_tree("2000.00001", graph.copy())

    return tree_dir
