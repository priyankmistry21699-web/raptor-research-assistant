"""
Tests for app/core/raptor_index.py — RAPTOR tree operations.

Uses a deterministic temporary RAPTOR tree fixture so tests do not depend on
ignored local artifacts under data/raw/paper_trees/.
"""

import pytest

from app.core.raptor_index import (
    load_tree,
    get_paper_info,
    get_topics,
    get_sections,
    get_chunks,
    get_tree_structure,
    get_tree_stats,
    list_all_papers,
    get_context_for_chunk,
)


@pytest.fixture(autouse=True)
def sample_tree_dir(sample_raptor_tree_dir):
    return sample_raptor_tree_dir


class TestListAllPapers:
    """Tests for list_all_papers()."""

    def test_returns_list(self):
        papers = list_all_papers()
        assert isinstance(papers, list)

    def test_has_papers(self):
        papers = list_all_papers()
        assert len(papers) > 0, "Expected at least some papers in the tree directory"


class TestLoadTree:
    """Tests for load_tree()."""

    def test_load_valid(self):
        papers = list_all_papers()
        if papers:
            G = load_tree(papers[0])
            assert G is not None
            assert len(G.nodes) > 0

    def test_load_nonexistent(self):
        G = load_tree("NONEXISTENT_PAPER_ID")
        assert G is None

    def test_load_known_paper(self):
        G = load_tree("1706.03762")
        if G is not None:
            assert len(G.nodes) > 10  # Should have many nodes


class TestGetPaperInfo:
    """Tests for get_paper_info()."""

    def test_returns_dict(self):
        G = load_tree("1706.03762")
        if G is not None:
            info = get_paper_info(G)
            assert isinstance(info, dict)
            assert "arxiv_id" in info or "title" in info


class TestGetTopics:
    """Tests for get_topics()."""

    def test_returns_list(self):
        G = load_tree("1706.03762")
        if G is not None:
            topics = get_topics(G)
            assert isinstance(topics, list)

    def test_topics_have_required_fields(self):
        G = load_tree("1706.03762")
        if G is not None:
            topics = get_topics(G)
            if topics:
                t = topics[0]
                assert "title" in t


class TestGetSections:
    """Tests for get_sections()."""

    def test_all_sections(self):
        G = load_tree("1706.03762")
        if G is not None:
            sections = get_sections(G)
            assert isinstance(sections, list)
            assert len(sections) > 0

    def test_sections_by_topic(self):
        G = load_tree("1706.03762")
        if G is not None:
            topics = get_topics(G)
            if topics:
                sections = get_sections(G, topics[0].get("node_id"))
                assert isinstance(sections, list)


class TestGetChunks:
    """Tests for get_chunks()."""

    def test_chunks_by_section(self):
        G = load_tree("1706.03762")
        if G is not None:
            sections = get_sections(G)
            if sections:
                chunks = get_chunks(G, sections[0].get("node_id"))
                assert isinstance(chunks, list)
                if chunks:
                    assert "text" in chunks[0]


class TestGetTreeStructure:
    """Tests for get_tree_structure()."""

    def test_basic_structure(self):
        struct = get_tree_structure("1706.03762")
        if struct is not None:
            assert "tree_levels" in struct
            assert struct["tree_levels"] in (3, 4)
            assert "total_chunks" in struct

    def test_nonexistent_paper(self):
        struct = get_tree_structure("NONEXISTENT")
        assert struct is None


class TestGetTreeStats:
    """Tests for get_tree_stats()."""

    def test_basic_stats(self):
        stats = get_tree_stats("1706.03762")
        if stats is not None:
            assert isinstance(stats, dict)

    def test_nonexistent(self):
        stats = get_tree_stats("NONEXISTENT")
        assert stats is None


class TestGetContextForChunk:
    """Tests for get_context_for_chunk()."""

    def test_returns_context(self):
        G = load_tree("1706.03762")
        if G is not None:
            chunk_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "chunk"]
            if chunk_nodes:
                ctx = get_context_for_chunk(G, chunk_nodes[0])
                assert isinstance(ctx, dict)
                assert "paper_title" in ctx
