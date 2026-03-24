"""
RAPTOR Retrieval Engine — Hybrid retrieval combining vector search + tree traversal.

Strategy:
1. Vector search → find top-k similar chunks from Chroma.
2. For each result, load the paper's RAPTOR tree.
3. Walk UP the tree from each chunk to gather hierarchical context
   (section title/summary → topic title/summary → paper title).
4. Return enriched results with full context for prompt construction.
"""

import os
from typing import List, Dict, Any, Optional
from app.core.embedding import EmbeddingModel
from app.core.vector_db import VectorDB
from app.core.raptor_index import (
    load_tree,
    get_context_for_chunk,
    get_tree_structure,
    get_chunks,
)


class RaptorRetriever:
    """Hybrid retrieval: vector search + RAPTOR tree context."""

    def __init__(
        self, chroma_dir: str = None, embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.embedder = EmbeddingModel(model_name=embedding_model)
        self.vector_db = VectorDB(chroma_dir=chroma_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        arxiv_id: Optional[str] = None,
        include_tree_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method.
        Returns top-k chunks enriched with hierarchical context from RAPTOR trees.

        Args:
            query: User query text.
            top_k: Number of results to return.
            arxiv_id: If set, restrict search to a single paper.
            include_tree_context: If True, enrich results with tree context.
        """
        query_embedding = self.embedder.encode(query)

        if arxiv_id:
            raw_results = self.vector_db.search_by_paper(
                query_embedding, arxiv_id, top_k
            )
        else:
            raw_results = self.vector_db.search(query_embedding, top_k)

        if not include_tree_context:
            return raw_results

        # Enrich with RAPTOR tree context
        enriched = []
        tree_cache = {}  # Cache loaded trees

        for result in raw_results:
            paper_id = result["arxiv_id"]
            chunk_idx = result["chunk_index"]
            chunk_node = f"chunk_{chunk_idx}"

            # Load tree (cached)
            if paper_id not in tree_cache:
                tree_cache[paper_id] = load_tree(paper_id)

            G = tree_cache[paper_id]
            tree_context = {}

            if G is not None and chunk_node in G.nodes:
                tree_context = get_context_for_chunk(G, chunk_node)

            enriched.append(
                {
                    **result,
                    "tree_context": tree_context,
                    "context_text": _build_context_text(result, tree_context),
                }
            )

        return enriched

    def retrieve_by_tree(
        self,
        arxiv_id: str,
        topic: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tree-based retrieval: walk the tree to get chunks by topic/section.
        Useful for browsing a paper's structure.
        """
        G = load_tree(arxiv_id)
        if G is None:
            return []

        if section:
            # Find matching section node
            for node in G.nodes:
                n = G.nodes[node]
                if n.get("type") == "section":
                    if (
                        n.get("section_num") == section
                        or section.lower() in n.get("title", "").lower()
                    ):
                        return get_chunks(G, node)

        if topic:
            # Find matching topic node
            for node in G.nodes:
                n = G.nodes[node]
                if n.get("type") == "topic":
                    if topic.lower() in n.get("title", "").lower():
                        return get_chunks(G, node)

        # Return all chunks
        return get_chunks(G, "root")

    def get_paper_overview(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get hierarchical overview of a paper."""
        return get_tree_structure(arxiv_id)

    def list_available_papers(self) -> List[str]:
        """List all arXiv IDs that have RAPTOR trees."""
        from app.core.raptor_index import TREE_DIR

        papers = []
        if os.path.isdir(TREE_DIR):
            for f in os.listdir(TREE_DIR):
                if f.endswith("_tree.gpickle"):
                    papers.append(f.replace("_tree.gpickle", ""))
        return sorted(papers)


def _build_context_text(result: Dict, tree_context: Dict) -> str:
    """
    Build a formatted context string for prompt construction.
    Combines hierarchical context with chunk text.
    """
    parts = []

    paper_title = tree_context.get("paper_title", "")
    if paper_title:
        parts.append(f"Paper: {paper_title}")

    topic = tree_context.get("topic", "")
    if topic:
        parts.append(f"Topic: {topic}")

    topic_summary = tree_context.get("topic_summary", "")
    if topic_summary:
        parts.append(f"Topic Summary: {topic_summary}")

    section_title = tree_context.get("section_title", "")
    section_num = tree_context.get("section_num", "")
    if section_title:
        sec_label = (
            f"Section {section_num}: {section_title}"
            if section_num
            else f"Section: {section_title}"
        )
        parts.append(sec_label)

    section_summary = tree_context.get("section_summary", "")
    if section_summary:
        parts.append(f"Section Summary: {section_summary}")

    parts.append(f"\nContent:\n{result.get('text', '')}")

    return "\n".join(parts)
