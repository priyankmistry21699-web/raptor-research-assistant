"""
FastAPI endpoints for the RAPTOR Retrieval Engine.

Provides:
  POST /retrieve        — hybrid vector search + tree context
  POST /retrieve/tree   — browse a paper by topic/section
  GET  /retrieve/papers — list all available papers
  GET  /retrieve/paper/{arxiv_id} — get paper overview
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.core.retrieval import RaptorRetriever

router = APIRouter(prefix="/retrieve", tags=["retrieval"])

# Singleton retriever — initialized once, reused across requests
_retriever: Optional[RaptorRetriever] = None


def get_retriever() -> RaptorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RaptorRetriever()
    return _retriever


# --- Request/Response models ---

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    arxiv_id: Optional[str] = None
    include_tree_context: bool = True


class TreeRetrieveRequest(BaseModel):
    arxiv_id: str
    topic: Optional[str] = None
    section: Optional[str] = None


class ChunkResult(BaseModel):
    id: str
    text: str
    arxiv_id: str
    chunk_index: int
    distance: float = 0.0
    tree_context: Dict[str, Any] = {}
    context_text: str = ""


# --- Endpoints ---

@router.post("", response_model=List[ChunkResult])
def retrieve(req: RetrieveRequest):
    """
    Hybrid retrieval: vector search + RAPTOR tree traversal.

    1. Embeds the query
    2. Searches ChromaDB for top-k similar chunks
    3. For each chunk, loads the RAPTOR tree and walks UP to get
       section title/summary → topic title/summary → paper title
    4. Returns enriched results ready for prompt construction
    """
    retriever = get_retriever()
    results = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        arxiv_id=req.arxiv_id,
        include_tree_context=req.include_tree_context,
    )
    return [
        ChunkResult(
            id=r.get("id", ""),
            text=r.get("text", ""),
            arxiv_id=r.get("arxiv_id", ""),
            chunk_index=r.get("chunk_index", 0),
            distance=r.get("distance", 0.0),
            tree_context=r.get("tree_context", {}),
            context_text=r.get("context_text", ""),
        )
        for r in results
    ]


@router.post("/tree", response_model=List[Dict[str, Any]])
def retrieve_by_tree(req: TreeRetrieveRequest):
    """
    Tree-based retrieval: browse a paper's chunks by topic or section.
    No embedding needed — walks the RAPTOR tree directly.
    """
    retriever = get_retriever()
    chunks = retriever.retrieve_by_tree(
        arxiv_id=req.arxiv_id,
        topic=req.topic,
        section=req.section,
    )
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for {req.arxiv_id}")
    return chunks


@router.get("/papers", response_model=List[str])
def list_papers():
    """List all arXiv IDs that have RAPTOR trees."""
    return get_retriever().list_available_papers()


@router.get("/paper/{arxiv_id}")
def paper_overview(arxiv_id: str):
    """Get hierarchical overview of a paper (topics → sections → chunk counts)."""
    retriever = get_retriever()
    overview = retriever.get_paper_overview(arxiv_id)
    if overview is None:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
    return overview
