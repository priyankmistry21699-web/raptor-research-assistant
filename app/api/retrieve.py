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
        raise HTTPException(
            status_code=404, detail=f"No chunks found for {req.arxiv_id}"
        )
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


# --- Paper-Specific Learning & Debate endpoints ---


class PaperSpecificQuery(BaseModel):
    query: str
    arxiv_id: str
    top_k: int = 10
    include_debate_context: bool = True


class FineTunePaperRequest(BaseModel):
    arxiv_id: str
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4


@router.post("/paper-specific-query")
def paper_specific_query(req: PaperSpecificQuery):
    """
    Query a specific paper with isolated context.
    Only retrieves chunks from the specified paper, no cross-paper contamination.
    """
    retriever = get_retriever()

    # Force retrieval to only this paper
    results = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        arxiv_id=req.arxiv_id,  # This isolates to the specific paper
        include_tree_context=True,
    )

    if req.include_debate_context:
        # Add debate context - show contrasting views or alternative interpretations
        debate_context = _generate_debate_context(req.query, results)
        return {
            "paper_isolated_results": results,
            "debate_context": debate_context,
            "paper_id": req.arxiv_id,
        }

    return {"results": results, "paper_id": req.arxiv_id}


@router.post("/fine-tune-paper")
def fine_tune_paper(req: FineTunePaperRequest):
    """
    Fine-tune a model specifically on one paper's content.
    Creates a paper-specific model for deeper understanding.
    """
    from app.core.finetune import fine_tune_on_paper

    try:
        result = fine_tune_on_paper(
            arxiv_id=req.arxiv_id,
            learning_rate=req.learning_rate,
            num_epochs=req.num_epochs,
            batch_size=req.batch_size,
        )
        return {
            "status": "success",
            "paper_id": req.arxiv_id,
            "model_path": result.get("model_path"),
            "training_stats": result.get("stats"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")


@router.get("/paper-models/{arxiv_id}")
def get_paper_models(arxiv_id: str):
    """List available fine-tuned models for a specific paper."""
    from app.core.llm_client import list_available_models

    models = list_available_models()
    paper_models = {}

    for model_name, model_info in models.items():
        if f"paper_{arxiv_id}" in model_name:
            paper_models[model_name] = model_info

    return {"paper_id": arxiv_id, "available_models": paper_models}


def _generate_debate_context(query: str, results: List[Dict]) -> Dict:
    """
    Generate debate context by finding contrasting or alternative viewpoints
    within the paper's content.
    """
    debate_points = []

    # Look for contrasting language in results
    for result in results:
        text = result.get("text", "").lower()
        if any(
            word in text
            for word in ["however", "but", "although", "while", "unlike", "contrast"]
        ):
            debate_points.append(
                {
                    "type": "contrast",
                    "text": result.get("text", "")[:200] + "...",
                    "chunk_id": result.get("id", ""),
                }
            )
        elif any(
            word in text
            for word in [
                "alternative",
                "another approach",
                "different method",
                "instead",
            ]
        ):
            debate_points.append(
                {
                    "type": "alternative",
                    "text": result.get("text", "")[:200] + "...",
                    "chunk_id": result.get("id", ""),
                }
            )

    return {
        "debate_points": debate_points[:5],  # Limit to top 5
        "total_points": len(debate_points),
    }
