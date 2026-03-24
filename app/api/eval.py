"""
Evaluation API — Endpoints for RAGAS-based RAG quality evaluation.

Endpoints:
  POST /eval/single          — Evaluate a single Q&A pair
  POST /eval/batch           — Evaluate a batch of Q&A samples
  POST /eval/pipeline        — End-to-end: run queries through RAG pipeline + evaluate
  POST /eval/compare         — Compare multiple models on the same queries
  GET  /eval/history         — Get recent evaluation results
  GET  /eval/stats           — Get aggregate evaluation statistics
"""

import os
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.core.evaluation import (
    evaluate_single,
    evaluate_batch,
    evaluate_pipeline,
    compare_models,
    get_eval_history,
    get_eval_stats,
)

router = APIRouter(prefix="/eval", tags=["evaluation"])


# --- Request / Response models ---


class EvalSingleRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    reference: Optional[str] = None
    metric_names: Optional[List[str]] = None
    llm_model: str = "ollama/mistral:latest"


class EvalBatchRequest(BaseModel):
    samples: List[Dict[str, Any]]
    metric_names: Optional[List[str]] = None
    llm_model: str = "ollama/mistral:latest"


class EvalPipelineRequest(BaseModel):
    queries: List[str]
    model: str = "mistral"
    top_k: int = 5
    task: str = "qa"
    metric_names: Optional[List[str]] = None
    llm_model: str = "ollama/mistral:latest"
    references: Optional[List[str]] = None


class CompareRequest(BaseModel):
    queries: List[str]
    models: List[str]
    top_k: int = 5
    task: str = "qa"
    metric_names: Optional[List[str]] = None
    llm_model: str = "ollama/mistral:latest"


# --- Endpoints ---


@router.post("/single")
def eval_single(req: EvalSingleRequest):
    """
    Evaluate a single Q&A pair against RAGAS metrics.

    Metrics (default: faithfulness, answer_relevancy, context_precision):
      - faithfulness: Is the answer grounded in the context?
      - answer_relevancy: Is the answer relevant to the question?
      - context_precision: Are the retrieved contexts relevant?
      - factual_correctness: Is the answer factually correct? (needs reference)
    """
    if not req.contexts:
        raise HTTPException(
            status_code=400, detail="At least one context string is required"
        )

    result = evaluate_single(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        reference=req.reference,
        metric_names=req.metric_names,
        llm_model=req.llm_model,
    )
    return result


@router.post("/batch")
def eval_batch(req: EvalBatchRequest):
    """
    Evaluate a batch of Q&A samples.

    Each sample in the list should have: question, answer, contexts, and optionally reference.
    Returns per-sample scores and aggregate statistics.
    """
    if not req.samples:
        raise HTTPException(status_code=400, detail="At least one sample is required")

    for i, s in enumerate(req.samples):
        if "question" not in s or "answer" not in s:
            raise HTTPException(
                status_code=400,
                detail=f"Sample {i} missing required fields: question, answer",
            )
        if "contexts" not in s or not s["contexts"]:
            raise HTTPException(status_code=400, detail=f"Sample {i} missing contexts")

    result = evaluate_batch(
        samples=req.samples,
        metric_names=req.metric_names,
        llm_model=req.llm_model,
    )
    return result


@router.post("/pipeline")
def eval_pipeline(req: EvalPipelineRequest):
    """
    End-to-end pipeline evaluation.

    Runs each query through the full RAG pipeline (retrieve → prompt → LLM)
    and then evaluates the results with RAGAS metrics.

    This tests the entire system, not just individual components.
    """
    if not req.queries:
        raise HTTPException(status_code=400, detail="At least one query is required")

    result = evaluate_pipeline(
        queries=req.queries,
        model=req.model,
        top_k=req.top_k,
        task=req.task,
        metric_names=req.metric_names,
        llm_model=req.llm_model,
        references=req.references,
    )
    return result


@router.post("/compare")
def eval_compare(req: CompareRequest):
    """
    Compare multiple models on the same set of queries.

    Runs each query through the RAG pipeline with each model,
    evaluates with RAGAS, and returns a side-by-side comparison.
    """
    if not req.queries:
        raise HTTPException(status_code=400, detail="At least one query is required")
    if len(req.models) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 models are required for comparison"
        )

    result = compare_models(
        queries=req.queries,
        models=req.models,
        top_k=req.top_k,
        task=req.task,
        metric_names=req.metric_names,
        llm_model=req.llm_model,
    )
    return result


@router.get("/history")
def eval_history(limit: int = 50):
    """Get recent evaluation results."""
    return get_eval_history(limit=limit)


@router.get("/stats")
def eval_stats():
    """Get aggregate statistics across all evaluations."""
    return get_eval_stats()
