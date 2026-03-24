"""
Evaluation System — RAGAS-based evaluation of RAG pipeline quality.

Metrics (via ragas v0.4.3):
  - Faithfulness        : Is the answer grounded in the retrieved context?
  - Answer Relevancy    : Is the answer relevant to the question?
  - Context Precision   : Are the retrieved chunks relevant? (no reference needed)
  - Factual Correctness : Is the answer factually correct? (needs reference answer)

Supports:
  - Single-query evaluation (evaluate one Q&A pair)
  - Batch evaluation (run a test set across the pipeline)
  - Multi-model comparison (same queries, different models)
  - Persistent result storage (JSONL)

Uses:
  - LiteLLM wrapper for ragas LLM calls (routes through Ollama/Groq)
  - HuggingFace embeddings for ragas embedding calls (all-MiniLM-L6-v2)
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Storage path for evaluation results
EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "evaluation")
EVAL_RESULTS_FILE = os.path.join(EVAL_DIR, "eval_results.jsonl")


def _get_ragas_llm(model: str = "ollama/mistral:latest"):
    """
    Create a ragas-compatible LLM wrapper using LiteLLM.

    Model format for LiteLLM:
      - Ollama: "ollama/mistral:latest" (needs OLLAMA_API_BASE env var)
      - Groq:   "groq/llama-3.3-70b-versatile" (needs GROQ_API_KEY env var)
    """
    from ragas.llms.litellm_llm import LiteLLMStructuredLLM

    # Set Ollama base URL for LiteLLM if using Ollama
    if model.startswith("ollama/"):
        os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11435")

    return LiteLLMStructuredLLM(model=model)


def _get_ragas_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Create a ragas-compatible embedding model."""
    from ragas.embeddings import HuggingfaceEmbeddings

    return HuggingfaceEmbeddings(model_name=model_name)


def _get_metrics(metric_names: Optional[List[str]] = None):
    """
    Load ragas metric instances by name.

    Available metrics:
      - faithfulness
      - answer_relevancy
      - context_precision  (no reference needed)
      - factual_correctness (needs reference answer)
    """
    from ragas.metrics.collections.faithfulness import Faithfulness
    from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
    from ragas.metrics.collections.context_precision import (
        ContextPrecisionWithoutReference,
    )
    from ragas.metrics.collections.factual_correctness import FactualCorrectness

    available = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecisionWithoutReference(),
        "factual_correctness": FactualCorrectness(),
    }

    if metric_names is None:
        # Default: metrics that don't require a reference answer
        metric_names = ["faithfulness", "answer_relevancy", "context_precision"]

    selected = []
    for name in metric_names:
        if name in available:
            selected.append(available[name])
        else:
            logger.warning(
                f"Unknown metric '{name}', skipping. Available: {list(available.keys())}"
            )

    return selected


def _append_result(result: Dict[str, Any]):
    """Append an evaluation result to the JSONL file."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(EVAL_RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def evaluate_single(
    question: str,
    answer: str,
    contexts: List[str],
    reference: Optional[str] = None,
    metric_names: Optional[List[str]] = None,
    llm_model: str = "ollama/mistral:latest",
) -> Dict[str, Any]:
    """
    Evaluate a single Q&A pair against RAGAS metrics.

    Args:
        question: The user question
        answer: The LLM-generated answer
        contexts: List of retrieved context strings
        reference: Optional ground-truth answer (needed for factual_correctness)
        metric_names: Which metrics to run (default: faithfulness, answer_relevancy, context_precision)
        llm_model: LiteLLM model string for ragas judge LLM

    Returns:
        Dict with metric scores and metadata
    """
    from ragas import SingleTurnSample, EvaluationDataset, evaluate

    # If factual_correctness requested but no reference, remove it
    if metric_names and "factual_correctness" in metric_names and not reference:
        logger.warning(
            "factual_correctness requires a reference answer, removing from metrics"
        )
        metric_names = [m for m in metric_names if m != "factual_correctness"]

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference or "",
    )

    dataset = EvaluationDataset(samples=[sample])
    metrics = _get_metrics(metric_names)

    if not metrics:
        return {"status": "error", "error": "No valid metrics selected"}

    ragas_llm = _get_ragas_llm(llm_model)
    ragas_embeddings = _get_ragas_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        show_progress=False,
    )

    # Extract scores
    scores = {}
    result_df = result.to_pandas()
    for col in result_df.columns:
        if col not in ("user_input", "response", "retrieved_contexts", "reference"):
            val = result_df[col].iloc[0]
            # Convert numpy types to Python native
            scores[col] = float(val) if val is not None else None

    eval_record = {
        "question": question,
        "answer": answer[:500],
        "num_contexts": len(contexts),
        "reference_provided": reference is not None,
        "scores": scores,
        "llm_model": llm_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_result(eval_record)

    return eval_record


def evaluate_batch(
    samples: List[Dict[str, Any]],
    metric_names: Optional[List[str]] = None,
    llm_model: str = "ollama/mistral:latest",
) -> Dict[str, Any]:
    """
    Evaluate a batch of Q&A samples.

    Each sample dict should have:
      - question: str
      - answer: str
      - contexts: List[str]
      - reference: Optional[str]

    Returns:
        Dict with per-sample scores and aggregate statistics
    """
    from ragas import SingleTurnSample, EvaluationDataset, evaluate

    has_reference = all(s.get("reference") for s in samples)
    if metric_names and "factual_correctness" in metric_names and not has_reference:
        metric_names = [m for m in metric_names if m != "factual_correctness"]

    ragas_samples = []
    for s in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s.get("contexts", []),
                reference=s.get("reference", ""),
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)
    metrics = _get_metrics(metric_names)

    if not metrics:
        return {"status": "error", "error": "No valid metrics selected"}

    ragas_llm = _get_ragas_llm(llm_model)
    ragas_embeddings = _get_ragas_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        show_progress=False,
    )

    result_df = result.to_pandas()

    # Per-sample scores
    per_sample = []
    metric_cols = [
        c
        for c in result_df.columns
        if c not in ("user_input", "response", "retrieved_contexts", "reference")
    ]
    for idx, row in result_df.iterrows():
        sample_scores = {}
        for col in metric_cols:
            val = row[col]
            sample_scores[col] = float(val) if val is not None else None
        per_sample.append(
            {
                "question": samples[idx]["question"],
                "scores": sample_scores,
            }
        )

    # Aggregate statistics
    aggregates = {}
    for col in metric_cols:
        vals = [float(v) for v in result_df[col].dropna()]
        if vals:
            aggregates[col] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "count": len(vals),
            }

    batch_record = {
        "type": "batch",
        "num_samples": len(samples),
        "aggregates": aggregates,
        "llm_model": llm_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_result(batch_record)

    return {
        "status": "success",
        "num_samples": len(samples),
        "per_sample": per_sample,
        "aggregates": aggregates,
        "llm_model": llm_model,
        "timestamp": batch_record["timestamp"],
    }


def evaluate_pipeline(
    queries: List[str],
    model: str = "mistral",
    top_k: int = 5,
    task: str = "qa",
    metric_names: Optional[List[str]] = None,
    llm_model: str = "ollama/mistral:latest",
    references: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline evaluation: run queries through the full RAG pipeline
    (retrieve → prompt → LLM → evaluate).

    Args:
        queries: List of test questions
        model: LLM model alias for answer generation ('mistral', 'groq-llama')
        top_k: Number of chunks to retrieve
        task: Task type for prompt construction
        metric_names: RAGAS metrics to compute
        llm_model: LiteLLM model string for the RAGAS judge
        references: Optional ground-truth answers (one per query)

    Returns:
        Full evaluation results with per-query scores and aggregates
    """
    from app.core.retrieval import RaptorRetriever
    from app.core.prompt_builder import build_messages
    from app.core.llm_client import run_llm_messages

    retriever = RaptorRetriever()
    samples = []

    for i, query in enumerate(queries):
        # Step 1: Retrieve
        results = retriever.retrieve(
            query=query, top_k=top_k, include_tree_context=True
        )
        contexts = [r.get("text", "") for r in results]

        # Step 2: Build prompt
        chunks = []
        for r in results:
            ctx = r.get("tree_context", {})
            chunks.append(
                {
                    "arxiv_id": r.get("arxiv_id", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "chunk_text": r.get("text", ""),
                    "section_num": ctx.get("section_num", ""),
                    "section_title": ctx.get("section_title", ""),
                    "section_summary": ctx.get("section_summary", ""),
                    "topic": ctx.get("topic", ""),
                    "topic_summary": ctx.get("topic_summary", ""),
                    "paper_title": ctx.get("paper_title", ""),
                }
            )

        messages = build_messages(chunks, query, task=task)

        # Step 3: Generate answer
        answer = run_llm_messages(messages, model=model, task=task)

        sample = {
            "question": query,
            "answer": answer,
            "contexts": contexts,
        }
        if references and i < len(references):
            sample["reference"] = references[i]

        samples.append(sample)
        logger.info(f"Pipeline eval [{i + 1}/{len(queries)}]: {query[:60]}...")

    # Step 4: Evaluate all samples
    return evaluate_batch(
        samples=samples,
        metric_names=metric_names,
        llm_model=llm_model,
    )


def compare_models(
    queries: List[str],
    models: List[str],
    top_k: int = 5,
    task: str = "qa",
    metric_names: Optional[List[str]] = None,
    llm_model: str = "ollama/mistral:latest",
) -> Dict[str, Any]:
    """
    Run the same queries across multiple models and compare RAGAS scores.

    Args:
        queries: Test questions
        models: List of model aliases to compare (e.g., ['mistral', 'groq-llama'])
        top_k: Chunks per retrieval
        task: Task type
        metric_names: Which RAGAS metrics to compute
        llm_model: LiteLLM model string for the RAGAS judge

    Returns:
        Per-model results and comparison summary
    """
    results_by_model = {}
    for model_alias in models:
        logger.info(f"Evaluating model: {model_alias}")
        result = evaluate_pipeline(
            queries=queries,
            model=model_alias,
            top_k=top_k,
            task=task,
            metric_names=metric_names,
            llm_model=llm_model,
        )
        results_by_model[model_alias] = result

    # Build comparison summary
    comparison = {}
    for model_alias, result in results_by_model.items():
        agg = result.get("aggregates", {})
        comparison[model_alias] = {
            metric: stats.get("mean") for metric, stats in agg.items()
        }

    summary = {
        "status": "success",
        "queries": queries,
        "models_evaluated": models,
        "results_by_model": results_by_model,
        "comparison": comparison,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_result({"type": "comparison", **summary})

    return summary


def get_eval_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent evaluation results."""
    if not os.path.exists(EVAL_RESULTS_FILE):
        return []
    entries = []
    with open(EVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries[-limit:]


def get_eval_stats() -> Dict[str, Any]:
    """Get aggregate statistics across all evaluations."""
    entries = get_eval_history(limit=10000)
    if not entries:
        return {"total_evaluations": 0}

    single_evals = [e for e in entries if "scores" in e]
    batch_evals = [e for e in entries if e.get("type") == "batch"]

    # Aggregate all single-eval scores
    all_scores: Dict[str, List[float]] = {}
    for e in single_evals:
        for metric, val in e.get("scores", {}).items():
            if val is not None:
                all_scores.setdefault(metric, []).append(val)

    metric_averages = {}
    for metric, vals in all_scores.items():
        metric_averages[metric] = {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
        }

    return {
        "total_evaluations": len(entries),
        "single_evaluations": len(single_evals),
        "batch_evaluations": len(batch_evals),
        "metric_averages": metric_averages,
    }
