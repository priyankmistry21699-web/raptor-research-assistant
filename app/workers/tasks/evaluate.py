"""
Evaluation pipeline — Celery task.

Runs RAGAS evaluation asynchronously. Updates EvalRun status on
start, completion, and failure.
"""

import logging
import uuid
from datetime import datetime, timezone

from app.workers.celery_app import celery_app
from app.db.session import SyncSessionLocal
from app.db.models.eval_run import EvalRun

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=60)
def run_evaluation(self, run_id: str):
    """Execute an evaluation run asynchronously."""
    run_uuid = uuid.UUID(run_id)
    session = SyncSessionLocal()

    try:
        run = session.get(EvalRun, run_uuid)
        if not run:
            logger.error("EvalRun %s not found", run_id)
            return

        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        session.commit()

        config = run.config or {}
        eval_type = run.eval_type

        if eval_type == "ragas":
            results = _run_ragas_eval(session, run, config)
        else:
            results = {"error": f"Unsupported eval type: {eval_type}"}

        run.results = results
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        session.commit()
        logger.info("Eval run %s completed", run_id)

    except Exception as exc:
        session.rollback()
        run = session.get(EvalRun, run_uuid)
        if run:
            run.status = "failed"
            run.error_message = str(exc)[:2000]
            run.completed_at = datetime.now(timezone.utc)
            session.commit()
        logger.exception("Eval run %s failed", run_id)
        raise self.retry(exc=exc)
    finally:
        session.close()


def _run_ragas_eval(session, run: EvalRun, config: dict) -> dict:
    """
    Run RAGAS evaluation on a set of QA pairs.
    Evaluates faithfulness, answer relevancy, context precision.
    """
    from app.db.models.chat import ChatMessage, ChatSession
    from sqlalchemy import select

    collection_id = run.collection_id
    sample_limit = config.get("sample_limit", 50)

    # Gather QA pairs from chat sessions for this collection
    query = (
        select(ChatMessage)
        .join(ChatSession, ChatMessage.session_id == ChatSession.id)
        .where(ChatMessage.role == "assistant")
    )
    if collection_id:
        query = query.where(ChatSession.collection_id == collection_id)
    query = query.order_by(ChatMessage.created_at.desc()).limit(sample_limit)

    messages = session.execute(query).scalars().all()

    if not messages:
        return {"status": "no_data", "message": "No chat messages found for evaluation"}

    # Build evaluation dataset
    questions = []
    answers = []
    contexts = []

    for msg in messages:
        # Get the user question (previous message in session)
        user_msg = session.execute(
            select(ChatMessage)
            .where(
                ChatMessage.session_id == msg.session_id,
                ChatMessage.role == "user",
                ChatMessage.created_at < msg.created_at,
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if user_msg:
            questions.append(user_msg.content)
            answers.append(msg.content)
            citations = msg.citations or {}
            ctx_texts = [
                str(v.get("text", v.get("snippet", "")))
                for v in citations.values()
                if isinstance(v, dict)
            ]
            contexts.append(ctx_texts if ctx_texts else [""])

    if not questions:
        return {"status": "no_data", "message": "No valid QA pairs found"}

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
        )
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        return {
            "status": "success",
            "sample_count": len(questions),
            "scores": scores,
        }
    except ImportError:
        return {
            "status": "partial",
            "sample_count": len(questions),
            "message": "RAGAS not installed — returning sample counts only",
        }
    except Exception as e:
        return {
            "status": "error",
            "sample_count": len(questions),
            "message": str(e)[:500],
        }
