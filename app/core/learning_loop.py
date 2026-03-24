"""
Continuous Learning Loop — Automates the full improvement cycle.

Flow:
  User question → retrieval → LLM response → feedback
  → preference dataset → DPO fine-tuning → register improved model → use for next query

This module orchestrates:
  1. Monitoring feedback volume (enough data to trigger training?)
  2. Building preference dataset from accumulated feedback
  3. Launching DPO fine-tuning
  4. Registering the new model in MODEL_REGISTRY
  5. Selecting the best model for inference (base vs fine-tuned)
  6. Tracking loop history (when training ran, what improved)

Can run automatically (background thread) or be triggered manually via API.
"""

import json
import os
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from app.core.feedback import feedback_store
from app.core.preference import preference_store
from app.core.finetune import (
    run_dpo_training,
    get_training_status,
    list_finetuned_models,
    register_finetuned_model,
    DEFAULT_BASE_MODEL,
)
from app.core.llm_client import MODEL_REGISTRY, get_active_model

logger = logging.getLogger(__name__)

# --- Loop state ---
_loop_lock = threading.Lock()
_loop_state: Dict[str, Any] = {
    "auto_enabled": False,
    "min_new_feedback": 10,  # Min new feedback entries before auto-trigger
    "check_interval_seconds": 300,  # How often to check (5 min)
    "last_feedback_count": 0,  # Feedback count at last training trigger
    "last_triggered_at": None,
    "trigger_count": 0,
    "current_run": None,
    "active_model": None,  # Currently selected best model alias
}

# History of learning loop runs
LOOP_HISTORY_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "feedback", "loop_history.jsonl"
)

_auto_thread: Optional[threading.Thread] = None
_auto_stop_event = threading.Event()


def get_loop_status() -> Dict[str, Any]:
    """Get current state of the learning loop."""
    with _loop_lock:
        status = dict(_loop_state)
    status["training_status"] = get_training_status()
    status["active_model"] = get_active_model()
    status["feedback_count"] = feedback_store.count()
    status["preference_count"] = preference_store.count()
    status["finetuned_models"] = len(list_finetuned_models())
    return status


def get_loop_history() -> List[Dict[str, Any]]:
    """Get history of all learning loop runs."""
    if not os.path.exists(LOOP_HISTORY_FILE):
        return []
    entries = []
    with open(LOOP_HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _append_history(entry: Dict[str, Any]):
    """Append a loop run record to history."""
    os.makedirs(os.path.dirname(LOOP_HISTORY_FILE), exist_ok=True)
    with open(LOOP_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def trigger_learning_loop(
    base_model: str = DEFAULT_BASE_MODEL,
    num_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Manually trigger one full learning loop cycle:
      1. Check if we have enough new feedback
      2. Build preference dataset from all feedback
      3. Run DPO fine-tuning
      4. Register the new model
      5. Record in history

    Args:
        base_model: HuggingFace model ID for base model
        num_epochs: Training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        force: If True, skip the min_new_feedback check

    Returns:
        Dict with results of each step
    """
    started_at = datetime.now(timezone.utc).isoformat()

    # Check training isn't already running
    training_status = get_training_status()
    if training_status["running"]:
        return {
            "status": "skipped",
            "reason": f"Training already in progress: {training_status['run_name']}",
            "timestamp": started_at,
        }

    # Step 1: Check feedback volume
    current_count = feedback_store.count()
    with _loop_lock:
        last_count = _loop_state["last_feedback_count"]
        min_new = _loop_state["min_new_feedback"]

    new_feedback = current_count - last_count
    if not force and new_feedback < min_new:
        return {
            "status": "skipped",
            "reason": f"Not enough new feedback: {new_feedback}/{min_new} (use force=true to override)",
            "feedback_total": current_count,
            "new_since_last": new_feedback,
            "timestamp": started_at,
        }

    # Step 2: Build preference dataset
    logger.info(
        f"Learning loop: building preference dataset ({current_count} feedback entries)"
    )
    build_result = preference_store.build_from_feedback()
    pairs_count = build_result["pairs_created"]

    if pairs_count < 2:
        result = {
            "status": "skipped",
            "reason": f"Not enough preference pairs: {pairs_count}/2 minimum",
            "build_result": build_result,
            "timestamp": started_at,
        }
        _append_history(result)
        return result

    # Step 3: Run DPO training
    logger.info(f"Learning loop: starting DPO training with {pairs_count} pairs")
    pairs = preference_store.export_for_training()

    with _loop_lock:
        _loop_state["current_run"] = "training"

    train_result = run_dpo_training(
        preference_pairs=pairs,
        base_model=base_model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    if train_result.get("status") != "success":
        result = {
            "status": "failed",
            "reason": f"Training failed: {train_result.get('error', 'unknown')}",
            "build_result": build_result,
            "train_result": train_result,
            "timestamp": started_at,
        }
        with _loop_lock:
            _loop_state["current_run"] = None
        _append_history(result)
        return result

    # Step 4: Register the new model
    run_name = train_result["run_name"]
    alias = f"finetuned-{run_name}"
    register_result = register_finetuned_model(run_name, alias)

    logger.info(f"Learning loop: registered model as '{alias}'")

    # Step 5: Update loop state
    completed_at = datetime.now(timezone.utc).isoformat()
    with _loop_lock:
        _loop_state["last_feedback_count"] = current_count
        _loop_state["last_triggered_at"] = completed_at
        _loop_state["trigger_count"] += 1
        _loop_state["current_run"] = None
        _loop_state["active_model"] = alias

    result = {
        "status": "success",
        "started_at": started_at,
        "completed_at": completed_at,
        "feedback_processed": current_count,
        "pairs_created": pairs_count,
        "train_loss": train_result.get("train_loss"),
        "run_name": run_name,
        "model_alias": alias,
        "build_result": build_result,
        "train_result": train_result,
        "register_result": register_result,
    }
    _append_history(result)
    return result


def _auto_loop_worker():
    """Background worker that periodically checks if training should be triggered."""
    logger.info("Learning loop: auto-mode started")
    while not _auto_stop_event.is_set():
        with _loop_lock:
            interval = _loop_state["check_interval_seconds"]

        _auto_stop_event.wait(timeout=interval)
        if _auto_stop_event.is_set():
            break

        with _loop_lock:
            if not _loop_state["auto_enabled"]:
                continue

        try:
            result = trigger_learning_loop()
            if result["status"] == "success":
                logger.info(
                    f"Learning loop auto-trigger succeeded: {result['model_alias']}"
                )
            elif result["status"] == "skipped":
                logger.debug(f"Learning loop auto-check: {result['reason']}")
        except Exception as e:
            logger.error(f"Learning loop auto-trigger error: {e}", exc_info=True)

    logger.info("Learning loop: auto-mode stopped")


def enable_auto_loop(
    min_new_feedback: int = 10,
    check_interval_seconds: int = 300,
) -> Dict[str, Any]:
    """
    Enable automatic learning loop. A background thread will periodically
    check if enough new feedback has accumulated and trigger training.

    Args:
        min_new_feedback: Number of new feedback entries needed to trigger training
        check_interval_seconds: How often to check (in seconds)
    """
    global _auto_thread

    with _loop_lock:
        _loop_state["auto_enabled"] = True
        _loop_state["min_new_feedback"] = min_new_feedback
        _loop_state["check_interval_seconds"] = check_interval_seconds

    # Start background thread if not already running
    if _auto_thread is None or not _auto_thread.is_alive():
        _auto_stop_event.clear()
        _auto_thread = threading.Thread(target=_auto_loop_worker, daemon=True)
        _auto_thread.start()

    return {
        "status": "enabled",
        "min_new_feedback": min_new_feedback,
        "check_interval_seconds": check_interval_seconds,
    }


def disable_auto_loop() -> Dict[str, Any]:
    """Disable automatic learning loop."""
    with _loop_lock:
        _loop_state["auto_enabled"] = False

    _auto_stop_event.set()

    return {"status": "disabled"}


def configure_loop(
    min_new_feedback: Optional[int] = None,
    check_interval_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Update learning loop configuration without enabling/disabling."""
    with _loop_lock:
        if min_new_feedback is not None:
            _loop_state["min_new_feedback"] = min_new_feedback
        if check_interval_seconds is not None:
            _loop_state["check_interval_seconds"] = check_interval_seconds
        return {
            "min_new_feedback": _loop_state["min_new_feedback"],
            "check_interval_seconds": _loop_state["check_interval_seconds"],
            "auto_enabled": _loop_state["auto_enabled"],
        }


def select_best_model() -> Dict[str, str]:
    """
    Select the best available model for inference.

    Strategy:
      - If a fine-tuned model is registered, prefer it (most recent)
      - Otherwise fall back to the default base model ('mistral')

    Returns the model alias and metadata.
    """
    finetuned = [(k, v) for k, v in MODEL_REGISTRY.items() if v.get("is_finetuned")]

    if finetuned:
        alias, config = finetuned[-1]  # Most recently added
        return {
            "model": alias,
            "type": "finetuned",
            "base_model": config.get("base_model", ""),
            "adapter_path": config.get("adapter_path", ""),
        }

    return {
        "model": "mistral",
        "type": "base",
        "base_model": "mistral:latest",
        "adapter_path": "",
    }
