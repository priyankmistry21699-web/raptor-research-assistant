"""
Training API — Endpoints for preference dataset, DPO fine-tuning, and continuous learning loop.

Endpoints:
  POST /train/preferences/build    — Build preference dataset from all feedback
  GET  /train/preferences           — Get all preference pairs
  GET  /train/preferences/stats     — Get preference dataset statistics
  GET  /train/preferences/export    — Export clean (prompt, chosen, rejected) for DPO training
  POST /train/finetune              — Start DPO fine-tuning run
  GET  /train/finetune/status       — Check training progress
  GET  /train/finetune/models       — List all saved fine-tuned models
  POST /train/finetune/register     — Register fine-tuned model for inference
  POST /train/loop/trigger          — Manually trigger one learning loop cycle
  GET  /train/loop/status           — Get learning loop state
  POST /train/loop/auto             — Enable/disable automatic learning loop
  GET  /train/loop/history          — Get history of all loop runs
  GET  /train/loop/model            — Get the currently selected best model
  PUT  /train/loop/config           — Update learning loop configuration
"""
import os
import sys
import threading

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.preference import preference_store
from app.core.finetune import (
    run_dpo_training,
    get_training_status,
    list_finetuned_models,
    register_finetuned_model,
    DEFAULT_BASE_MODEL,
)
from app.core.learning_loop import (
    get_loop_status,
    get_loop_history,
    trigger_learning_loop,
    enable_auto_loop,
    disable_auto_loop,
    configure_loop,
    select_best_model,
)

router = APIRouter(prefix="/train", tags=["training"])


# --- Response models ---

class BuildResult(BaseModel):
    total_feedback: int
    pairs_created: int
    skipped: int
    output_file: str
    built_at: str

class PreferenceStats(BaseModel):
    total_pairs: int
    by_feedback_type: Dict[str, int]
    has_real_rejected: int
    output_file: str

class PreferencePair(BaseModel):
    prompt: str
    chosen: str
    rejected: str


# --- Endpoints ---

@router.post("/preferences/build", response_model=BuildResult)
def build_preferences():
    """
    Build (or rebuild) the preference dataset from all collected feedback.

    Converts each feedback entry into a (prompt, chosen, rejected) triple:
      - helpful → answer is chosen
      - correction → correction is chosen, original is rejected
      - incorrect/hallucination with correction → correction chosen, original rejected
      - incorrect/hallucination without correction → skipped
    """
    result = preference_store.build_from_feedback()
    return BuildResult(**result)


@router.get("/preferences", response_model=List[Dict[str, Any]])
def get_preferences():
    """Get all preference pairs (with metadata)."""
    return preference_store.get_all()


@router.get("/preferences/stats", response_model=PreferenceStats)
def get_preference_stats():
    """Get statistics about the current preference dataset."""
    stats = preference_store.get_stats()
    return PreferenceStats(**stats)


@router.get("/preferences/export", response_model=List[PreferencePair])
def export_preferences():
    """
    Export preference pairs in clean DPO training format.

    Returns list of {prompt, chosen, rejected} dicts — ready for TRL DPOTrainer.
    """
    pairs = preference_store.export_for_training()
    if not pairs:
        raise HTTPException(
            status_code=404,
            detail="No preference pairs available. Submit feedback first, then POST /train/preferences/build."
        )
    return [PreferencePair(**p) for p in pairs]


# =============================================
# Fine-tuning endpoints (Section 12)
# =============================================

class FinetuneRequest(BaseModel):
    base_model: str = DEFAULT_BASE_MODEL
    run_name: Optional[str] = None
    num_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 512
    max_prompt_length: int = 256
    beta: float = 0.1
    gradient_accumulation_steps: int = 4

class FinetuneResponse(BaseModel):
    status: str
    run_name: Optional[str] = None
    message: str

class TrainingStatus(BaseModel):
    running: bool
    run_name: Optional[str] = None
    progress: Optional[str] = None
    error: Optional[str] = None
    last_completed: Optional[Dict[str, Any]] = None

class RegisterRequest(BaseModel):
    run_name: str
    alias: Optional[str] = None


def _run_training_background(pairs, request):
    """Background worker for DPO training."""
    run_dpo_training(
        preference_pairs=pairs,
        base_model=request.base_model,
        run_name=request.run_name,
        num_epochs=request.num_epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        lora_r=request.lora_r,
        lora_alpha=request.lora_alpha,
        lora_dropout=request.lora_dropout,
        max_length=request.max_length,
        max_prompt_length=request.max_prompt_length,
        beta=request.beta,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
    )


@router.post("/finetune", response_model=FinetuneResponse)
def start_finetune(req: FinetuneRequest, background_tasks: BackgroundTasks):
    """
    Start a DPO fine-tuning run using the preference dataset.

    Training runs in the background. Check progress with GET /train/finetune/status.

    Uses LoRA adapters via PEFT for parameter-efficient training.
    Saves adapter weights to models/<run_name>/.
    """
    # Check if training is already running
    status = get_training_status()
    if status["running"]:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress: {status['run_name']}"
        )

    # Get preference pairs
    pairs = preference_store.export_for_training()
    if len(pairs) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 preference pairs for DPO training. "
                   f"Currently have {len(pairs)}. Submit more feedback first."
        )

    # Launch training in background
    background_tasks.add_task(_run_training_background, pairs, req)

    return FinetuneResponse(
        status="started",
        run_name=req.run_name,
        message=f"DPO training started with {len(pairs)} preference pairs. "
                f"Base model: {req.base_model}. Check /train/finetune/status for progress.",
    )


@router.get("/finetune/status", response_model=TrainingStatus)
def finetune_status():
    """Check the status of the current or last training run."""
    status = get_training_status()
    return TrainingStatus(**status)


@router.get("/finetune/models", response_model=List[Dict[str, Any]])
def get_finetuned_models():
    """List all saved fine-tuned model adapters in models/ directory."""
    return list_finetuned_models()


@router.post("/finetune/register")
def register_model(req: RegisterRequest):
    """
    Register a fine-tuned model adapter for use in inference.

    After registration, the model can be used via POST /llm with model=<alias>.
    """
    result = register_finetuned_model(req.run_name, req.alias)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# =============================================
# Paper-Specific Training endpoints (Step 18)
# =============================================

class PaperFinetuneRequest(BaseModel):
    arxiv_id: str
    base_model: str = DEFAULT_BASE_MODEL
    run_name: Optional[str] = None
    num_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 512
    max_prompt_length: int = 256
    beta: float = 0.1
    gradient_accumulation_steps: int = 4

class PaperFinetuneResponse(BaseModel):
    status: str
    run_name: Optional[str] = None
    arxiv_id: str
    message: str


def _run_paper_training_background(arxiv_id, request):
    """Background worker for paper-specific DPO training."""
    from app.core.finetune import fine_tune_on_paper
    fine_tune_on_paper(
        arxiv_id=arxiv_id,
        base_model=request.base_model,
        run_name=request.run_name,
        num_epochs=request.num_epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        lora_r=request.lora_r,
        lora_alpha=request.lora_alpha,
        lora_dropout=request.lora_dropout,
        max_length=request.max_length,
        max_prompt_length=request.max_prompt_length,
        beta=request.beta,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
    )


@router.post("/paper/finetune", response_model=PaperFinetuneResponse)
def start_paper_finetune(req: PaperFinetuneRequest, background_tasks: BackgroundTasks):
    """
    Start paper-specific DPO fine-tuning for a single paper.

    Generates Q&A pairs from the paper's content and fine-tunes a model
    specialized for that paper. Training runs in the background.

    Check progress with GET /train/finetune/status.
    """
    # Check if training is already running
    status = get_training_status()
    if status["running"]:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress: {status['run_name']}"
        )

    # Validate paper exists
    from app.core.raptor_index import list_all_papers
    papers = list_all_papers()
    if req.arxiv_id not in papers:
        raise HTTPException(
            status_code=404,
            detail=f"Paper {req.arxiv_id} not found in system"
        )

    # Launch training in background
    background_tasks.add_task(_run_paper_training_background, req.arxiv_id, req)

    return PaperFinetuneResponse(
        status="started",
        run_name=req.run_name,
        arxiv_id=req.arxiv_id,
        message=f"Paper-specific training started for {req.arxiv_id}. "
                f"Base model: {req.base_model}. Check /train/finetune/status for progress.",
    )


@router.get("/paper/models/{arxiv_id}")
def get_paper_models(arxiv_id: str):
    """
    Get all fine-tuned models available for a specific paper.

    Returns list of model configs that were trained on this paper.
    """
    from app.core.finetune import get_paper_specific_models
    models = get_paper_specific_models(arxiv_id)
    if not models:
        return {"arxiv_id": arxiv_id, "models": [], "message": "No paper-specific models found"}
    return {"arxiv_id": arxiv_id, "models": models}


# =============================================
# Continuous Learning Loop endpoints (Section 13)
# =============================================

class LoopTriggerRequest(BaseModel):
    base_model: str = DEFAULT_BASE_MODEL
    num_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-5
    force: bool = False

class AutoLoopRequest(BaseModel):
    enable: bool
    min_new_feedback: int = 10
    check_interval_seconds: int = 300

class LoopConfigRequest(BaseModel):
    min_new_feedback: Optional[int] = None
    check_interval_seconds: Optional[int] = None


@router.post("/loop/trigger")
def trigger_loop(req: LoopTriggerRequest, background_tasks: BackgroundTasks):
    """
    Manually trigger one full learning loop cycle:
      1. Build preference dataset from all feedback
      2. Run DPO fine-tuning
      3. Register the new model for inference

    Set force=true to skip the minimum feedback threshold check.
    Training runs in the background.
    """
    # Quick pre-checks before launching background task
    training_status = get_training_status()
    if training_status["running"]:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress: {training_status['run_name']}"
        )

    pairs = preference_store.export_for_training()
    if not req.force and len(pairs) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 preference pairs. Currently have {len(pairs)}. "
                   f"Submit more feedback or use force=true."
        )

    def _run_loop():
        trigger_learning_loop(
            base_model=req.base_model,
            num_epochs=req.num_epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            force=req.force,
        )

    background_tasks.add_task(_run_loop)

    return {
        "status": "triggered",
        "message": "Learning loop started in background. Check /train/loop/status for progress.",
        "preference_pairs": len(pairs),
    }


@router.get("/loop/status")
def loop_status():
    """Get current state of the continuous learning loop."""
    return get_loop_status()


@router.post("/loop/auto")
def auto_loop(req: AutoLoopRequest):
    """
    Enable or disable automatic learning loop.

    When enabled, a background thread periodically checks if enough
    new feedback has accumulated and automatically triggers training.
    """
    if req.enable:
        return enable_auto_loop(
            min_new_feedback=req.min_new_feedback,
            check_interval_seconds=req.check_interval_seconds,
        )
    else:
        return disable_auto_loop()


@router.get("/loop/history")
def loop_history():
    """Get history of all learning loop runs (triggers, successes, failures)."""
    return get_loop_history()


@router.get("/loop/model")
def best_model():
    """
    Get the currently selected best model for inference.

    Returns the fine-tuned model if one is registered, otherwise the base model.
    Use this to know which model the system recommends.
    """
    return select_best_model()


@router.put("/loop/config")
def update_loop_config(req: LoopConfigRequest):
    """Update learning loop configuration (thresholds, intervals)."""
    return configure_loop(
        min_new_feedback=req.min_new_feedback,
        check_interval_seconds=req.check_interval_seconds,
    )
