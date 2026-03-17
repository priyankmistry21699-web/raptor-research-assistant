"""
Model Fine-Tuning — DPO preference-based fine-tuning using TRL + PEFT.

Supports:
  - DPO (Direct Preference Optimization) training on (prompt, chosen, rejected) pairs
  - LoRA adapters via PEFT for parameter-efficient training
  - Saves adapters to models/<run_name>/ for serving
  - Registers fine-tuned model in llm_client MODEL_REGISTRY for inference

Workflow:
  1. Load preference dataset from PreferenceStore.export_for_training()
  2. Tokenize into DPO format
  3. Train with LoRA adapters on a base model (e.g., mistralai/Mistral-7B-v0.1)
  4. Save adapter weights + tokenizer to models/<run_name>/
  5. Optionally merge adapter into base model for standalone serving
"""
import os
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-v0.1"

# Track active training runs
_training_lock = threading.Lock()
_training_status: Dict[str, Any] = {
    "running": False,
    "run_name": None,
    "progress": None,
    "error": None,
    "last_completed": None,
}


def get_training_status() -> Dict[str, Any]:
    """Get the current training status."""
    with _training_lock:
        return dict(_training_status)


def list_finetuned_models() -> List[Dict[str, str]]:
    """List all saved fine-tuned model adapters."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = []
    for name in sorted(os.listdir(MODELS_DIR)):
        model_dir = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(model_dir):
            continue
        config_path = os.path.join(model_dir, "training_config.json")
        info = {"name": name, "path": model_dir}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                info["config"] = json.load(f)
        # Check if adapter files exist
        info["has_adapter"] = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
        info["has_tokenizer"] = os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
        models.append(info)
    return models


def run_dpo_training(
    preference_pairs: List[Dict[str, str]],
    base_model: str = DEFAULT_BASE_MODEL,
    run_name: Optional[str] = None,
    num_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_length: int = 512,
    max_prompt_length: int = 256,
    beta: float = 0.1,
    gradient_accumulation_steps: int = 4,
) -> Dict[str, Any]:
    """
    Run DPO fine-tuning with LoRA adapters.

    Args:
        preference_pairs: List of {"prompt", "chosen", "rejected"} dicts
        base_model: HuggingFace model ID or local path
        run_name: Name for this training run (used as output dir name)
        num_epochs: Number of training epochs
        batch_size: Per-device training batch size
        learning_rate: Learning rate for optimizer
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        max_length: Maximum sequence length for chosen/rejected
        max_prompt_length: Maximum prompt length
        beta: DPO beta parameter (controls deviation from reference model)
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        Dict with training results, output path, and metrics
    """
    # Lazy imports — these are heavy and shouldn't load at module import time
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import DPOTrainer, DPOConfig
    except ImportError as e:
        return {
            "status": "error",
            "error": f"Missing dependency: {e}. Install with: pip install torch transformers datasets trl peft",
        }

    with _training_lock:
        if _training_status["running"]:
            return {
                "status": "error",
                "error": f"Training already in progress: {_training_status['run_name']}",
            }
        _training_status["running"] = True
        _training_status["error"] = None

    if not run_name:
        run_name = f"dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = os.path.join(MODELS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    with _training_lock:
        _training_status["run_name"] = run_name
        _training_status["progress"] = "loading_model"

    try:
        # --- Validate dataset ---
        if len(preference_pairs) < 2:
            raise ValueError(
                f"Need at least 2 preference pairs for DPO training, got {len(preference_pairs)}"
            )

        logger.info(f"Starting DPO training: {run_name}")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  Training pairs: {len(preference_pairs)}")
        logger.info(f"  Output: {output_dir}")

        # --- Save training config ---
        train_config = {
            "base_model": base_model,
            "run_name": run_name,
            "num_pairs": len(preference_pairs),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
            "beta": beta,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
            json.dump(train_config, f, indent=2)

        # --- Load tokenizer ---
        with _training_lock:
            _training_status["progress"] = "loading_tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Load base model with quantization for memory efficiency ---
        with _training_lock:
            _training_status["progress"] = "loading_base_model"

        device_map = "auto"
        model_kwargs = {"trust_remote_code": True}

        # Try 4-bit quantization if bitsandbytes available (saves memory)
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_config
            logger.info("  Using 4-bit quantization (bitsandbytes)")
        except ImportError:
            logger.info("  bitsandbytes not available, loading in float16")
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            **model_kwargs,
        )

        # --- Configure LoRA ---
        with _training_lock:
            _training_status["progress"] = "configuring_lora"

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # --- Prepare dataset ---
        with _training_lock:
            _training_status["progress"] = "preparing_dataset"

        dataset = Dataset.from_list(preference_pairs)
        # Split 90/10 if enough data
        if len(preference_pairs) >= 10:
            split = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        # --- Training arguments ---
        with _training_lock:
            _training_status["progress"] = "training"

        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=1,
            save_steps=50,
            save_total_limit=2,
            remove_unused_columns=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            report_to="none",
            beta=beta,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )

        # --- Initialize DPO Trainer ---
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        # --- Train ---
        train_result = trainer.train()

        # --- Save adapter + tokenizer ---
        with _training_lock:
            _training_status["progress"] = "saving_model"

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save training metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Adapter saved to {output_dir}")

        result = {
            "status": "success",
            "run_name": run_name,
            "output_dir": output_dir,
            "base_model": base_model,
            "num_pairs_trained": len(preference_pairs),
            "train_loss": train_result.training_loss,
            "completed_at": metrics["completed_at"],
        }

        with _training_lock:
            _training_status["running"] = False
            _training_status["progress"] = "completed"
            _training_status["last_completed"] = result

        return result

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        error_result = {
            "status": "error",
            "run_name": run_name,
            "error": str(e),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(output_dir, "training_error.json"), 'w') as f:
            json.dump(error_result, f, indent=2)

        with _training_lock:
            _training_status["running"] = False
            _training_status["progress"] = "failed"
            _training_status["error"] = str(e)

        return error_result


def register_finetuned_model(run_name: str, alias: Optional[str] = None) -> Dict[str, Any]:
    """
    Register a fine-tuned adapter in the LLM model registry for inference.

    This adds the fine-tuned model to MODEL_REGISTRY so it can be used
    via run_llm_messages(model=alias).

    Note: For adapter-based models, inference requires loading base model + adapter,
    which is different from API-based models. This registers the metadata;
    actual loading happens at inference time.
    """
    from app.core.llm_client import MODEL_REGISTRY

    model_dir = os.path.join(MODELS_DIR, run_name)
    if not os.path.isdir(model_dir):
        return {"status": "error", "error": f"Model directory not found: {model_dir}"}

    config_path = os.path.join(model_dir, "training_config.json")
    if not os.path.exists(config_path):
        return {"status": "error", "error": "No training_config.json found"}

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_alias = alias or f"finetuned-{run_name}"

    MODEL_REGISTRY[model_alias] = {
        "model": model_dir,
        "api_url": "local",
        "api_key": "local",
        "base_model": config.get("base_model", DEFAULT_BASE_MODEL),
        "adapter_path": model_dir,
        "is_finetuned": True,
        "max_tokens": 1024,
        "temperature": 0.3,
        "timeout": 300,
    }

    return {
        "status": "registered",
        "alias": model_alias,
        "base_model": config.get("base_model"),
        "adapter_path": model_dir,
    }
