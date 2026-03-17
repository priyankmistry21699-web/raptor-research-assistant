"""
LLM Client — Handles inference calls to OpenAI-compatible APIs and local fine-tuned models.

Supports:
  - Local Ollama (Mistral, Llama, etc.) at http://localhost:11435
  - Groq cloud API
  - Any OpenAI-compatible endpoint
  - Local fine-tuned models (base model + LoRA adapter via PEFT)

Features:
  - Multi-model routing: switch models per request
  - Messages-based API (system + user roles)
  - Task-specific parameters (temperature, max_tokens)
  - Fine-tuned LoRA adapter inference with model caching
  - Automatic fallback on timeout/error
"""
import os
import logging
import threading
import requests
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- Default config from environment ---
DEFAULT_API_URL = os.getenv("LLM_API_URL", "http://localhost:11435/v1/chat/completions")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistral:latest")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "ollama")

# --- Model registry: pre-configured models you can switch between ---
MODEL_REGISTRY = {
    "mistral": {
        "model": "mistral:latest",
        "api_url": "http://localhost:11435/v1/chat/completions",
        "api_key": "ollama",
        "max_tokens": 1024,
        "temperature": 0.3,
        "timeout": 180,
    },
    "groq-llama": {
        "model": "llama-3.3-70b-versatile",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "max_tokens": 1024,
        "temperature": 0.2,
        "timeout": 30,
    },
}

# --- Task-specific generation parameters ---
TASK_PARAMS = {
    "qa": {"max_tokens": 1024, "temperature": 0.3},
    "summarize": {"max_tokens": 512, "temperature": 0.2},
    "compare": {"max_tokens": 1500, "temperature": 0.3},
    "explain": {"max_tokens": 1200, "temperature": 0.4},
}

# --- Cache for loaded fine-tuned models (base + adapter) ---
_finetuned_cache: Dict[str, Any] = {}  # alias → {"model": ..., "tokenizer": ...}
_finetuned_lock = threading.Lock()


def _load_finetuned_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a fine-tuned model (base + LoRA adapter) into memory.
    Returns dict with 'model' and 'tokenizer' keys.
    Cached so subsequent calls reuse the same loaded model.
    """
    adapter_path = config["adapter_path"]

    with _finetuned_lock:
        if adapter_path in _finetuned_cache:
            return _finetuned_cache[adapter_path]

    # Lazy imports — heavy deps
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_id = config.get("base_model", "mistralai/Mistral-7B-v0.1")

    logger.info(f"Loading fine-tuned model: base={base_model_id}, adapter={adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try 4-bit quantization for memory efficiency
    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
    except ImportError:
        model_kwargs["torch_dtype"] = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    result = {"model": model, "tokenizer": tokenizer}

    with _finetuned_lock:
        _finetuned_cache[adapter_path] = result

    logger.info(f"Fine-tuned model loaded and cached: {adapter_path}")
    return result


def _run_finetuned_inference(
    config: Dict[str, Any],
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """Run inference locally using a fine-tuned base model + LoRA adapter."""
    import torch

    loaded = _load_finetuned_model(config)
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    # Format messages into a single prompt string
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n")
        elif role == "user":
            prompt_parts.append(f"{content} [/INST]")
        elif role == "assistant":
            prompt_parts.append(f"{content}\n[INST] ")
    prompt_text = "".join(prompt_parts)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (skip the input prompt tokens)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_llm(
    prompt: str,
    api_key: str = None,
    model: str = None,
    task: str = "qa",
) -> str:
    """
    Send a single prompt string to the LLM. Simple interface for backward compatibility.

    Args:
        prompt: The full prompt text.
        api_key: API key (defaults to env LLM_API_KEY).
        model: Model name (defaults to env LLM_MODEL).
        task: Task type for parameter tuning ('qa', 'summarize', 'compare', 'explain').
    """
    messages = [{"role": "user", "content": prompt}]
    return run_llm_messages(messages, api_key=api_key, model=model, task=task)


def run_llm_messages(
    messages: List[Dict[str, str]],
    api_key: str = None,
    model: str = None,
    api_url: str = None,
    task: str = "qa",
    max_tokens: int = None,
    temperature: float = None,
) -> str:
    """
    Send a messages array to the LLM (OpenAI-compatible format).
    Automatically routes to local fine-tuned inference if the model is a LoRA adapter.

    Args:
        messages: List of {'role': 'system'|'user'|'assistant', 'content': '...'}.
        api_key: API key.
        model: Model name or registry alias ('mistral', 'groq-llama', or fine-tuned alias).
        api_url: API endpoint URL.
        task: Task type for default parameters.
        max_tokens: Override max tokens.
        temperature: Override temperature.
    """
    # Check if model is a registry alias
    config = MODEL_REGISTRY.get(model, {})

    # Task-specific defaults
    task_cfg = TASK_PARAMS.get(task, TASK_PARAMS["qa"])
    max_tokens = max_tokens or task_cfg["max_tokens"]
    temperature = temperature if temperature is not None else task_cfg["temperature"]

    # Route to local fine-tuned inference if this is a LoRA adapter model
    if config and config.get("is_finetuned"):
        logger.info(f"LLM call (fine-tuned local): alias={model}, adapter={config.get('adapter_path')}, task={task}")
        return _run_finetuned_inference(config, messages, max_tokens, temperature)

    # Standard API-based inference
    if config:
        api_url = api_url or config["api_url"]
        api_key = api_key or config["api_key"]
        model = config["model"]
        timeout = config.get("timeout", 180)
    else:
        api_url = api_url or DEFAULT_API_URL
        api_key = api_key or DEFAULT_API_KEY
        model = model or DEFAULT_MODEL
        timeout = 180

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    logger.info(f"LLM call: model={model}, url={api_url}, task={task}")

    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def list_available_models() -> Dict[str, Dict]:
    """Return the model registry for inspection, including fine-tuned models."""
    result = {}
    for name, cfg in MODEL_REGISTRY.items():
        info = {"model": cfg["model"], "api_url": cfg["api_url"]}
        if cfg.get("is_finetuned"):
            info["is_finetuned"] = True
            info["base_model"] = cfg.get("base_model", "")
            info["adapter_path"] = cfg.get("adapter_path", "")
        result[name] = info
    return result


def get_active_model() -> str:
    """Return the alias of the currently recommended model (prefers fine-tuned if available)."""
    finetuned = [k for k, v in MODEL_REGISTRY.items() if v.get("is_finetuned")]
    if finetuned:
        return finetuned[-1]  # Most recently registered
    return "mistral"


def check_model_health(model_alias: str = "mistral") -> Dict[str, str]:
    """Quick health check — send a tiny prompt to verify the model responds."""
    try:
        result = run_llm("Say OK.", model=model_alias, task="qa")
        return {"status": "ok", "model": model_alias, "response": result[:50]}
    except Exception as e:
        return {"status": "error", "model": model_alias, "error": str(e)}
