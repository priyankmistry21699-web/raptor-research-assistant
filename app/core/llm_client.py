"""
LLM Client — Handles inference calls to OpenAI-compatible APIs.

Supports:
  - Local Ollama (Mistral, Llama, etc.) at http://localhost:11435
  - Groq cloud API
  - Any OpenAI-compatible endpoint

Features:
  - Multi-model routing: switch models per request
  - Messages-based API (system + user roles)
  - Task-specific parameters (temperature, max_tokens)
  - Automatic fallback on timeout/error
"""
import os
import logging
import requests
from typing import List, Dict, Optional
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

    Args:
        messages: List of {'role': 'system'|'user'|'assistant', 'content': '...'}.
        api_key: API key.
        model: Model name or registry alias ('mistral', 'groq-llama').
        api_url: API endpoint URL.
        task: Task type for default parameters.
        max_tokens: Override max tokens.
        temperature: Override temperature.
    """
    # Check if model is a registry alias
    config = MODEL_REGISTRY.get(model, {})
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

    # Task-specific defaults
    task_cfg = TASK_PARAMS.get(task, TASK_PARAMS["qa"])
    max_tokens = max_tokens or task_cfg["max_tokens"]
    temperature = temperature if temperature is not None else task_cfg["temperature"]

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
    """Return the model registry for inspection."""
    return {
        name: {"model": cfg["model"], "api_url": cfg["api_url"]}
        for name, cfg in MODEL_REGISTRY.items()
    }


def check_model_health(model_alias: str = "mistral") -> Dict[str, str]:
    """Quick health check — send a tiny prompt to verify the model responds."""
    try:
        result = run_llm("Say OK.", model=model_alias, task="qa")
        return {"status": "ok", "model": model_alias, "response": result[:50]}
    except Exception as e:
        return {"status": "error", "model": model_alias, "error": str(e)}
