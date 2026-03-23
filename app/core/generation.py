"""
Generation Layer — LLM routing, prompt building, and structured response.

Uses LiteLLM for unified multi-provider access:
  - OpenAI, Anthropic, Groq (cloud)
  - Ollama, vLLM (self-hosted)

Falls back through providers on failure.
"""

import logging
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Provider mapping ──────────────────────────────────────────────────

def _litellm_model_name() -> str:
    """Map settings to a LiteLLM model string."""
    provider = settings.llm.provider.lower()
    model = settings.llm.model

    model_map = {
        "ollama": f"ollama/{model}",
        "openai": model,
        "anthropic": model,
        "groq": f"groq/{model}",
        "vllm": f"openai/{model}",  # vLLM exposes OpenAI-compatible API
    }
    return model_map.get(provider, model)


def _litellm_kwargs() -> dict:
    """Build extra kwargs for litellm.completion based on provider."""
    kwargs: dict[str, Any] = {}
    provider = settings.llm.provider.lower()

    if provider == "ollama":
        kwargs["api_base"] = settings.llm.ollama_base_url
    elif provider == "openai" and settings.llm.openai_api_key:
        kwargs["api_key"] = settings.llm.openai_api_key
    elif provider == "anthropic" and settings.llm.anthropic_api_key:
        kwargs["api_key"] = settings.llm.anthropic_api_key
    elif provider == "groq" and settings.llm.groq_api_key:
        kwargs["api_key"] = settings.llm.groq_api_key

    return kwargs


# ── Prompt templates ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable research assistant. Answer the user's question "
    "using ONLY the provided context. If the context does not contain enough "
    "information, say so clearly. Cite sources using [Source N] notation."
)

CONTEXT_TEMPLATE = (
    "### Context\n{context}\n\n"
    "### Question\n{question}"
)

CONVERSATIONAL_SYSTEM = (
    "You are a helpful assistant. Respond naturally to the user's message."
)


def _build_messages(
    question: str,
    context: str | None = None,
    chat_history: list[dict] | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    """Build the messages array for the LLM call."""
    messages: list[dict] = []

    if context:
        sys_prompt = system_prompt or SYSTEM_PROMPT
        messages.append({"role": "system", "content": sys_prompt})
    else:
        messages.append({"role": "system", "content": system_prompt or CONVERSATIONAL_SYSTEM})

    # Append chat history (last N turns to stay within token limits)
    if chat_history:
        for msg in chat_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Build user message
    if context:
        user_content = CONTEXT_TEMPLATE.format(context=context, question=question)
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})
    return messages


# ── Fallback chain ────────────────────────────────────────────────────

_FALLBACK_PROVIDERS = [
    ("groq", "groq/llama-3.3-70b-versatile"),
    ("openai", "gpt-4o-mini"),
]


def _try_fallback(messages: list[dict], temperature: float, max_tokens: int) -> str | None:
    """Attempt fallback providers when the primary fails."""
    import litellm

    for provider, model in _FALLBACK_PROVIDERS:
        api_key = None
        if provider == "groq":
            api_key = settings.llm.groq_api_key
        elif provider == "openai":
            api_key = settings.llm.openai_api_key

        if not api_key:
            continue

        try:
            logger.info("Attempting fallback: %s", model)
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("Fallback %s failed: %s", model, e)
            continue

    return None


# ── Main generation ───────────────────────────────────────────────────

def generate(
    question: str,
    context: str | None = None,
    chat_history: list[dict] | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Generate an LLM response.

    Returns:
        {
            "content": str,       # The generated text
            "model_used": str,    # Model that produced the response
            "latency_ms": int,    # Round-trip time
            "token_count": int,   # Approximate output tokens
        }
    """
    import litellm

    temperature = temperature if temperature is not None else settings.llm.temperature
    max_tokens = max_tokens or settings.llm.max_tokens

    messages = _build_messages(question, context, chat_history, system_prompt)
    model = _litellm_model_name()
    extra = _litellm_kwargs()

    start = time.perf_counter()

    try:
        resp = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        content = resp.choices[0].message.content
        model_used = model
        token_count = getattr(resp.usage, "completion_tokens", None) or len(content.split())
    except Exception as e:
        logger.warning("Primary LLM (%s) failed: %s — trying fallbacks", model, e)
        content = _try_fallback(messages, temperature, max_tokens)
        if content is None:
            raise RuntimeError(f"All LLM providers failed. Primary error: {e}") from e
        model_used = "fallback"
        token_count = len(content.split())

    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "content": content,
        "model_used": model_used,
        "latency_ms": latency_ms,
        "token_count": token_count,
    }


def generate_with_retrieval(
    question: str,
    collection_id,
    chat_history: list[dict] | None = None,
    top_k: int = 10,
    session=None,
) -> dict:
    """
    End-to-end RAG: retrieve context then generate.

    Returns:
        {
            "content": str,
            "citations": list,
            "model_used": str,
            "latency_ms": int,
            "token_count": int,
        }
    """
    from app.core.retrieval_orchestrator import retrieve, build_context_text

    retrieval = retrieve(
        query=question,
        collection_id=collection_id,
        top_k=top_k,
        session=session,
    )

    context = build_context_text(retrieval)

    result = generate(
        question=question,
        context=context,
        chat_history=chat_history,
    )

    result["citations"] = retrieval["citations"]
    return result
