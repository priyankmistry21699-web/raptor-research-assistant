"""
Tests for app/core/llm_client.py — LLM inference routing.

Note: Actual LLM calls are not tested here (requires running Ollama/Groq).
Tests focus on model registry, routing logic, and configuration.
"""
import pytest
from app.core.llm_client import (
    list_available_models,
    get_active_model,
    MODEL_REGISTRY,
    TASK_PARAMS,
)


class TestModelRegistry:
    """Tests for MODEL_REGISTRY and model management."""

    def test_registry_has_models(self):
        assert len(MODEL_REGISTRY) >= 1

    def test_mistral_in_registry(self):
        assert "mistral" in MODEL_REGISTRY
        cfg = MODEL_REGISTRY["mistral"]
        assert "model" in cfg
        assert "api_url" in cfg

    def test_list_available_models(self):
        models = list_available_models()
        assert isinstance(models, dict)
        assert "mistral" in models

    def test_get_active_model(self):
        model = get_active_model()
        assert isinstance(model, str)
        assert len(model) > 0


class TestTaskParams:
    """Tests for task-specific LLM parameters."""

    def test_all_tasks_defined(self):
        expected = {"qa", "summarize", "compare", "explain"}
        assert expected.issubset(set(TASK_PARAMS.keys()))

    def test_task_params_have_required_keys(self):
        for task, params in TASK_PARAMS.items():
            assert "max_tokens" in params, f"'{task}' missing max_tokens"
            assert "temperature" in params, f"'{task}' missing temperature"

    def test_temperature_range(self):
        for task, params in TASK_PARAMS.items():
            temp = params["temperature"]
            assert 0.0 <= temp <= 2.0, f"'{task}' temperature {temp} out of range"

    def test_max_tokens_positive(self):
        for task, params in TASK_PARAMS.items():
            assert params["max_tokens"] > 0
