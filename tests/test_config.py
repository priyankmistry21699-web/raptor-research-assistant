"""
Tests for config.yaml — validates configuration structure and values.

Ensures all required sections and keys are present and correctly typed.
"""
import os
import pytest
import yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="module")
def config():
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestConfigStructure:
    """Tests for top-level config sections."""

    REQUIRED_SECTIONS = [
        "arxiv", "embedding", "vector_db", "raptor", "llm",
        "retrieval", "session", "feedback", "preference",
        "finetune", "learning_loop", "evaluation", "server", "frontend",
    ]

    def test_config_loads(self, config):
        assert config is not None
        assert isinstance(config, dict)

    @pytest.mark.parametrize("section", REQUIRED_SECTIONS)
    def test_required_section_exists(self, config, section):
        assert section in config, f"Missing config section: {section}"


class TestArxivConfig:
    def test_categories(self, config):
        cats = config["arxiv"]["categories"]
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_max_results(self, config):
        assert isinstance(config["arxiv"]["max_results"], int)
        assert config["arxiv"]["max_results"] > 0


class TestEmbeddingConfig:
    def test_model(self, config):
        assert "model" in config["embedding"]
        assert "MiniLM" in config["embedding"]["model"] or "bge" in config["embedding"]["model"].lower()


class TestVectorDBConfig:
    def test_provider(self, config):
        assert config["vector_db"]["provider"] in ("chroma", "chromadb")

    def test_collection(self, config):
        assert isinstance(config["vector_db"]["collection"], str)


class TestRaptorConfig:
    def test_tree_dir(self, config):
        assert "tree_dir" in config["raptor"]

    def test_levels(self, config):
        levels = config["raptor"]["levels"]
        assert isinstance(levels, list)
        assert "chunk" in levels
        assert "paper" in levels

    def test_topic_clustering(self, config):
        tc = config["raptor"]["topic_clustering"]
        assert "method" in tc
        assert "max_topics" in tc

    def test_summarization(self, config):
        s = config["raptor"]["summarization"]
        assert "max_tokens" in s
        assert s["max_tokens"] > 0


class TestLLMConfig:
    def test_default_model(self, config):
        assert "default_model" in config["llm"]

    def test_models_defined(self, config):
        models = config["llm"]["models"]
        assert isinstance(models, dict)
        assert len(models) >= 1

    def test_mistral_config(self, config):
        if "mistral" in config["llm"]["models"]:
            m = config["llm"]["models"]["mistral"]
            assert "model" in m
            assert "api_url" in m

    def test_task_params(self, config):
        tp = config["llm"]["task_params"]
        for task in ["qa", "summarize", "compare", "explain"]:
            assert task in tp, f"Missing task_params for '{task}'"
            assert "max_tokens" in tp[task]
            assert "temperature" in tp[task]


class TestRetrievalConfig:
    def test_top_k(self, config):
        assert config["retrieval"]["top_k"] > 0

    def test_include_tree_context(self, config):
        assert isinstance(config["retrieval"]["include_tree_context"], bool)


class TestSessionConfig:
    def test_max_sessions(self, config):
        assert config["session"]["max_sessions"] > 0

    def test_max_history_turns(self, config):
        assert config["session"]["max_history_turns"] > 0


class TestFeedbackConfig:
    def test_storage(self, config):
        assert "storage" in config["feedback"]

    def test_types(self, config):
        types = config["feedback"]["types"]
        assert isinstance(types, list)
        assert "helpful" in types
        assert "correction" in types


class TestPreferenceConfig:
    def test_storage(self, config):
        assert "storage" in config["preference"]


class TestFinetuneConfig:
    def test_base_model(self, config):
        assert "base_model" in config["finetune"]

    def test_method(self, config):
        assert config["finetune"]["method"] == "dpo"

    def test_lora_config(self, config):
        lora = config["finetune"]["lora"]
        assert "r" in lora
        assert "alpha" in lora
        assert "dropout" in lora
        assert "target_modules" in lora

    def test_training_config(self, config):
        t = config["finetune"]["training"]
        assert t["num_epochs"] > 0
        assert t["batch_size"] > 0
        assert float(t["learning_rate"]) > 0


class TestLearningLoopConfig:
    def test_auto_enabled(self, config):
        assert isinstance(config["learning_loop"]["auto_enabled"], bool)

    def test_min_new_feedback(self, config):
        assert config["learning_loop"]["min_new_feedback"] > 0

    def test_check_interval(self, config):
        assert config["learning_loop"]["check_interval_seconds"] > 0


class TestEvaluationConfig:
    def test_llm_model(self, config):
        assert "llm_model" in config["evaluation"]

    def test_default_metrics(self, config):
        metrics = config["evaluation"]["default_metrics"]
        assert isinstance(metrics, list)
        assert len(metrics) > 0


class TestServerConfig:
    def test_host(self, config):
        assert "host" in config["server"]

    def test_port(self, config):
        assert isinstance(config["server"]["port"], int)
        assert 1024 <= config["server"]["port"] <= 65535

    def test_cors_origins(self, config):
        assert isinstance(config["server"]["cors_origins"], list)


class TestFrontendConfig:
    def test_type(self, config):
        assert config["frontend"]["type"] == "gradio"

    def test_port(self, config):
        assert isinstance(config["frontend"]["port"], int)

    def test_tabs(self, config):
        tabs = config["frontend"]["tabs"]
        assert isinstance(tabs, list)
        assert "chat" in tabs
        assert "papers" in tabs
        assert "upload" in tabs
        assert "dashboard" in tabs

    def test_defaults(self, config):
        assert config["frontend"]["default_model"] == "auto"
        assert config["frontend"]["default_task"] == "qa"
        assert config["frontend"]["top_k"] > 0
