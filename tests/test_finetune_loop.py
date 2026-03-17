"""
Tests for app/core/finetune.py and app/core/learning_loop.py.

Does not run actual DPO training (requires GPU + model weights).
Tests focus on status, listing, and configuration functions.
"""
import pytest


class TestFinetuneModule:
    """Tests for finetune.py status and listing functions."""

    def test_import(self):
        from app.core.finetune import (
            get_training_status,
            list_finetuned_models,
            register_finetuned_model,
            run_dpo_training,
        )

    def test_get_training_status(self):
        from app.core.finetune import get_training_status
        status = get_training_status()
        assert isinstance(status, dict)
        assert "running" in status

    def test_list_finetuned_models(self):
        from app.core.finetune import list_finetuned_models
        models = list_finetuned_models()
        assert isinstance(models, list)


class TestLearningLoopModule:
    """Tests for learning_loop.py status and configuration functions."""

    def test_import(self):
        from app.core.learning_loop import (
            get_loop_status,
            get_loop_history,
            trigger_learning_loop,
            enable_auto_loop,
            disable_auto_loop,
            configure_loop,
            select_best_model,
        )

    def test_get_loop_status(self):
        from app.core.learning_loop import get_loop_status
        status = get_loop_status()
        assert isinstance(status, dict)
        assert "auto_enabled" in status

    def test_get_loop_history(self):
        from app.core.learning_loop import get_loop_history
        history = get_loop_history()
        assert isinstance(history, list)

    def test_configure_loop(self):
        from app.core.learning_loop import configure_loop, get_loop_status
        configure_loop(min_new_feedback=20, check_interval_seconds=600)
        status = get_loop_status()
        assert status["min_new_feedback"] == 20
        assert status["check_interval_seconds"] == 600
        # Reset to defaults
        configure_loop(min_new_feedback=10, check_interval_seconds=300)


class TestEvaluationModule:
    """Tests for evaluation.py — import and history functions only."""

    def test_import(self):
        from app.core.evaluation import (
            evaluate_single,
            evaluate_batch,
            evaluate_pipeline,
            compare_models,
            get_eval_history,
            get_eval_stats,
        )

    def test_get_eval_history(self):
        from app.core.evaluation import get_eval_history
        history = get_eval_history()
        assert isinstance(history, list)

    def test_get_eval_stats(self):
        from app.core.evaluation import get_eval_stats
        stats = get_eval_stats()
        assert isinstance(stats, dict)
