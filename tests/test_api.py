"""
Tests for FastAPI API routers — chat, retrieve, feedback, train, eval.

Uses FastAPI's TestClient for HTTP-level integration testing.
Mocks external dependencies (LLM, ChromaDB) where needed.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the full FastAPI app."""
    from app.api.mcp_server import app

    return TestClient(app)


# ============================================================
# System Endpoints
# ============================================================


class TestSystemEndpoints:
    """Tests for top-level system routes."""

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "routes" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "papers" in data or "uptime_seconds" in data

    def test_config(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


# ============================================================
# Chat Endpoints
# ============================================================


class TestChatEndpoints:
    """Tests for /chat/* routes."""

    def test_create_session(self, client):
        resp = client.post("/chat/session")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data

    def test_list_sessions(self, client):
        # Create at least one session first
        client.post("/chat/session")
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_session(self, client):
        create_resp = client.post("/chat/session")
        sid = create_resp.json()["session_id"]
        resp = client.get(f"/chat/session/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid

    def test_get_nonexistent_session(self, client):
        resp = client.get("/chat/session/nonexistent_id")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        create_resp = client.post("/chat/session")
        sid = create_resp.json()["session_id"]
        resp = client.delete(f"/chat/session/{sid}")
        assert resp.status_code == 200
        # Confirm deleted
        get_resp = client.get(f"/chat/session/{sid}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent_session(self, client):
        resp = client.delete("/chat/session/nonexistent_id")
        assert resp.status_code == 404


# ============================================================
# Feedback Endpoints
# ============================================================


class TestFeedbackEndpoints:
    """Tests for /feedback/* routes."""

    def test_submit_feedback(self, client):
        resp = client.post(
            "/feedback",
            json={
                "session_id": "test_s1",
                "question": "What is attention?",
                "answer": "A mechanism.",
                "feedback_type": "helpful",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["feedback_type"] == "helpful"

    def test_get_all_feedback(self, client):
        resp = client.get("/feedback")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_feedback_stats(self, client):
        resp = client.get("/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data

    def test_get_feedback_by_type(self, client):
        # Submit a specific type first
        client.post(
            "/feedback",
            json={
                "session_id": "s2",
                "question": "Q",
                "answer": "A",
                "feedback_type": "incorrect",
            },
        )
        resp = client.get("/feedback/type/incorrect")
        assert resp.status_code == 200

    def test_get_feedback_by_session(self, client):
        resp = client.get("/feedback/session/test_s1")
        assert resp.status_code == 200


# ============================================================
# Retrieve Endpoints
# ============================================================


class TestRetrieveEndpoints:
    """Tests for /retrieve/* routes."""

    def test_list_papers(self, client):
        resp = client.get("/retrieve/papers")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_paper_overview(self, client):
        # Get a paper that exists
        papers_resp = client.get("/retrieve/papers")
        papers = papers_resp.json()
        if papers:
            resp = client.get(f"/retrieve/paper/{papers[0]}")
            assert resp.status_code == 200

    def test_get_paper_overview_nonexistent(self, client):
        resp = client.get("/retrieve/paper/NONEXISTENT_999")
        assert resp.status_code == 404


# ============================================================
# Train Endpoints
# ============================================================


class TestTrainEndpoints:
    """Tests for /train/* routes."""

    def test_get_preferences(self, client):
        resp = client.get("/train/preferences")
        assert resp.status_code == 200

    def test_get_preference_stats(self, client):
        resp = client.get("/train/preferences/stats")
        assert resp.status_code == 200

    def test_get_finetune_status(self, client):
        resp = client.get("/train/finetune/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data

    def test_get_finetune_models(self, client):
        resp = client.get("/train/finetune/models")
        assert resp.status_code == 200

    def test_get_loop_status(self, client):
        resp = client.get("/train/loop/status")
        assert resp.status_code == 200

    def test_get_loop_history(self, client):
        resp = client.get("/train/loop/history")
        assert resp.status_code == 200

    def test_get_loop_model(self, client):
        resp = client.get("/train/loop/model")
        assert resp.status_code == 200


# ============================================================
# Eval Endpoints
# ============================================================


class TestEvalEndpoints:
    """Tests for /eval/* routes."""

    def test_get_eval_history(self, client):
        resp = client.get("/eval/history")
        assert resp.status_code == 200

    def test_get_eval_stats(self, client):
        resp = client.get("/eval/stats")
        assert resp.status_code == 200


# ============================================================
# LLM Endpoints
# ============================================================


class TestLLMEndpoints:
    """Tests for /llm/* routes."""

    def test_list_models(self, client):
        resp = client.get("/llm/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "mistral" in data

    def test_llm_health(self, client):
        resp = client.get("/llm/health")
        assert resp.status_code == 200
