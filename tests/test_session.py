"""
Tests for app/core/session.py — Session and SessionManager.
"""

from app.core.session import Session, SessionManager


class TestSession:
    """Tests for the Session class."""

    def test_create_session(self):
        s = Session("test123")
        assert s.session_id == "test123"
        assert s.history == []
        assert len(s.papers_referenced) == 0
        assert s.created_at is not None

    def test_add_user_message(self):
        s = Session("s1")
        s.add_message(role="user", content="Hello")
        assert len(s.history) == 1
        assert s.history[0]["role"] == "user"
        assert s.history[0]["content"] == "Hello"
        assert s.history[0]["citations"] == []
        assert "timestamp" in s.history[0]

    def test_add_assistant_message_with_citations(self):
        s = Session("s1")
        citations = [{"arxiv_id": "1706.03762", "paper_title": "Attention"}]
        s.add_message(role="assistant", content="Answer", citations=citations)
        assert len(s.history) == 1
        assert s.history[0]["citations"] == citations
        assert "1706.03762" in s.papers_referenced

    def test_papers_referenced_tracking(self):
        s = Session("s1")
        s.add_message("assistant", "A1", citations=[{"arxiv_id": "paper_a"}])
        s.add_message("assistant", "A2", citations=[{"arxiv_id": "paper_b"}])
        s.add_message("assistant", "A3", citations=[{"arxiv_id": "paper_a"}])
        assert s.papers_referenced == {"paper_a", "paper_b"}

    def test_get_llm_history(self):
        s = Session("s1")
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            s.add_message(role=role, content=f"msg_{i}")

        history = s.get_llm_history(max_turns=3)
        assert len(history) == 6  # 3 turns = 6 messages (user+assistant)
        assert all("role" in h and "content" in h for h in history)
        # Should not include citations or timestamps
        assert "citations" not in history[0]
        assert "timestamp" not in history[0]

    def test_get_llm_history_empty(self):
        s = Session("s1")
        assert s.get_llm_history() == []

    def test_to_dict(self):
        s = Session("s1")
        s.add_message("user", "Hi")
        d = s.to_dict()
        assert d["session_id"] == "s1"
        assert d["message_count"] == 1
        assert isinstance(d["history"], list)
        assert isinstance(d["papers_referenced"], list)
        assert "created_at" in d

    def test_summary(self):
        s = Session("s1")
        s.add_message("user", "Hi")
        summary = s.summary()
        assert summary["session_id"] == "s1"
        assert summary["message_count"] == 1
        assert "history" not in summary


class TestSessionManager:
    """Tests for the SessionManager class."""

    def test_create_session(self, session_manager):
        s = session_manager.create_session()
        assert s.session_id is not None
        assert len(s.session_id) == 12

    def test_get_session(self, session_manager):
        s = session_manager.create_session()
        found = session_manager.get_session(s.session_id)
        assert found is s

    def test_get_nonexistent_session(self, session_manager):
        assert session_manager.get_session("nonexistent") is None

    def test_get_or_create_existing(self, session_manager):
        s1 = session_manager.create_session()
        s2 = session_manager.get_or_create(s1.session_id)
        assert s2 is s1

    def test_get_or_create_new(self, session_manager):
        s = session_manager.get_or_create(None)
        assert s.session_id is not None

    def test_get_or_create_missing_id(self, session_manager):
        s = session_manager.get_or_create("nonexistent_id")
        assert s.session_id != "nonexistent_id"  # Creates a new one

    def test_delete_session(self, session_manager):
        s = session_manager.create_session()
        assert session_manager.delete_session(s.session_id) is True
        assert session_manager.get_session(s.session_id) is None

    def test_delete_nonexistent(self, session_manager):
        assert session_manager.delete_session("nope") is False

    def test_list_sessions(self, session_manager):
        session_manager.create_session()
        session_manager.create_session()
        sessions = session_manager.list_sessions()
        assert len(sessions) == 2
        assert all("session_id" in s for s in sessions)
        assert all("message_count" in s for s in sessions)

    def test_max_sessions_eviction(self):
        mgr = SessionManager(max_sessions=3)
        ids = []
        for _ in range(5):
            s = mgr.create_session()
            ids.append(s.session_id)
        # Only last 3 should remain
        sessions = mgr.list_sessions()
        assert len(sessions) == 3
        # First 2 should be evicted
        assert mgr.get_session(ids[0]) is None
        assert mgr.get_session(ids[1]) is None
        # Last 3 should exist
        assert mgr.get_session(ids[4]) is not None
