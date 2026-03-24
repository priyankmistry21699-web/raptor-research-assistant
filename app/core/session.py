"""
Session Manager — In-memory session store for multi-turn chat.

Each session tracks:
  - Unique session ID (UUID4)
  - Chat history: list of messages with role, content, citations, timestamp
  - Papers referenced during the conversation
  - Creation timestamp

Thread-safe via threading.Lock for concurrent FastAPI/Gradio access.
"""

import uuid
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


class Session:
    """A single chat session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.history: List[Dict[str, Any]] = []
        self.papers_referenced: set = set()

    def add_message(
        self,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, str]]] = None,
    ):
        """Append a message to this session's history."""
        entry = {
            "role": role,
            "content": content,
            "citations": citations or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(entry)

        # Track referenced papers from citations
        if citations:
            for c in citations:
                arxiv_id = c.get("arxiv_id", "")
                if arxiv_id:
                    self.papers_referenced.add(arxiv_id)

    def get_llm_history(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """
        Return chat history in the format expected by prompt_builder.
        Only includes role + content, trimmed to the last N turns.
        """
        recent = self.history[-(max_turns * 2) :]  # user+assistant pairs
        return [{"role": h["role"], "content": h["content"]} for h in recent]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "message_count": len(self.history),
            "papers_referenced": list(self.papers_referenced),
            "history": self.history,
        }

    def summary(self) -> Dict[str, Any]:
        """Short summary without full history."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "message_count": len(self.history),
            "papers_referenced": list(self.papers_referenced),
        }


class SessionManager:
    """Thread-safe in-memory session store."""

    def __init__(self, max_sessions: int = 100):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def create_session(self) -> Session:
        """Create a new session with a unique ID."""
        session_id = uuid.uuid4().hex[:12]
        session = Session(session_id)
        with self._lock:
            # Evict oldest if at capacity
            if len(self._sessions) >= self._max_sessions:
                oldest_id = next(iter(self._sessions))
                del self._sessions[oldest_id]
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, or None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create a new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return summaries of all active sessions."""
        with self._lock:
            return [s.summary() for s in self._sessions.values()]


# Global singleton — shared between chat API and Gradio UI
session_manager = SessionManager()
