"""
Feedback System — Collects, stores, and queries user feedback on LLM answers.

Feedback types:
  - helpful       : The answer was useful and accurate
  - incorrect     : The answer contained factual errors
  - hallucination : The answer fabricated information not in the sources
  - correction    : User provides a corrected version of the answer

Each feedback entry stores:
  - The original question and LLM answer
  - Retrieved context (chunks/citations) used to generate the answer
  - Session ID, model used, task type
  - Feedback type + optional user correction text
  - Timestamp

Storage: JSONL file (one JSON object per line) for easy appending and streaming reads.
This feeds into Section 11 (Preference Dataset Creation) for RLHF/DPO training.
"""
import json
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Feedback types the user can submit
FEEDBACK_TYPES = {"helpful", "incorrect", "hallucination", "correction"}

# Default storage path
DEFAULT_FEEDBACK_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'feedback'
)
DEFAULT_FEEDBACK_FILE = os.path.join(DEFAULT_FEEDBACK_DIR, 'feedback.jsonl')


class FeedbackEntry:
    """A single feedback record."""

    def __init__(
        self,
        session_id: str,
        question: str,
        answer: str,
        feedback_type: str,
        correction: str = "",
        model_used: str = "",
        task: str = "qa",
        citations: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ):
        if feedback_type not in FEEDBACK_TYPES:
            raise ValueError(
                f"Invalid feedback type '{feedback_type}'. Must be one of: {FEEDBACK_TYPES}"
            )
        self.session_id = session_id
        self.question = question
        self.answer = answer
        self.feedback_type = feedback_type
        self.correction = correction
        self.model_used = model_used
        self.task = task
        self.citations = citations or []
        self.context_chunks = context_chunks or []
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "feedback_type": self.feedback_type,
            "correction": self.correction,
            "model_used": self.model_used,
            "task": self.task,
            "citations": self.citations,
            "context_chunks": self.context_chunks,
            "timestamp": self.timestamp,
        }


class FeedbackStore:
    """
    Thread-safe feedback storage backed by a JSONL file.

    Each line in the file is one JSON object — easy to append,
    easy to stream-read for preference dataset creation.
    """

    def __init__(self, filepath: str = DEFAULT_FEEDBACK_FILE):
        self._filepath = filepath
        self._lock = threading.Lock()
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)

    def add(self, entry: FeedbackEntry) -> Dict[str, Any]:
        """Append a feedback entry to the JSONL file."""
        record = entry.to_dict()
        with self._lock:
            with open(self._filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record

    def submit(
        self,
        session_id: str,
        question: str,
        answer: str,
        feedback_type: str,
        correction: str = "",
        model_used: str = "",
        task: str = "qa",
        citations: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convenience method: create entry and store it in one call."""
        entry = FeedbackEntry(
            session_id=session_id,
            question=question,
            answer=answer,
            feedback_type=feedback_type,
            correction=correction,
            model_used=model_used,
            task=task,
            citations=citations,
            context_chunks=context_chunks,
        )
        return self.add(entry)

    def get_all(self) -> List[Dict[str, Any]]:
        """Read all feedback entries."""
        if not os.path.exists(self._filepath):
            return []
        entries = []
        with self._lock:
            with open(self._filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a specific session."""
        return [e for e in self.get_all() if e.get("session_id") == session_id]

    def get_by_type(self, feedback_type: str) -> List[Dict[str, Any]]:
        """Get all feedback of a specific type (e.g., all 'correction' entries)."""
        return [e for e in self.get_all() if e.get("feedback_type") == feedback_type]

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics of all collected feedback."""
        entries = self.get_all()
        type_counts = {}
        for e in entries:
            ft = e.get("feedback_type", "unknown")
            type_counts[ft] = type_counts.get(ft, 0) + 1
        return {
            "total": len(entries),
            "by_type": type_counts,
            "unique_sessions": len(set(e.get("session_id", "") for e in entries)),
        }

    def count(self) -> int:
        """Total number of feedback entries."""
        if not os.path.exists(self._filepath):
            return 0
        with self._lock:
            with open(self._filepath, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())


# Global singleton
feedback_store = FeedbackStore()
