"""
Preference Dataset Creation — Converts user feedback into DPO training pairs.

Converts feedback entries from the FeedbackStore into preference data:
  (prompt, chosen, rejected) triples suitable for DPO/RLHF fine-tuning.

Conversion logic:
  - "helpful"       → answer is CHOSEN (rejected = "I don't know" placeholder)
  - "incorrect"     → answer is REJECTED (chosen = correction if provided, else skipped)
  - "hallucination" → answer is REJECTED (chosen = correction if provided, else skipped)
  - "correction"    → original answer is REJECTED, correction is CHOSEN

Storage: JSONL file with fields: prompt, chosen, rejected, metadata
Also supports HuggingFace Dataset-compatible export.
"""
import json
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from app.core.feedback import feedback_store

# Default storage path
DEFAULT_PREFERENCE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'preference'
)
DEFAULT_PREFERENCE_FILE = os.path.join(DEFAULT_PREFERENCE_DIR, 'preferences.jsonl')

# Placeholder rejection for "helpful" feedback where we don't have a real rejected answer
_DEFAULT_REJECTED = (
    "I'm sorry, I don't have enough information to answer that question accurately."
)


def _build_prompt_text(entry: Dict[str, Any]) -> str:
    """Reconstruct the prompt text from a feedback entry's question + context."""
    question = entry.get("question", "")
    citations = entry.get("citations", [])
    context_chunks = entry.get("context_chunks", [])

    parts = []
    # Include context if available
    if context_chunks:
        parts.append("Context:")
        for i, chunk in enumerate(context_chunks[:5], 1):
            text = chunk.get("chunk_text", chunk.get("text", ""))
            if text:
                parts.append(f"[{i}] {text[:500]}")
        parts.append("")
    elif citations:
        parts.append("Sources:")
        for c in citations[:5]:
            title = c.get("title", c.get("paper_title", ""))
            if title:
                parts.append(f"- {title}")
        parts.append("")

    parts.append(f"Question: {question}")
    return "\n".join(parts)


def feedback_to_preference(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a single feedback entry to a preference pair.

    Returns a dict with keys: prompt, chosen, rejected, metadata
    or None if the entry cannot be converted.
    """
    fb_type = entry.get("feedback_type", "")
    answer = entry.get("answer", "")
    correction = entry.get("correction", "")
    question = entry.get("question", "")

    if not question or not answer:
        return None

    prompt = _build_prompt_text(entry)
    metadata = {
        "session_id": entry.get("session_id", ""),
        "feedback_type": fb_type,
        "model_used": entry.get("model_used", ""),
        "task": entry.get("task", "qa"),
        "timestamp": entry.get("timestamp", ""),
    }

    if fb_type == "helpful":
        # User said the answer was good → answer is chosen
        return {
            "prompt": prompt,
            "chosen": answer,
            "rejected": _DEFAULT_REJECTED,
            "metadata": metadata,
        }

    elif fb_type == "correction":
        # User provided a better answer → correction is chosen, original is rejected
        if not correction.strip():
            return None
        return {
            "prompt": prompt,
            "chosen": correction,
            "rejected": answer,
            "metadata": metadata,
        }

    elif fb_type in ("incorrect", "hallucination"):
        # Answer was bad → it's the rejected response
        if correction.strip():
            # User also gave a correction → use it as chosen
            return {
                "prompt": prompt,
                "chosen": correction,
                "rejected": answer,
                "metadata": metadata,
            }
        # No correction provided — skip (we need both chosen & rejected for DPO)
        return None

    return None


class PreferenceStore:
    """
    Thread-safe preference dataset storage.

    Reads feedback, converts to preference pairs, stores as JSONL.
    Supports incremental builds (only process new feedback).
    """

    def __init__(self, filepath: str = DEFAULT_PREFERENCE_FILE):
        self._filepath = filepath
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)

    def _read_all(self) -> List[Dict[str, Any]]:
        """Read all existing preference pairs."""
        if not os.path.exists(self._filepath):
            return []
        entries = []
        with open(self._filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _write_all(self, entries: List[Dict[str, Any]]):
        """Overwrite the preference file with given entries."""
        with open(self._filepath, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _append(self, entry: Dict[str, Any]):
        """Append a single preference pair."""
        with open(self._filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def build_from_feedback(self) -> Dict[str, Any]:
        """
        Build/rebuild the full preference dataset from all feedback.

        Returns stats about the conversion.
        """
        all_feedback = feedback_store.get_all()
        pairs = []
        skipped = 0

        for fb in all_feedback:
            pref = feedback_to_preference(fb)
            if pref:
                pairs.append(pref)
            else:
                skipped += 1

        with self._lock:
            self._write_all(pairs)

        return {
            "total_feedback": len(all_feedback),
            "pairs_created": len(pairs),
            "skipped": skipped,
            "output_file": self._filepath,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }

    def add_from_feedback_entry(self, feedback_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert a single feedback entry and append it to the dataset.
        Returns the preference pair if created, None if skipped.
        """
        pref = feedback_to_preference(feedback_entry)
        if pref:
            with self._lock:
                self._append(pref)
        return pref

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all preference pairs."""
        with self._lock:
            return self._read_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the preference dataset."""
        pairs = self._read_all()
        type_counts: Dict[str, int] = {}
        for p in pairs:
            ft = p.get("metadata", {}).get("feedback_type", "unknown")
            type_counts[ft] = type_counts.get(ft, 0) + 1

        return {
            "total_pairs": len(pairs),
            "by_feedback_type": type_counts,
            "has_real_rejected": sum(
                1 for p in pairs
                if p.get("rejected", "") != _DEFAULT_REJECTED
            ),
            "output_file": self._filepath,
        }

    def export_for_training(self) -> List[Dict[str, str]]:
        """
        Export preference pairs in the format expected by TRL DPOTrainer:
          [{"prompt": ..., "chosen": ..., "rejected": ...}, ...]

        Strips metadata for clean training input.
        """
        pairs = self._read_all()
        return [
            {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
            for p in pairs
        ]

    def count(self) -> int:
        """Total number of preference pairs."""
        if not os.path.exists(self._filepath):
            return 0
        with self._lock:
            with open(self._filepath, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())


# Global singleton
preference_store = PreferenceStore()
