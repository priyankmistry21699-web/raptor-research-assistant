"""
Input sanitization utilities.

Validates and sanitizes user-provided text before processing
(LLM prompts, collection names, document titles, etc.).
"""

import re
import html
from typing import Optional


# Patterns commonly used in prompt injection attacks
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"disregard\s+(previous|above|all)",
    r"you\s+are\s+now\s+(?:a|an|the)\s+",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def sanitize_prompt(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user prompt text for LLM consumption.
    - Strips dangerous control characters
    - Truncates to max length
    - Does NOT strip injection patterns (only flags them)
    """
    # Remove null bytes and other control chars (keep newline, tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:max_length].strip()


def check_prompt_injection(text: str) -> Optional[str]:
    """
    Check for common prompt injection patterns.
    Returns the matched pattern description if found, None if clean.
    """
    for pattern in _compiled_patterns:
        if pattern.search(text):
            return pattern.pattern
    return None


def sanitize_name(text: str, max_length: int = 200) -> str:
    """Sanitize a collection/document name — alphanumeric, spaces, hyphens, underscores."""
    text = text.strip()[:max_length]
    # Allow only safe characters
    text = re.sub(r"[^\w\s\-.]", "", text)
    return text.strip()


def sanitize_html(text: str) -> str:
    """HTML-escape text for safe inclusion in API responses."""
    return html.escape(text, quote=True)
