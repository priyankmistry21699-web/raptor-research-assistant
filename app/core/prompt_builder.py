"""
Prompt Construction — Assembles prompts for the LLM using retrieved context and user question.

Supports multiple task types:
  - qa          : Answer a question using retrieved context
  - summarize   : Summarize a paper or topic
  - compare     : Compare findings across multiple papers
  - explain     : Explain a concept from the papers

Each prompt includes:
  1. System instruction (role + task)
  2. Hierarchical context (paper → topic → section → chunk)
  3. User question
  4. Chat history (if provided, for multi-turn conversations)
"""
from typing import List, Dict, Any, Optional

from app.core.prompt import (
    SYSTEM_PROMPTS,
    TASK_INSTRUCTIONS,
    CONTEXT_HEADER,
    CONTEXT_CHUNK_TEMPLATE,
    CHAT_HISTORY_HEADER,
)


def format_context_block(chunk: Dict[str, Any], index: int) -> str:
    """
    Format a single retrieved chunk into a structured context block.
    Uses full hierarchical info: paper → topic → section → chunk text.
    """
    paper = chunk.get("paper_title", "Unknown Paper")
    topic = chunk.get("topic", "")
    topic_summary = chunk.get("topic_summary", "")
    section_title = chunk.get("section_title", "")
    section_num = chunk.get("section_num", "")
    section_summary = chunk.get("section_summary", "")
    text = chunk.get("chunk_text", "")
    arxiv_id = chunk.get("arxiv_id", "")

    return CONTEXT_CHUNK_TEMPLATE.format(
        index=index,
        paper=paper,
        arxiv_id=arxiv_id,
        topic=topic,
        topic_summary=topic_summary,
        section_label=f"{section_num}: {section_title}" if section_num else section_title,
        section_summary=section_summary,
        text=text,
    )


def format_chat_history(history: List[Dict[str, str]]) -> str:
    """Format previous chat turns for multi-turn context."""
    if not history:
        return ""
    lines = [CHAT_HISTORY_HEADER]
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def build_prompt(
    chunks: List[Dict[str, Any]],
    user_question: str,
    task: str = "qa",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Assemble a full prompt for the LLM.

    Args:
        chunks: Retrieved context chunks (from RaptorRetriever).
        user_question: The user's current question.
        task: One of 'qa', 'summarize', 'compare', 'explain'.
        chat_history: Optional list of previous turns [{'role': 'user'|'assistant', 'content': '...'}].

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    # 1. System instruction
    system = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["qa"])

    # 2. Context blocks
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(format_context_block(chunk, i))
    context_str = CONTEXT_HEADER + "\n" + "\n".join(context_blocks)

    # 3. Chat history (if multi-turn)
    history_str = format_chat_history(chat_history) if chat_history else ""

    # 4. Task-specific instruction + user question
    task_instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS["qa"])

    # Assemble final prompt
    parts = [system, context_str]
    if history_str:
        parts.append(history_str)
    parts.append(f"Question:\n{user_question}")
    parts.append(f"Instructions:\n{task_instruction}")

    return "\n\n".join(parts)


def build_messages(
    chunks: List[Dict[str, Any]],
    user_question: str,
    task: str = "qa",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build an OpenAI-compatible messages list (system + user messages).
    Use this when calling APIs that accept a messages array instead of a single prompt string.

    Returns:
        [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
    """
    system = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["qa"])

    # Build user message with context
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(format_context_block(chunk, i))
    context_str = CONTEXT_HEADER + "\n" + "\n".join(context_blocks)

    task_instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS["qa"])

    user_content = f"{context_str}\n\nQuestion:\n{user_question}\n\nInstructions:\n{task_instruction}"

    messages = [{"role": "system", "content": system}]

    # Add chat history
    if chat_history:
        for turn in chat_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_content})

    return messages
