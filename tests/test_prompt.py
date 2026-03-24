"""
Tests for app/core/prompt_builder.py and app/core/prompt.py — Prompt construction.
"""

from app.core.prompt_builder import (
    format_context_block,
    format_chat_history,
    build_prompt,
    build_messages,
)
from app.core.prompt import (
    SYSTEM_PROMPTS,
    TASK_INSTRUCTIONS,
    CONTEXT_HEADER,
    CONTEXT_CHUNK_TEMPLATE,
    CHAT_HISTORY_HEADER,
)


class TestPromptTemplates:
    """Tests for prompt.py template constants."""

    def test_all_task_types_covered(self):
        tasks = {"qa", "summarize", "compare", "explain"}
        assert set(SYSTEM_PROMPTS.keys()) == tasks
        assert set(TASK_INSTRUCTIONS.keys()) == tasks

    def test_system_prompts_not_empty(self):
        for task, prompt in SYSTEM_PROMPTS.items():
            assert len(prompt) > 50, f"System prompt for '{task}' is too short"

    def test_task_instructions_not_empty(self):
        for task, inst in TASK_INSTRUCTIONS.items():
            assert len(inst) > 20, f"Task instruction for '{task}' is too short"

    def test_context_header(self):
        assert isinstance(CONTEXT_HEADER, str)
        assert len(CONTEXT_HEADER) > 0

    def test_context_chunk_template_placeholders(self):
        required_placeholders = [
            "{index}",
            "{paper}",
            "{arxiv_id}",
            "{topic}",
            "{section_label}",
            "{text}",
        ]
        for ph in required_placeholders:
            assert ph in CONTEXT_CHUNK_TEMPLATE, f"Missing placeholder {ph}"


class TestFormatContextBlock:
    """Tests for format_context_block()."""

    def test_basic_format(self, sample_chunks):
        block = format_context_block(sample_chunks[0], 1)
        assert "1706.03762" in block
        assert "Attention Is All You Need" in block
        assert "Introduction" in block
        assert "dominant sequence" in block

    def test_missing_fields_graceful(self):
        chunk = {"chunk_text": "Some text"}
        block = format_context_block(chunk, 1)
        assert "Some text" in block


class TestFormatChatHistory:
    """Tests for format_chat_history()."""

    def test_empty_history(self):
        assert format_chat_history([]) == ""
        assert format_chat_history(None) == ""

    def test_basic_history(self):
        history = [
            {"role": "user", "content": "What is attention?"},
            {"role": "assistant", "content": "It is a mechanism..."},
        ]
        result = format_chat_history(history)
        assert CHAT_HISTORY_HEADER in result
        assert "User: What is attention?" in result
        assert "Assistant: It is a mechanism..." in result


class TestBuildPrompt:
    """Tests for build_prompt()."""

    def test_qa_prompt(self, sample_chunks):
        prompt = build_prompt(sample_chunks, "What is Transformer?", task="qa")
        assert SYSTEM_PROMPTS["qa"] in prompt
        assert "What is Transformer?" in prompt
        assert "1706.03762" in prompt

    def test_all_task_types(self, sample_chunks):
        for task in ["qa", "summarize", "compare", "explain"]:
            prompt = build_prompt(sample_chunks, "Question?", task=task)
            assert SYSTEM_PROMPTS[task] in prompt

    def test_with_chat_history(self, sample_chunks):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        prompt = build_prompt(sample_chunks, "Follow up?", chat_history=history)
        assert CHAT_HISTORY_HEADER in prompt

    def test_without_chat_history(self, sample_chunks):
        prompt = build_prompt(sample_chunks, "Question?")
        assert CHAT_HISTORY_HEADER not in prompt

    def test_empty_chunks(self):
        prompt = build_prompt([], "Question?")
        assert "Question?" in prompt


class TestBuildMessages:
    """Tests for build_messages()."""

    def test_returns_list_of_messages(self, sample_chunks):
        messages = build_messages(sample_chunks, "What is Transformer?")
        assert isinstance(messages, list)
        assert len(messages) >= 2

    def test_message_format(self, sample_chunks):
        messages = build_messages(sample_chunks, "Q?")
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("system", "user", "assistant")

    def test_system_message_first(self, sample_chunks):
        messages = build_messages(sample_chunks, "Q?")
        assert messages[0]["role"] == "system"

    def test_all_task_types(self, sample_chunks):
        for task in ["qa", "summarize", "compare", "explain"]:
            messages = build_messages(sample_chunks, "Q?", task=task)
            assert messages[0]["role"] == "system"
