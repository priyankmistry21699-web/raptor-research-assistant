"""
Tests for app/frontend/ui.py — Gradio UI module.

Tests focus on importability, function existence, and helper logic.
Does not launch the actual Gradio server.
"""


class TestUIImports:
    """Verify the UI module and its key functions import correctly."""

    def test_import_create_ui(self):
        from app.frontend.ui import create_ui

        assert callable(create_ui)

    def test_import_chat_fn(self):
        from app.frontend.ui import chat_fn

        assert callable(chat_fn)

    def test_import_paper_browser_fns(self):
        from app.frontend.ui import list_papers_fn, paper_overview_fn

        assert callable(list_papers_fn)
        assert callable(paper_overview_fn)

    def test_import_dashboard_fn(self):
        from app.frontend.ui import dashboard_fn

        assert callable(dashboard_fn)


class TestUIHelpers:
    """Tests for UI helper functions."""

    def test_format_citations_md_empty(self):
        from app.frontend.ui import _format_citations_md

        result = _format_citations_md([])
        assert "No citations" in result

    def test_format_citations_md_with_data(self):
        from app.frontend.ui import _format_citations_md

        citations = [
            {
                "arxiv_id": "1706.03762",
                "paper_title": "Attention",
                "section": "Intro",
                "topic": "NLP",
                "excerpt": "Test...",
            },
        ]
        result = _format_citations_md(citations)
        assert "1706.03762" in result
        assert "Attention" in result
        assert "References" in result

    def test_format_session_list(self):
        from app.frontend.ui import _format_session_list

        result = _format_session_list()
        assert isinstance(result, list)

    def test_list_papers_fn(self):
        from app.frontend.ui import list_papers_fn

        result = list_papers_fn()
        assert isinstance(result, str)
        # Should have paper data or "No papers" message
        assert len(result) > 0

    def test_dashboard_fn(self):
        from app.frontend.ui import dashboard_fn

        result = dashboard_fn()
        assert isinstance(result, str)
        assert "Dashboard" in result or "Papers" in result.lower()
