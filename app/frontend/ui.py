"""
Gradio Chat UI — Interactive two-way chatbot with session support.

Features:
  - Multi-turn chat with persistent sessions
  - Task type selector (Q&A, Summarize, Compare, Explain)
  - Model selector (Mistral local, Groq cloud)
  - Citations panel showing paper references for each answer
  - Session management (create / switch / view history)

Launch:
  python -m app.frontend.ui
"""
import os
import sys
import logging

import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.session import session_manager, Session
from app.core.retrieval import RaptorRetriever
from app.core.prompt_builder import build_messages
from app.core.llm_client import run_llm_messages
from app.core.feedback import feedback_store

logger = logging.getLogger(__name__)

# Shared retriever (lazy init)
_retriever = None


def _get_retriever() -> RaptorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RaptorRetriever()
    return _retriever


def _retrieve_and_format(query: str, top_k: int = 5):
    """Retrieve chunks and build citations."""
    retriever = _get_retriever()
    results = retriever.retrieve(query=query, top_k=top_k, include_tree_context=True)
    chunks = []
    citations = []
    seen = set()
    for r in results:
        ctx = r.get("tree_context", {})
        chunk = {
            "arxiv_id": r.get("arxiv_id", ""),
            "chunk_index": r.get("chunk_index", 0),
            "chunk_text": r.get("text", ""),
            "section_num": ctx.get("section_num", ""),
            "section_title": ctx.get("section_title", ""),
            "section_summary": ctx.get("section_summary", ""),
            "topic": ctx.get("topic", ""),
            "topic_summary": ctx.get("topic_summary", ""),
            "paper_title": ctx.get("paper_title", ""),
        }
        chunks.append(chunk)

        key = (chunk["arxiv_id"], chunk["section_title"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "arxiv_id": chunk["arxiv_id"],
                "paper_title": chunk["paper_title"],
                "section": chunk["section_title"],
                "topic": chunk["topic"],
                "excerpt": chunk["chunk_text"][:200] + "...",
            })
    return chunks, citations


def _format_citations_md(citations: list) -> str:
    """Format citations as Markdown for the citations panel."""
    if not citations:
        return "*No citations for this response.*"
    lines = ["### References\n"]
    for i, c in enumerate(citations, 1):
        paper = c.get("paper_title", "Unknown")
        arxiv = c.get("arxiv_id", "")
        section = c.get("section", "")
        topic = c.get("topic", "")
        excerpt = c.get("excerpt", "")
        lines.append(f"**[{i}] {paper}**")
        if arxiv:
            lines.append(f"  - arXiv: `{arxiv}`")
        if topic:
            lines.append(f"  - Topic: {topic}")
        if section:
            lines.append(f"  - Section: {section}")
        if excerpt:
            lines.append(f"  - Excerpt: _{excerpt}_")
        lines.append("")
    return "\n".join(lines)


def _format_session_list() -> list:
    """Get session list as dropdown choices."""
    sessions = session_manager.list_sessions()
    if not sessions:
        return []
    return [
        (f"{s['session_id']} ({s['message_count']} msgs)", s["session_id"])
        for s in sessions
    ]


# ============================================================
# Core chat function — called by Gradio
# ============================================================

def chat_fn(
    message: str,
    chat_history: list,
    session_id: str,
    task: str,
    model: str,
    top_k: int,
):
    """
    Process a user message through the full pipeline:
    1. Get/create session
    2. Store user message
    3. Retrieve context
    4. Build prompt with chat history
    5. Call LLM
    6. Store assistant response
    7. Return updated chat + citations
    """
    if not message.strip():
        return chat_history, "", session_id, "*Type a message to start chatting.*"

    # Map display names to internal values
    task_map = {
        "Q&A": "qa",
        "Summarize": "summarize",
        "Compare": "compare",
        "Explain": "explain",
    }
    task_key = task_map.get(task, "qa")

    model_map = {
        "Mistral (Local)": "mistral",
        "Groq Llama 3.3 (Cloud)": "groq-llama",
    }
    model_key = model_map.get(model, "mistral")

    # 1. Get or create session
    session = session_manager.get_or_create(session_id if session_id else None)
    current_session_id = session.session_id

    # 2. Store user message
    session.add_message(role="user", content=message)

    # 3. Retrieve context
    try:
        chunks, citations = _retrieve_and_format(message, top_k=int(top_k))
    except Exception as e:
        error_msg = f"Retrieval error: {str(e)}"
        logger.error(error_msg)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"Error during retrieval: {str(e)}"})
        return chat_history, "", current_session_id, error_msg

    # 4. Get prior history for LLM (exclude the message we just added)
    llm_history = session.get_llm_history(max_turns=10)
    if llm_history and llm_history[-1]["role"] == "user":
        llm_history = llm_history[:-1]

    # 5. Call LLM
    try:
        messages = build_messages(
            chunks, message, task=task_key, chat_history=llm_history or None
        )
        answer = run_llm_messages(messages, model=model_key, task=task_key)
    except Exception as e:
        error_msg = f"LLM error: {str(e)}"
        logger.error(error_msg)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"Error calling LLM: {str(e)}"})
        return chat_history, "", current_session_id, error_msg

    # 6. Store assistant response
    session.add_message(role="assistant", content=answer, citations=citations)

    # 7. Update Gradio chat history
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    citations_md = _format_citations_md(citations)

    return chat_history, "", current_session_id, citations_md


def submit_feedback_fn(
    feedback_type: str,
    chat_history: list,
    session_id: str,
    task: str,
    model: str,
    correction_text: str,
):
    """
    Submit feedback on the last assistant response.
    Extracts the last Q/A pair from chat history and stores feedback.
    """
    if not chat_history or len(chat_history) < 2:
        return "*No answer to give feedback on yet.*"

    # Find last user question and assistant answer
    last_answer = None
    last_question = None
    for msg in reversed(chat_history):
        if msg["role"] == "assistant" and last_answer is None:
            last_answer = msg["content"]
        elif msg["role"] == "user" and last_answer is not None:
            last_question = msg["content"]
            break

    if not last_question or not last_answer:
        return "*No complete Q&A pair found.*"

    task_map = {"Q&A": "qa", "Summarize": "summarize", "Compare": "compare", "Explain": "explain"}
    model_map = {"Mistral (Local)": "mistral", "Groq Llama 3.3 (Cloud)": "groq-llama"}

    # Get citations from the session's last assistant message
    citations = []
    session = session_manager.get_session(session_id)
    if session:
        for msg in reversed(session.history):
            if msg["role"] == "assistant" and msg.get("citations"):
                citations = msg["citations"]
                break

    try:
        feedback_store.submit(
            session_id=session_id or "unknown",
            question=last_question,
            answer=last_answer,
            feedback_type=feedback_type,
            correction=correction_text if feedback_type == "correction" else "",
            model_used=model_map.get(model, "mistral"),
            task=task_map.get(task, "qa"),
            citations=citations,
        )
        label = feedback_type.replace("_", " ").title()
        return f"**Feedback recorded:** {label}. Thank you!"
    except Exception as e:
        return f"**Error saving feedback:** {str(e)}"


def new_session_fn():
    """Create a fresh session and clear chat."""
    session = session_manager.create_session()
    return [], session.session_id, "*New session started.*", _format_session_list()


def load_session_fn(session_choice):
    """Load an existing session's chat history."""
    if not session_choice:
        return [], "", "*No session selected.*"
    session = session_manager.get_session(session_choice)
    if not session:
        return [], "", "*Session not found.*"
    # Rebuild Gradio chat history from session
    chat_history = []
    for msg in session.history:
        chat_history.append({"role": msg["role"], "content": msg["content"]})
    return chat_history, session.session_id, _format_citations_md(
        session.history[-1].get("citations", []) if session.history else []
    )


# ============================================================
# Build Gradio Interface
# ============================================================

def create_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAPTOR Research Assistant",
        theme=gr.themes.Soft(),
        css=".citation-panel { max-height: 400px; overflow-y: auto; }"
    ) as demo:
        gr.Markdown(
            "# RAPTOR Research Assistant\n"
            "Ask questions about ML/DL research papers. "
            "The system retrieves relevant context from 200+ indexed papers "
            "using RAPTOR hierarchical search and generates answers with citations."
        )

        # Hidden state for session ID
        session_state = gr.State("")

        with gr.Row():
            # --- Left column: Chat ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    type="messages",
                    show_copy_button=True,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about research papers...",
                        label="Message",
                        scale=4,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # --- Right column: Controls + Citations ---
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                task_dropdown = gr.Dropdown(
                    choices=["Q&A", "Summarize", "Compare", "Explain"],
                    value="Q&A",
                    label="Task Type",
                )
                model_dropdown = gr.Dropdown(
                    choices=["Mistral (Local)", "Groq Llama 3.3 (Cloud)"],
                    value="Mistral (Local)",
                    label="Model",
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1, label="Top-K Results"
                )

                gr.Markdown("### Session")
                new_session_btn = gr.Button("New Session", variant="secondary")
                session_dropdown = gr.Dropdown(
                    choices=_format_session_list(),
                    label="Load Session",
                    interactive=True,
                )
                load_session_btn = gr.Button("Load", size="sm")

                gr.Markdown("### Feedback")
                with gr.Row():
                    helpful_btn = gr.Button("Helpful", size="sm", variant="primary")
                    incorrect_btn = gr.Button("Incorrect", size="sm")
                with gr.Row():
                    hallucination_btn = gr.Button("Hallucination", size="sm")
                    correction_btn = gr.Button("Correction", size="sm")
                correction_input = gr.Textbox(
                    placeholder="Type corrected answer here...",
                    label="Correction Text",
                    lines=2,
                    visible=True,
                )
                feedback_display = gr.Markdown(
                    value="*Rate the last response using the buttons above.*"
                )

                gr.Markdown("### Citations")
                citations_display = gr.Markdown(
                    value="*Citations will appear here after each response.*",
                    elem_classes=["citation-panel"],
                )

        # --- Event handlers ---

        # Send message (button or Enter key)
        send_inputs = [msg_input, chatbot, session_state, task_dropdown, model_dropdown, top_k_slider]
        send_outputs = [chatbot, msg_input, session_state, citations_display]

        send_btn.click(
            fn=chat_fn,
            inputs=send_inputs,
            outputs=send_outputs,
        )
        msg_input.submit(
            fn=chat_fn,
            inputs=send_inputs,
            outputs=send_outputs,
        )

        # New session
        new_session_btn.click(
            fn=new_session_fn,
            outputs=[chatbot, session_state, citations_display, session_dropdown],
        )

        # Load existing session
        load_session_btn.click(
            fn=load_session_fn,
            inputs=[session_dropdown],
            outputs=[chatbot, session_state, citations_display],
        )

        # Feedback buttons
        feedback_inputs = [chatbot, session_state, task_dropdown, model_dropdown, correction_input]

        helpful_btn.click(
            fn=lambda *args: submit_feedback_fn("helpful", *args),
            inputs=feedback_inputs,
            outputs=[feedback_display],
        )
        incorrect_btn.click(
            fn=lambda *args: submit_feedback_fn("incorrect", *args),
            inputs=feedback_inputs,
            outputs=[feedback_display],
        )
        hallucination_btn.click(
            fn=lambda *args: submit_feedback_fn("hallucination", *args),
            inputs=feedback_inputs,
            outputs=[feedback_display],
        )
        correction_btn.click(
            fn=lambda *args: submit_feedback_fn("correction", *args),
            inputs=feedback_inputs,
            outputs=[feedback_display],
        )

    return demo


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

    logging.basicConfig(level=logging.INFO)
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
