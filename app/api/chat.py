"""
Chat API — Session-aware chat endpoints.

Endpoints:
  POST   /chat             — Send a message, get a response (creates session if needed)
  POST   /chat/session     — Create a new empty session
  GET    /chat/session/{id} — Get full session history
  GET    /chat/sessions    — List all active sessions
  DELETE /chat/session/{id} — Delete a session
"""
import os
import sys
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.session import session_manager
from app.core.retrieval import RaptorRetriever
from app.core.prompt_builder import build_messages
from app.core.llm_client import run_llm_messages

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Shared retriever instance
_retriever = None


def _get_retriever() -> RaptorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RaptorRetriever()
    return _retriever


def _retrieve_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Run hybrid retrieval and format chunks for prompt builder."""
    retriever = _get_retriever()
    results = retriever.retrieve(query=query, top_k=top_k, include_tree_context=True)
    chunks = []
    for r in results:
        ctx = r.get("tree_context", {})
        chunks.append({
            "arxiv_id": r.get("arxiv_id", ""),
            "chunk_index": r.get("chunk_index", 0),
            "chunk_text": r.get("text", ""),
            "section_num": ctx.get("section_num", ""),
            "section_title": ctx.get("section_title", ""),
            "section_summary": ctx.get("section_summary", ""),
            "topic": ctx.get("topic", ""),
            "topic_summary": ctx.get("topic_summary", ""),
            "paper_title": ctx.get("paper_title", ""),
        })
    return chunks


def _build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract citation info from retrieved chunks."""
    citations = []
    seen = set()
    for c in chunks:
        key = (c.get("arxiv_id", ""), c.get("section_title", ""))
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "arxiv_id": c.get("arxiv_id", ""),
            "paper_title": c.get("paper_title", ""),
            "section": c.get("section_title", ""),
            "topic": c.get("topic", ""),
            "excerpt": c.get("chunk_text", "")[:200] + "...",
        })
    return citations


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    task: str = "qa"
    model: str = "mistral"
    top_k: int = 5

class Citation(BaseModel):
    arxiv_id: str = ""
    paper_title: str = ""
    section: str = ""
    topic: str = ""
    excerpt: str = ""

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Citation]
    model_used: str

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    papers_referenced: List[str]

class SessionDetail(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    papers_referenced: List[str]
    history: List[Dict[str, Any]]


# --- Endpoints ---

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint: send a message, get an answer with citations.

    - If session_id is provided, continues that session (with chat history).
    - If session_id is None, creates a new session automatically.
    - Chat history is passed to the LLM for context-aware multi-turn conversation.
    """
    # 1. Get or create session
    session = session_manager.get_or_create(req.session_id)

    # 2. Store user message
    session.add_message(role="user", content=req.message)

    # 3. Retrieve relevant chunks
    chunks = _retrieve_chunks(req.message, req.top_k)

    # 4. Build citations from retrieved chunks
    citations = _build_citations(chunks)

    # 5. Get chat history for context (excluding the message we just added)
    chat_history = session.get_llm_history(max_turns=10)
    # Remove the last entry (the user message we just added) since it goes in the prompt directly
    if chat_history and chat_history[-1]["role"] == "user":
        chat_history = chat_history[:-1]

    # 6. Build messages with history and send to LLM
    messages = build_messages(
        chunks, req.message, task=req.task, chat_history=chat_history or None
    )
    answer = run_llm_messages(messages, model=req.model, task=req.task)

    # 7. Store assistant response with citations
    session.add_message(role="assistant", content=answer, citations=citations)

    return ChatResponse(
        session_id=session.session_id,
        answer=answer,
        citations=[Citation(**c) for c in citations],
        model_used=req.model,
    )


@router.post("/session", response_model=SessionInfo)
def create_session():
    """Create a new empty chat session."""
    session = session_manager.create_session()
    return SessionInfo(**session.summary())


@router.get("/session/{session_id}", response_model=SessionDetail)
def get_session(session_id: str):
    """Get full history for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetail(**session.to_dict())


@router.get("/sessions", response_model=List[SessionInfo])
def list_sessions():
    """List all active sessions."""
    return [SessionInfo(**s) for s in session_manager.list_sessions()]


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and its history."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}
