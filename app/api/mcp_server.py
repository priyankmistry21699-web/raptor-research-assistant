"""
RAPTOR Research Assistant — Backend Server (FastAPI)

Production-ready backend that orchestrates all system components:
  - Hybrid retrieval (vector search + RAPTOR tree traversal)
  - Prompt construction and LLM inference (multi-model)
  - Session-aware chat with citations
  - User feedback collection and preference dataset creation
  - DPO fine-tuning and continuous learning loop
  - RAGAS evaluation of RAG quality
  - System health, status, and configuration endpoints

Routers:
  /chat       — session-aware Q&A with citations (5 routes)
  /retrieve   — hybrid vector + tree retrieval, paper browsing (4 routes)
  /feedback   — feedback collection and querying (5 routes)
  /train      — preferences, fine-tuning, learning loop (14 routes)
  /eval       — RAGAS evaluation and model comparison (6 routes)

Pipeline endpoints (top-level):
  POST /prompt         — retrieve + build prompt
  POST /llm            — retrieve + prompt + LLM answer (full pipeline)
  GET  /llm/models     — list available models (incl. fine-tuned)
  GET  /llm/health     — check if current model is responding

System endpoints:
  GET  /               — API overview and route summary
  GET  /health         — system health check (LLM, ChromaDB, trees)
  GET  /status         — full system status (papers, chunks, sessions, models)
  GET  /config         — current configuration summary

Launch:
  uvicorn app.api.mcp_server:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import time
import logging
import glob
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

logger = logging.getLogger(__name__)

from app.core.retrieval import RaptorRetriever
from app.core.prompt_builder import build_prompt, build_messages
from app.core.llm_client import (
    run_llm, run_llm_messages, list_available_models,
    check_model_health, get_active_model,
)
from app.core.session import session_manager
from app.core.feedback import feedback_store
from app.core.vector_db import VectorDB

from app.api.chat import router as chat_router
from app.api.retrieve import router as retrieve_router
from app.api.feedback import router as feedback_router
from app.api.train import router as train_router
from app.api.eval import router as eval_router

# --- Config loading ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')


def _load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


# --- Lifespan: startup / shutdown ---
_startup_time: float = 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    cfg = _load_config()
    server_cfg = cfg.get("server", {})
    logger.info(
        "RAPTOR Backend starting — host=%s port=%s",
        server_cfg.get("host", "0.0.0.0"),
        server_cfg.get("port", 8000),
    )
    logger.info("Active model: %s", get_active_model())
    yield
    logger.info("RAPTOR Backend shutting down")


# --- App ---
app = FastAPI(
    title="RAPTOR Research Assistant",
    description="Backend API for the RAPTOR Research Assistant — "
                "hybrid retrieval, LLM inference, feedback, fine-tuning, and evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS ---
cfg = _load_config()
server_cfg = cfg.get("server", {})
cors_origins = server_cfg.get("cors_origins", ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request logging middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(
        "%s %s → %s (%.0fms)",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


# --- Global exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# --- Routers ---
app.include_router(chat_router)
app.include_router(retrieve_router)
app.include_router(feedback_router)
app.include_router(train_router)
app.include_router(eval_router)

# Lazy-init shared retriever for pipeline endpoints
_retriever: Optional[RaptorRetriever] = None


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


# ============================================================
#  System endpoints
# ============================================================

@app.get("/", tags=["system"])
def root():
    """API overview — lists all available route groups."""
    return {
        "name": "RAPTOR Research Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "routes": {
            "/chat": "Session-aware Q&A with citations",
            "/retrieve": "Hybrid vector + tree retrieval, paper browsing",
            "/feedback": "Feedback collection and querying",
            "/train": "Preferences, DPO fine-tuning, learning loop",
            "/eval": "RAGAS evaluation and model comparison",
            "/prompt": "Retrieve + build prompt (pipeline)",
            "/llm": "Full pipeline: retrieve → prompt → LLM answer",
            "/health": "System health check",
            "/status": "Full system status",
            "/config": "Current configuration",
        },
    }


@app.get("/health", tags=["system"])
def health_check():
    """
    System health check — verifies LLM connectivity, ChromaDB, and RAPTOR trees.
    Returns overall status and per-component details.
    """
    components = {}

    # LLM health
    try:
        llm_result = check_model_health(get_active_model())
        components["llm"] = {
            "status": llm_result.get("status", "unknown"),
            "model": llm_result.get("model", ""),
        }
    except Exception as e:
        components["llm"] = {"status": "error", "error": str(e)}

    # ChromaDB health
    try:
        db = VectorDB()
        count = db.count()
        components["chromadb"] = {"status": "ok", "chunks": count}
    except Exception as e:
        components["chromadb"] = {"status": "error", "error": str(e)}

    # RAPTOR trees
    tree_dir = os.path.join(BASE_DIR, 'data', 'raw', 'paper_trees')
    try:
        tree_files = glob.glob(os.path.join(tree_dir, '*_tree.gpickle'))
        components["raptor_trees"] = {"status": "ok", "papers": len(tree_files)}
    except Exception as e:
        components["raptor_trees"] = {"status": "error", "error": str(e)}

    overall = "healthy" if all(
        c.get("status") == "ok" for c in components.values()
    ) else "degraded"

    return {"status": overall, "components": components}


@app.get("/status", tags=["system"])
def system_status():
    """
    Full system status — papers, chunks, sessions, feedback, models, uptime.
    """
    # Papers
    tree_dir = os.path.join(BASE_DIR, 'data', 'raw', 'paper_trees')
    tree_files = glob.glob(os.path.join(tree_dir, '*_tree.gpickle'))
    paper_count = len(tree_files)

    # Chunks in ChromaDB
    try:
        db = VectorDB()
        chunk_count = db.count()
    except Exception:
        chunk_count = -1

    # Sessions
    sessions = session_manager.list_sessions()
    session_count = len(sessions)

    # Feedback
    feedback_count = feedback_store.count()

    # Models
    models = list_available_models()
    active_model = get_active_model()
    finetuned_count = sum(1 for v in models.values() if v.get("is_finetuned"))

    # Uptime
    uptime_seconds = time.time() - _startup_time if _startup_time else 0

    return {
        "papers": paper_count,
        "chunks_in_db": chunk_count,
        "active_sessions": session_count,
        "feedback_entries": feedback_count,
        "models": {
            "total": len(models),
            "finetuned": finetuned_count,
            "active": active_model,
        },
        "uptime_seconds": round(uptime_seconds, 1),
    }


@app.get("/config", tags=["system"])
def get_config():
    """Return current system configuration (sensitive keys redacted)."""
    config = _load_config()
    # Redact API keys
    for model_cfg in config.get("llm", {}).get("models", {}).values():
        if "api_key" in model_cfg:
            model_cfg["api_key"] = "***"
    return config


# ============================================================
#  Pipeline endpoints (MCP-style: retrieve → prompt → LLM)
# ============================================================

class PromptRequest(BaseModel):
    query: str
    top_k: int = 5
    task: str = "qa"
    chat_history: Optional[List[Dict[str, str]]] = None

class PromptResult(BaseModel):
    prompt: str

class LLMRequest(BaseModel):
    query: str
    top_k: int = 5
    task: str = "qa"
    model: str = "auto"
    chat_history: Optional[List[Dict[str, str]]] = None

class LLMResult(BaseModel):
    answer: str
    prompt: str
    model_used: str = ""


@app.post("/prompt", response_model=PromptResult, tags=["pipeline"])
def prompt(req: PromptRequest):
    """Retrieve context and build a prompt for LLM."""
    chunks = _retrieve_chunks(req.query, req.top_k)
    prompt_str = build_prompt(chunks, req.query, task=req.task, chat_history=req.chat_history)
    return PromptResult(prompt=prompt_str)


@app.post("/llm", response_model=LLMResult, tags=["pipeline"])
def llm(req: LLMRequest):
    """
    Full pipeline: retrieve → prompt → LLM answer.

    Supports 4 task types:
      - qa: Answer a question from retrieved context
      - summarize: Summarize papers/topics
      - compare: Compare findings across papers
      - explain: Explain a concept step by step

    Model routing:
      - model="auto" → picks the best available (prefers fine-tuned)
      - model="mistral" → local Ollama Mistral
      - model="groq-llama" → Groq cloud Llama 3.3
      - model="finetuned-xxx" → fine-tuned LoRA adapter
    """
    chunks = _retrieve_chunks(req.query, req.top_k)
    messages = build_messages(chunks, req.query, task=req.task, chat_history=req.chat_history)
    model_name = req.model if req.model != "auto" else get_active_model()
    answer = run_llm_messages(messages, model=model_name, task=req.task)
    return LLMResult(answer=answer, prompt=messages[-1]["content"], model_used=model_name)


@app.get("/llm/models", tags=["pipeline"])
def models():
    """List all configured models in the registry (base + fine-tuned)."""
    return list_available_models()


@app.get("/llm/health", tags=["pipeline"])
def llm_health():
    """Check if the active LLM model is responding."""
    return check_model_health(get_active_model())
