"""
MCP Server — orchestrates hybrid retrieval, prompt construction, and LLM inference.

Endpoints:
  POST /retrieve    — hybrid vector search + RAPTOR tree context
  POST /prompt      — retrieve + build prompt
  POST /llm         — retrieve + prompt + LLM answer (full pipeline)
  GET  /llm/models  — list available models
  GET  /llm/health  — check if current model is responding
"""
import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

from app.core.retrieval import RaptorRetriever
from app.core.prompt_builder import build_prompt, build_messages
from app.core.llm_client import run_llm, run_llm_messages, list_available_models, check_model_health
from app.api.chat import router as chat_router
from app.api.feedback import router as feedback_router

app = FastAPI(title="RAPTOR MCP Server")
app.include_router(chat_router)
app.include_router(feedback_router)
retriever = RaptorRetriever()


# --- Shared helper ---

def _retrieve_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Run hybrid retrieval and format chunks for prompt builder."""
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


# --- Request/Response models ---

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class ChunkResult(BaseModel):
    arxiv_id: str
    chunk_index: int
    chunk_text: str
    section_num: str = ""
    section_title: str = ""
    section_summary: str = ""
    topic: str = ""
    topic_summary: str = ""
    paper_title: str = ""

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
    model: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

class LLMResult(BaseModel):
    answer: str
    prompt: str
    model_used: str = ""


# --- Endpoints ---

@app.post("/retrieve", response_model=List[ChunkResult])
def retrieve(req: RetrieveRequest):
    """Hybrid retrieval: vector search + RAPTOR tree traversal."""
    chunks = _retrieve_chunks(req.query, req.top_k)
    return [ChunkResult(**c) for c in chunks]


@app.post("/prompt", response_model=PromptResult)
def prompt(req: PromptRequest):
    """Retrieve context and build a prompt for LLM."""
    chunks = _retrieve_chunks(req.query, req.top_k)
    prompt_str = build_prompt(chunks, req.query, task=req.task, chat_history=req.chat_history)
    return PromptResult(prompt=prompt_str)


@app.post("/llm", response_model=LLMResult)
def llm(req: LLMRequest):
    """
    Full pipeline: retrieve → prompt → LLM answer.

    Supports 4 task types:
      - qa: Answer a question from retrieved context
      - summarize: Summarize papers/topics
      - compare: Compare findings across papers
      - explain: Explain a concept step by step

    Model routing:
      - model=None → uses default (mistral:latest on Ollama)
      - model="mistral" → local Ollama Mistral
      - model="groq-llama" → Groq cloud Llama 3.3
      - model="any-model-name" → sent directly to default API URL
    """
    chunks = _retrieve_chunks(req.query, req.top_k)
    messages = build_messages(chunks, req.query, task=req.task, chat_history=req.chat_history)
    model_name = req.model or "mistral"
    answer = run_llm_messages(messages, model=model_name, task=req.task)
    return LLMResult(answer=answer, prompt=messages[-1]["content"], model_used=model_name)


@app.get("/llm/models")
def models():
    """List all configured models in the registry."""
    return list_available_models()


@app.get("/llm/health")
def health():
    """Check if the default LLM (Mistral on Ollama) is responding."""
    return check_model_health("mistral")
