"""
Gradio UI — Full frontend for RAPTOR Research Assistant.

Tabs:
  1. Chat       — Multi-turn Q&A with citations, session & feedback support
  2. Papers     — Browse 200+ papers, view RAPTOR tree (topics → sections → chunks)
  3. Upload     — Add new papers via arXiv ID or direct PDF upload
  4. Dashboard  — System status, feedback stats, model info

Launch:
  python -m app.frontend.ui
"""
import os
import sys
import json
import re
import logging
import glob
import pickle
import tempfile
import threading

import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.session import session_manager, Session
from app.core.retrieval import RaptorRetriever
from app.core.prompt_builder import build_messages
from app.core.llm_client import run_llm_messages, get_active_model, list_available_models
from app.core.feedback import feedback_store
from app.core.raptor_index import (
    load_tree, save_tree, get_paper_info, get_topics,
    get_sections, get_chunks, list_all_papers, get_tree_structure,
)
from app.core.embedding import EmbeddingModel
from app.core.vector_db import VectorDB
from app.core.ingestion import download_pdf, RAW_DATA_DIR

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Shared retriever (lazy init)
_retriever = None


def _get_retriever() -> RaptorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RaptorRetriever()
    return _retriever


# ============================================================
#  Tab 1: Chat — helpers
# ============================================================

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
    sessions = session_manager.list_sessions()
    if not sessions:
        return []
    return [
        (f"{s['session_id']} ({s['message_count']} msgs)", s["session_id"])
        for s in sessions
    ]


def chat_fn(message, chat_history, session_id, task, model, top_k):
    if not message.strip():
        return chat_history, "", session_id, "*Type a message to start chatting.*"

    task_map = {"Q&A": "qa", "Summarize": "summarize", "Compare": "compare", "Explain": "explain"}
    task_key = task_map.get(task, "qa")

    model_map = {
        "Auto (Best Available)": "auto",
        "Mistral (Local)": "mistral",
        "Groq Llama 3.3 (Cloud)": "groq-llama",
    }
    model_key = model_map.get(model, "auto")
    if model_key == "auto":
        model_key = get_active_model()

    session = session_manager.get_or_create(session_id if session_id else None)
    current_session_id = session.session_id
    session.add_message(role="user", content=message)

    try:
        chunks, citations = _retrieve_and_format(message, top_k=int(top_k))
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"Error during retrieval: {e}"})
        return chat_history, "", current_session_id, f"Retrieval error: {e}"

    llm_history = session.get_llm_history(max_turns=10)
    if llm_history and llm_history[-1]["role"] == "user":
        llm_history = llm_history[:-1]

    try:
        messages = build_messages(chunks, message, task=task_key, chat_history=llm_history or None)
        answer = run_llm_messages(messages, model=model_key, task=task_key)
    except Exception as e:
        logger.error("LLM error: %s", e)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"Error calling LLM: {e}"})
        return chat_history, "", current_session_id, f"LLM error: {e}"

    session.add_message(role="assistant", content=answer, citations=citations)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, "", current_session_id, _format_citations_md(citations)


def submit_feedback_fn(feedback_type, chat_history, session_id, task, model, correction_text):
    if not chat_history or len(chat_history) < 2:
        return "*No answer to give feedback on yet.*"

    last_answer = last_question = None
    for msg in reversed(chat_history):
        if msg["role"] == "assistant" and last_answer is None:
            last_answer = msg["content"]
        elif msg["role"] == "user" and last_answer is not None:
            last_question = msg["content"]
            break

    if not last_question or not last_answer:
        return "*No complete Q&A pair found.*"

    task_map = {"Q&A": "qa", "Summarize": "summarize", "Compare": "compare", "Explain": "explain"}
    model_map = {"Auto (Best Available)": "auto", "Mistral (Local)": "mistral", "Groq Llama 3.3 (Cloud)": "groq-llama"}

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
        return f"**Error saving feedback:** {e}"


def new_session_fn():
    session = session_manager.create_session()
    return [], session.session_id, "*New session started.*", _format_session_list()


def load_session_fn(session_choice):
    if not session_choice:
        return [], "", "*No session selected.*"
    session = session_manager.get_session(session_choice)
    if not session:
        return [], "", "*Session not found.*"
    chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in session.history]
    return chat_history, session.session_id, _format_citations_md(
        session.history[-1].get("citations", []) if session.history else []
    )


# ============================================================
#  Tab 2: Paper Browser — helpers
# ============================================================

def _load_metadata() -> dict:
    """Load paper metadata into a dict keyed by arxiv_id."""
    meta_path = os.path.join(BASE_DIR, 'data', 'raw', 'papers_metadata_with_id.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(BASE_DIR, 'data', 'raw', 'papers_metadata.json')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    return {p.get("arxiv_id", ""): p for p in papers if p.get("arxiv_id")}


def list_papers_fn():
    """Return a Markdown table of all papers."""
    meta = _load_metadata()
    papers = list_all_papers()
    if not papers:
        return "*No papers found.*"

    lines = ["| # | arXiv ID | Title | Category | Date |",
             "|---|----------|-------|----------|------|"]
    for i, pid in enumerate(sorted(papers), 1):
        m = meta.get(pid, {})
        title = m.get("title", "–")[:60]
        cat = m.get("category", "–")
        date = m.get("published_date", "–")
        lines.append(f"| {i} | `{pid}` | {title} | {cat} | {date} |")
    return "\n".join(lines)


def paper_overview_fn(arxiv_id: str):
    """Show a paper's RAPTOR tree overview."""
    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        return "*Enter an arXiv ID above.*", []

    G = load_tree(arxiv_id)
    if G is None:
        return f"*Paper `{arxiv_id}` not found in RAPTOR trees.*", []

    info = get_paper_info(G)
    topics = get_topics(G)
    meta = _load_metadata().get(arxiv_id, {})

    lines = [f"## {meta.get('title', info.get('title', arxiv_id))}"]
    lines.append(f"**arXiv:** `{arxiv_id}`")
    if meta.get("authors"):
        authors = meta["authors"]
        if isinstance(authors, list):
            authors = ", ".join(authors[:5])
            if len(meta["authors"]) > 5:
                authors += f" ... (+{len(meta['authors']) - 5})"
        lines.append(f"**Authors:** {authors}")
    if meta.get("category"):
        lines.append(f"**Category:** {meta['category']}")
    if meta.get("published_date"):
        lines.append(f"**Published:** {meta['published_date']}")
    if meta.get("abstract"):
        lines.append(f"\n**Abstract:** {meta['abstract'][:500]}{'...' if len(meta.get('abstract', '')) > 500 else ''}")

    lines.append(f"\n### RAPTOR Tree — {len(topics)} topics\n")

    topic_choices = []
    for t in topics:
        title = t.get("title", "Untitled")
        summary = t.get("summary", "")
        scount = t.get("section_count", 0)
        lines.append(f"**{title}** ({scount} sections)")
        if summary:
            lines.append(f"> {summary[:200]}")
        lines.append("")
        topic_choices.append(title)

    return "\n".join(lines), gr.update(choices=topic_choices, value=None)


def topic_sections_fn(arxiv_id: str, topic_title: str):
    """Show sections under a topic."""
    arxiv_id = arxiv_id.strip()
    if not arxiv_id or not topic_title:
        return "*Select a topic first.*", []

    G = load_tree(arxiv_id)
    if G is None:
        return "*Paper not found.*", []

    # Find the topic node
    topic_node = None
    for node in G.successors("root"):
        n = G.nodes[node]
        if n.get("type") == "topic" and n.get("title") == topic_title:
            topic_node = node
            break

    if topic_node is None:
        return f"*Topic '{topic_title}' not found.*", []

    sections = get_sections(G, topic_node)
    if not sections:
        return "*No sections found under this topic.*", []

    lines = [f"### Sections under: {topic_title}\n"]
    section_choices = []
    for s in sections:
        num = s.get("section_num", "")
        title = s.get("title", "Untitled")
        summary = s.get("summary", "")
        ccount = s.get("chunk_count", 0)
        label = f"{num} {title}".strip()
        lines.append(f"**{label}** ({ccount} chunks)")
        if summary:
            lines.append(f"> {summary[:300]}")
        lines.append("")
        section_choices.append(label)

    return "\n".join(lines), gr.update(choices=section_choices, value=None)


def section_chunks_fn(arxiv_id: str, section_label: str):
    """Show chunks in a section."""
    arxiv_id = arxiv_id.strip()
    if not arxiv_id or not section_label:
        return "*Select a section first.*"

    G = load_tree(arxiv_id)
    if G is None:
        return "*Paper not found.*"

    # Find section node by matching section_num + title
    target_node = None
    for node in G.nodes:
        n = G.nodes[node]
        if n.get("type") == "section":
            label = f"{n.get('section_num', '')} {n.get('title', '')}".strip()
            if label == section_label:
                target_node = node
                break

    if target_node is None:
        return f"*Section '{section_label}' not found.*"

    chunks = get_chunks(G, target_node)
    if not chunks:
        return "*No chunks in this section.*"

    lines = [f"### Chunks in: {section_label}\n"]
    for c in chunks[:20]:  # Limit display
        idx = c.get("chunk_index", "?")
        text = c.get("text", "")[:500]
        lines.append(f"**Chunk {idx}:**\n{text}\n")
        lines.append("---")

    if len(chunks) > 20:
        lines.append(f"\n*... and {len(chunks) - 20} more chunks.*")

    return "\n".join(lines)


# ============================================================
#  Tab 3: Upload Paper — helpers
# ============================================================

def _ingest_paper_from_arxiv(arxiv_id: str, progress_fn=None) -> str:
    """Full pipeline: fetch metadata → download PDF → chunk → embed → store → build tree."""
    import arxiv as arxiv_lib
    import fitz

    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        return "Error: Please enter an arXiv ID."

    # Check if already exists
    existing = list_all_papers()
    if arxiv_id in existing:
        return f"Paper `{arxiv_id}` is already in the system."

    steps = []

    # Step 1: Fetch metadata from arXiv
    steps.append("Fetching metadata from arXiv...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        search = arxiv_lib.Search(id_list=[arxiv_id])
        result = next(search.results())
        metadata = {
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "category": result.primary_category,
            "pdf_url": result.pdf_url,
            "published_date": result.published.strftime("%Y-%m-%d"),
            "arxiv_id": arxiv_id,
        }
        steps.append(f"  Title: {metadata['title']}")
    except Exception as e:
        return f"Error fetching arXiv metadata: {e}"

    # Step 2: Download PDF
    steps.append("Downloading PDF...")
    if progress_fn:
        progress_fn("\n".join(steps))
    pdf_path = os.path.join(RAW_DATA_DIR, f"{arxiv_id}.pdf")
    try:
        download_pdf(metadata["pdf_url"], pdf_path)
        steps.append(f"  Saved to {pdf_path}")
    except Exception as e:
        return f"Error downloading PDF: {e}"

    # Step 3: Extract text and chunk
    steps.append("Extracting text and chunking...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
        chunk_size, overlap = 400, 50
        chunks = []
        chunk_idx = 0
        for para in paragraphs:
            tokens = para.split()
            i = 0
            while i < len(tokens):
                chunk_text = " ".join(tokens[i:i + chunk_size])
                chunks.append({"arxiv_id": arxiv_id, "chunk_index": chunk_idx, "text": chunk_text})
                chunk_idx += 1
                i += chunk_size - overlap
        steps.append(f"  Extracted {len(chunks)} chunks")
    except Exception as e:
        return f"Error processing PDF: {e}"

    if not chunks:
        return "Error: No text could be extracted from the PDF."

    # Step 4: Generate embeddings
    steps.append("Generating embeddings...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        embedder = EmbeddingModel()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = [embedder.encode(t) for t in chunk_texts]
        steps.append(f"  Generated {len(embeddings)} embeddings")
    except Exception as e:
        return f"Error generating embeddings: {e}"

    # Step 5: Store in ChromaDB
    steps.append("Storing in ChromaDB...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        db = VectorDB()
        ids = [f"{arxiv_id}_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {"arxiv_id": arxiv_id, "chunk_index": c["chunk_index"],
             "paper_title": metadata["title"], "category": metadata["category"]}
            for c in chunks
        ]
        db.upsert_chunks(ids=ids, embeddings=embeddings, documents=chunk_texts, metadatas=metadatas)
        steps.append(f"  Stored {len(ids)} chunks in ChromaDB")
    except Exception as e:
        return f"Error storing in ChromaDB: {e}"

    # Step 6: Build RAPTOR tree (basic: root → single topic → sections by page → chunks)
    steps.append("Building RAPTOR tree...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_node("root", type="paper", title=metadata["title"],
                    arxiv_id=arxiv_id, metadata=metadata)

        # Create a single topic from the category
        topic_id = "topic_0"
        G.add_node(topic_id, type="topic", title=metadata.get("category", "General"))
        G.add_edge("root", topic_id)

        # Group chunks into pseudo-sections of ~10 chunks each
        section_size = 10
        section_idx = 0
        for start in range(0, len(chunks), section_size):
            batch = chunks[start:start + section_size]
            sec_id = f"section_{section_idx}"
            G.add_node(sec_id, type="section", section_num=str(section_idx + 1),
                       title=f"Section {section_idx + 1}", summary="")
            G.add_edge(topic_id, sec_id)
            for c in batch:
                chunk_id = f"chunk_{c['chunk_index']}"
                G.add_node(chunk_id, type="chunk", chunk_index=c["chunk_index"], text=c["text"])
                G.add_edge(sec_id, chunk_id)
            section_idx += 1

        save_tree(arxiv_id, G)
        steps.append(f"  Built tree: 1 topic, {section_idx} sections, {len(chunks)} chunks")
    except Exception as e:
        return f"Error building tree: {e}"

    # Step 7: Update metadata file
    steps.append("Updating metadata...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        meta_path = os.path.join(BASE_DIR, 'data', 'raw', 'papers_metadata_with_id.json')
        existing_meta = []
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
        # Avoid duplicate
        if not any(p.get("arxiv_id") == arxiv_id for p in existing_meta):
            existing_meta.append(metadata)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(existing_meta, f, indent=2, ensure_ascii=False)
    except Exception as e:
        steps.append(f"  Warning: could not update metadata file: {e}")

    # Reset retriever so it picks up the new paper
    global _retriever
    _retriever = None

    steps.append(f"\n**Done!** Paper `{arxiv_id}` — *{metadata['title']}* ingested successfully.")
    steps.append(f"  - {len(chunks)} chunks embedded and stored")
    steps.append(f"  - RAPTOR tree built with {section_idx} sections")
    steps.append("  - Ready for querying in the Chat tab")

    return "\n".join(steps)


def _ingest_paper_from_pdf(pdf_file, arxiv_id_input: str, title_input: str, progress_fn=None) -> str:
    """Ingest a directly uploaded PDF."""
    import fitz

    if pdf_file is None:
        return "Error: No file uploaded."

    arxiv_id = arxiv_id_input.strip()
    if not arxiv_id:
        # Generate a pseudo-ID from the filename
        basename = os.path.basename(pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file))
        arxiv_id = re.sub(r'[^a-zA-Z0-9._-]', '_', basename.replace('.pdf', ''))

    title = title_input.strip() or arxiv_id

    existing = list_all_papers()
    if arxiv_id in existing:
        return f"Paper `{arxiv_id}` already exists in the system."

    steps = []

    # Step 1: Save PDF
    steps.append(f"Saving PDF as `{arxiv_id}.pdf`...")
    if progress_fn:
        progress_fn("\n".join(steps))
    pdf_path = os.path.join(RAW_DATA_DIR, f"{arxiv_id}.pdf")
    try:
        # Gradio gives us either a file path string or a tempfile object
        src = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
        import shutil
        shutil.copy2(src, pdf_path)
        steps.append(f"  Saved to {pdf_path}")
    except Exception as e:
        return f"Error saving PDF: {e}"

    # Step 2: Extract and chunk
    steps.append("Extracting text and chunking...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()

        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
        chunk_size, overlap = 400, 50
        chunks = []
        chunk_idx = 0
        for para in paragraphs:
            tokens = para.split()
            i = 0
            while i < len(tokens):
                chunk_text = " ".join(tokens[i:i + chunk_size])
                chunks.append({"arxiv_id": arxiv_id, "chunk_index": chunk_idx, "text": chunk_text})
                chunk_idx += 1
                i += chunk_size - overlap
        steps.append(f"  Extracted {len(chunks)} chunks")
    except Exception as e:
        return f"Error processing PDF: {e}"

    if not chunks:
        return "Error: No text extracted from the PDF."

    # Step 3–6: Embed → Store → Build tree → Update metadata
    metadata = {
        "title": title,
        "authors": [],
        "abstract": "",
        "category": "uploaded",
        "pdf_url": "",
        "published_date": "",
        "arxiv_id": arxiv_id,
    }

    steps.append("Generating embeddings...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        embedder = EmbeddingModel()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = [embedder.encode(t) for t in chunk_texts]
        steps.append(f"  Generated {len(embeddings)} embeddings")
    except Exception as e:
        return f"Error generating embeddings: {e}"

    steps.append("Storing in ChromaDB...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        db = VectorDB()
        ids = [f"{arxiv_id}_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {"arxiv_id": arxiv_id, "chunk_index": c["chunk_index"],
             "paper_title": title, "category": "uploaded"}
            for c in chunks
        ]
        db.upsert_chunks(ids=ids, embeddings=embeddings, documents=chunk_texts, metadatas=metadatas)
        steps.append(f"  Stored {len(ids)} chunks")
    except Exception as e:
        return f"Error storing in ChromaDB: {e}"

    steps.append("Building RAPTOR tree...")
    if progress_fn:
        progress_fn("\n".join(steps))
    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_node("root", type="paper", title=title, arxiv_id=arxiv_id, metadata=metadata)

        topic_id = "topic_0"
        G.add_node(topic_id, type="topic", title="General")
        G.add_edge("root", topic_id)

        section_size = 10
        section_idx = 0
        for start in range(0, len(chunks), section_size):
            batch = chunks[start:start + section_size]
            sec_id = f"section_{section_idx}"
            G.add_node(sec_id, type="section", section_num=str(section_idx + 1),
                       title=f"Section {section_idx + 1}", summary="")
            G.add_edge(topic_id, sec_id)
            for c in batch:
                chunk_id = f"chunk_{c['chunk_index']}"
                G.add_node(chunk_id, type="chunk", chunk_index=c["chunk_index"], text=c["text"])
                G.add_edge(sec_id, chunk_id)
            section_idx += 1

        save_tree(arxiv_id, G)
        steps.append(f"  Tree: 1 topic, {section_idx} sections, {len(chunks)} chunks")
    except Exception as e:
        return f"Error building tree: {e}"

    # Update metadata
    try:
        meta_path = os.path.join(BASE_DIR, 'data', 'raw', 'papers_metadata_with_id.json')
        existing_meta = []
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
        if not any(p.get("arxiv_id") == arxiv_id for p in existing_meta):
            existing_meta.append(metadata)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(existing_meta, f, indent=2, ensure_ascii=False)
    except Exception as e:
        steps.append(f"  Warning: metadata update failed: {e}")

    global _retriever
    _retriever = None

    steps.append(f"\n**Done!** Paper `{arxiv_id}` — *{title}* ingested successfully.")
    steps.append(f"  - {len(chunks)} chunks embedded and stored")
    steps.append("  - Ready for querying in the Chat tab")

    return "\n".join(steps)


# ============================================================
#  Tab 4: Dashboard — helpers
# ============================================================

def dashboard_fn():
    """Gather system status."""
    # Papers
    papers = list_all_papers()
    paper_count = len(papers)

    # Chunks in ChromaDB
    try:
        db = VectorDB()
        chunk_count = db.count()
    except Exception:
        chunk_count = -1

    # Tree stats
    tree_dir = os.path.join(BASE_DIR, 'data', 'raw', 'paper_trees')
    total_topics = total_sections = total_chunks = 0
    summaries_done = 0
    for f in glob.glob(os.path.join(tree_dir, '*_tree.gpickle')):
        try:
            with open(f, 'rb') as fh:
                G = pickle.load(fh)
            for n, d in G.nodes(data=True):
                ntype = d.get('type', '')
                if ntype == 'topic':
                    total_topics += 1
                elif ntype == 'section':
                    total_sections += 1
                    if d.get('summary', '').strip():
                        summaries_done += 1
                elif ntype == 'chunk':
                    total_chunks += 1
        except Exception:
            pass

    # Sessions
    sessions = session_manager.list_sessions()

    # Feedback
    fb_stats = feedback_store.get_stats()

    # Models
    models = list_available_models()
    active = get_active_model()
    finetuned = [k for k, v in models.items() if v.get("is_finetuned")]

    lines = [
        "## System Dashboard\n",
        "### Data",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Papers indexed | {paper_count} |",
        f"| Chunks in ChromaDB | {chunk_count:,} |",
        f"| RAPTOR topics | {total_topics} |",
        f"| RAPTOR sections | {total_sections} |",
        f"| Section summaries | {summaries_done}/{total_sections} ({100*summaries_done/max(total_sections,1):.0f}%) |",
        f"| Tree chunks | {total_chunks:,} |",
        "",
        "### Sessions & Feedback",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Active sessions | {len(sessions)} |",
        f"| Total feedback | {fb_stats.get('total', 0)} |",
    ]

    by_type = fb_stats.get("by_type", {})
    for ftype in ["helpful", "incorrect", "hallucination", "correction"]:
        lines.append(f"| Feedback: {ftype} | {by_type.get(ftype, 0)} |")

    lines.extend([
        "",
        "### Models",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Active model | `{active}` |",
        f"| Total models | {len(models)} |",
        f"| Fine-tuned models | {len(finetuned)} |",
    ])

    if finetuned:
        lines.append("")
        lines.append("**Fine-tuned models:**")
        for name in finetuned:
            info = models[name]
            lines.append(f"- `{name}` (base: {info.get('base_model', '?')})")

    return "\n".join(lines)


# ============================================================
#  Build full Gradio Interface
# ============================================================

def create_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAPTOR Research Assistant",
        theme=gr.themes.Soft(),
        css="""
        .citation-panel { max-height: 400px; overflow-y: auto; }
        .paper-viewer { max-height: 600px; overflow-y: auto; }
        """
    ) as demo:
        gr.Markdown(
            "# RAPTOR Research Assistant\n"
            "Ask questions about ML/DL research papers. "
            "The system retrieves context from 200+ indexed papers "
            "using RAPTOR hierarchical search and generates answers with citations."
        )

        # ========================================
        #  Tab 1: Chat
        # ========================================
        with gr.Tab("Chat"):
            session_state = gr.State("")

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat", height=500, type="messages", show_copy_button=True,
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask about research papers...",
                            label="Message", scale=4, lines=1,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    task_dropdown = gr.Dropdown(
                        choices=["Q&A", "Summarize", "Compare", "Explain"],
                        value="Q&A", label="Task Type",
                    )
                    model_dropdown = gr.Dropdown(
                        choices=["Auto (Best Available)", "Mistral (Local)", "Groq Llama 3.3 (Cloud)"],
                        value="Auto (Best Available)", label="Model",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1, label="Top-K Results",
                    )

                    gr.Markdown("### Session")
                    new_session_btn = gr.Button("New Session", variant="secondary")
                    session_dropdown = gr.Dropdown(
                        choices=_format_session_list(),
                        label="Load Session", interactive=True,
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
                        label="Correction Text", lines=2, visible=True,
                    )
                    feedback_display = gr.Markdown(
                        value="*Rate the last response using the buttons above.*"
                    )

                    gr.Markdown("### Citations")
                    citations_display = gr.Markdown(
                        value="*Citations will appear here after each response.*",
                        elem_classes=["citation-panel"],
                    )

            # Chat event handlers
            send_inputs = [msg_input, chatbot, session_state, task_dropdown, model_dropdown, top_k_slider]
            send_outputs = [chatbot, msg_input, session_state, citations_display]

            send_btn.click(fn=chat_fn, inputs=send_inputs, outputs=send_outputs)
            msg_input.submit(fn=chat_fn, inputs=send_inputs, outputs=send_outputs)

            new_session_btn.click(
                fn=new_session_fn,
                outputs=[chatbot, session_state, citations_display, session_dropdown],
            )
            load_session_btn.click(
                fn=load_session_fn, inputs=[session_dropdown],
                outputs=[chatbot, session_state, citations_display],
            )

            feedback_inputs = [chatbot, session_state, task_dropdown, model_dropdown, correction_input]
            helpful_btn.click(
                fn=lambda *args: submit_feedback_fn("helpful", *args),
                inputs=feedback_inputs, outputs=[feedback_display],
            )
            incorrect_btn.click(
                fn=lambda *args: submit_feedback_fn("incorrect", *args),
                inputs=feedback_inputs, outputs=[feedback_display],
            )
            hallucination_btn.click(
                fn=lambda *args: submit_feedback_fn("hallucination", *args),
                inputs=feedback_inputs, outputs=[feedback_display],
            )
            correction_btn.click(
                fn=lambda *args: submit_feedback_fn("correction", *args),
                inputs=feedback_inputs, outputs=[feedback_display],
            )

        # ========================================
        #  Tab 2: Paper Browser
        # ========================================
        with gr.Tab("Papers"):
            gr.Markdown("### Browse Indexed Papers\n"
                        "Explore the RAPTOR tree hierarchy: **Paper → Topics → Sections → Chunks**")

            with gr.Row():
                with gr.Column(scale=1):
                    paper_list_btn = gr.Button("Load Paper List", variant="primary")
                    paper_id_input = gr.Textbox(
                        placeholder="e.g. 1706.03762",
                        label="arXiv ID", lines=1,
                    )
                    view_paper_btn = gr.Button("View Paper Tree", variant="secondary")
                    topic_dropdown = gr.Dropdown(
                        choices=[], label="Select Topic", interactive=True,
                    )
                    view_topic_btn = gr.Button("View Sections", size="sm")
                    section_dropdown = gr.Dropdown(
                        choices=[], label="Select Section", interactive=True,
                    )
                    view_section_btn = gr.Button("View Chunks", size="sm")

                with gr.Column(scale=3):
                    paper_list_display = gr.Markdown(
                        value="*Click 'Load Paper List' to see all papers.*",
                        elem_classes=["paper-viewer"],
                    )
                    paper_detail_display = gr.Markdown(
                        value="", elem_classes=["paper-viewer"],
                    )
                    section_detail_display = gr.Markdown(
                        value="", elem_classes=["paper-viewer"],
                    )
                    chunk_display = gr.Markdown(
                        value="", elem_classes=["paper-viewer"],
                    )

            # Paper browser event handlers
            paper_list_btn.click(fn=list_papers_fn, outputs=[paper_list_display])
            view_paper_btn.click(
                fn=paper_overview_fn, inputs=[paper_id_input],
                outputs=[paper_detail_display, topic_dropdown],
            )
            view_topic_btn.click(
                fn=topic_sections_fn, inputs=[paper_id_input, topic_dropdown],
                outputs=[section_detail_display, section_dropdown],
            )
            view_section_btn.click(
                fn=section_chunks_fn, inputs=[paper_id_input, section_dropdown],
                outputs=[chunk_display],
            )

        # ========================================
        #  Tab 3: Upload Paper
        # ========================================
        with gr.Tab("Upload"):
            gr.Markdown(
                "### Add New Papers\n"
                "Upload a paper via arXiv ID (auto-fetches) or direct PDF upload.\n"
                "The paper will be processed through the full pipeline: "
                "**PDF → Chunks → Embeddings → ChromaDB → RAPTOR Tree**"
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Option 1: By arXiv ID")
                    arxiv_id_upload = gr.Textbox(
                        placeholder="e.g. 2301.00234",
                        label="arXiv ID", lines=1,
                    )
                    ingest_arxiv_btn = gr.Button("Fetch & Ingest from arXiv", variant="primary")

                with gr.Column():
                    gr.Markdown("#### Option 2: Upload PDF")
                    pdf_upload = gr.File(
                        label="Upload PDF", file_types=[".pdf"],
                    )
                    pdf_id_input = gr.Textbox(
                        placeholder="Optional: custom ID for this paper",
                        label="Paper ID (optional)", lines=1,
                    )
                    pdf_title_input = gr.Textbox(
                        placeholder="Optional: paper title",
                        label="Title (optional)", lines=1,
                    )
                    ingest_pdf_btn = gr.Button("Process & Ingest PDF", variant="primary")

            upload_log = gr.Markdown(
                value="*Upload results will appear here.*",
            )

            ingest_arxiv_btn.click(
                fn=_ingest_paper_from_arxiv, inputs=[arxiv_id_upload],
                outputs=[upload_log],
            )
            ingest_pdf_btn.click(
                fn=_ingest_paper_from_pdf,
                inputs=[pdf_upload, pdf_id_input, pdf_title_input],
                outputs=[upload_log],
            )

        # ========================================
        #  Tab 4: Dashboard
        # ========================================
        with gr.Tab("Dashboard"):
            gr.Markdown("### System Status\nOverview of data, sessions, feedback, and models.")
            refresh_btn = gr.Button("Refresh", variant="secondary")
            dashboard_display = gr.Markdown(value="*Click Refresh to load.*")
            refresh_btn.click(fn=dashboard_fn, outputs=[dashboard_display])

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
