# RAPTOR Research Assistant

> **Status: In Progress** — Sections 1–10 of 18 complete. Actively building RLHF pipeline, evaluation system, and fine-tuning workflows.

A modular AI research assistant that reads, summarizes, compares, and reasons over 200+ ML/DL research papers using **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) — a hierarchical RAG approach that organizes papers into tree structures for deeper context-aware retrieval.

The system features multi-model LLM reasoning (local Ollama + cloud APIs), a two-way chatbot with session memory, user feedback collection, and a planned RLHF/DPO fine-tuning loop for continuous improvement.

---

## Architecture Overview

### High-Level System Architecture

```mermaid
flowchart TB
    subgraph INGESTION["📥 Data Ingestion Pipeline"]
        direction LR
        A1[("🌐 arXiv API<br/>204 papers<br/>cs.AI · cs.LG · stat.ML")]
        A2["📄 PDF Extraction<br/><i>PyMuPDF</i>"]
        A3["✂️ Text Chunking<br/>300–500 tokens"]
        A4["🧮 Embeddings<br/><i>all-MiniLM-L6-v2</i><br/>384 dimensions"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph STORAGE["💾 Storage Layer"]
        direction LR
        B1[("🗄️ ChromaDB<br/>148,986 chunks<br/>+ metadata<br/>+ embeddings")]
        B2[("🌳 RAPTOR Trees<br/>204 NetworkX DiGraphs<br/>.gpickle files")]
    end

    subgraph INDEX["🌲 RAPTOR Hierarchical Index"]
        direction TB
        C1["📚 Paper <i>(root)</i>"]
        C2["🏷️ Topic 1<br/><i>clustered + summary</i>"]
        C3["🏷️ Topic 2<br/><i>clustered + summary</i>"]
        C4["📑 Section 1.1<br/><i>title + summary</i>"]
        C5["📑 Section 1.2"]
        C6["📑 Section 2.1"]
        C7["📝 Chunk"] 
        C8["📝 Chunk"]
        C9["📝 Chunk"]
        C1 --> C2 & C3
        C2 --> C4 & C5
        C3 --> C6
        C4 --> C7 & C8
        C6 --> C9
    end

    subgraph RETRIEVAL["🔍 Hybrid Retrieval Engine"]
        direction TB
        D0["❓ User Query"]
        D1["Vector Search<br/><i>cosine similarity</i><br/><i>ChromaDB</i>"]
        D2["RAPTOR Tree Walk-Up<br/><i>chunk → section → topic → paper</i>"]
        D3["Merged Results<br/><i>chunks + hierarchical context</i>"]
        D0 --> D1 & D2
        D1 --> D3
        D2 --> D3
    end

    subgraph REASONING["🧠 LLM Reasoning Pipeline"]
        direction TB
        E1["📋 Prompt Builder<br/><i>system + context + history + question</i><br/>Tasks: Q&A · Summarize · Compare · Explain"]
        E2["🤖 Mistral<br/><i>Ollama · Local</i>"]
        E3["☁️ Llama 3.3 70B<br/><i>Groq · Cloud</i>"]
        E4["💬 Answer + Citations"]
        E1 --> E2 & E3
        E2 --> E4
        E3 --> E4
    end

    subgraph INTERFACE["🖥️ User Interface Layer"]
        direction LR
        F1["💬 Gradio Chat UI<br/>Multi-turn · Sessions"]
        F2["⚡ FastAPI Server<br/>19 REST endpoints"]
    end

    subgraph FEEDBACK["🔄 Feedback & Learning Loop"]
        direction LR
        G1["👍👎 User Feedback<br/>helpful · incorrect<br/>hallucination · correction"]
        G2["📊 Preference Pairs<br/><i>chosen vs rejected</i>"]
        G3["🎯 DPO Fine-Tuning<br/><i>TRL/PEFT</i>"]
        G1 --> G2 --> G3
    end

    INGESTION --> STORAGE
    STORAGE --> INDEX
    INDEX --> RETRIEVAL
    RETRIEVAL --> REASONING
    REASONING --> INTERFACE
    INTERFACE --> FEEDBACK
    FEEDBACK -.->|"improved model"| REASONING

    style INGESTION fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style STORAGE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style INDEX fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style RETRIEVAL fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style REASONING fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style INTERFACE fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style FEEDBACK fill:#fff9c4,stroke:#f9a825,stroke-width:2px
```

### Request Flow — What Happens When You Ask a Question

```mermaid
sequenceDiagram
    actor User
    participant UI as 🖥️ Gradio UI
    participant Session as 🔑 Session Manager
    participant Retriever as 🔍 Hybrid Retriever
    participant ChromaDB as 🗄️ ChromaDB
    participant Tree as 🌳 RAPTOR Tree
    participant Prompt as 📋 Prompt Builder
    participant LLM as 🤖 LLM (Mistral/Groq)
    participant Feedback as 📊 Feedback Store

    User->>UI: "How does self-attention work?"
    UI->>Session: get_or_create(session_id)
    Session-->>UI: session (with chat history)
    
    UI->>Retriever: retrieve(query, top_k=5)
    Retriever->>ChromaDB: vector search (cosine similarity)
    ChromaDB-->>Retriever: top-5 chunk IDs + text
    
    loop For each chunk
        Retriever->>Tree: walk_up(chunk → section → topic → paper)
        Tree-->>Retriever: hierarchical context + summaries
    end
    
    Retriever-->>UI: chunks + tree context + citations
    
    UI->>Prompt: build_messages(chunks, question, task, history)
    Prompt-->>UI: [system_msg, user_msg]
    
    UI->>LLM: POST /v1/chat/completions
    LLM-->>UI: generated answer
    
    UI->>Session: store(user_msg + assistant_msg + citations)
    UI-->>User: answer + citations panel
    
    User->>Feedback: 👍 Helpful / 👎 Incorrect / etc.
    Feedback->>Feedback: append to feedback.jsonl
```

### RAPTOR Tree Structure — How Papers Are Organized

```mermaid
graph TD
    subgraph TREE["Example: Attention Is All You Need (1706.03762)"]
        R["📚 Paper Root<br/><b>Attention Is All You Need</b><br/><i>Vaswani et al. 2017</i>"]
        
        T1["🏷️ Topic: Architecture<br/><i>Sections about model design</i><br/>🤖 Summary: <i>The Transformer uses<br/>encoder-decoder with attention...</i>"]
        T2["🏷️ Topic: Training<br/><i>Sections about optimization</i><br/>🤖 Summary: <i>Training uses Adam optimizer<br/>with warmup and label smoothing...</i>"]
        T3["🏷️ Topic: Results<br/><i>Sections about experiments</i><br/>🤖 Summary: <i>Achieves SOTA on WMT<br/>translation benchmarks...</i>"]
        
        S1["📑 3.1 Encoder-Decoder<br/>🤖 <i>The encoder maps input to<br/>continuous representations...</i>"]
        S2["📑 3.2 Attention<br/>🤖 <i>Scaled dot-product attention<br/>computes compatibility...</i>"]
        S3["📑 5.1 Training Data<br/>🤖 <i>WMT 2014 English-German<br/>4.5M sentence pairs...</i>"]
        S4["📑 6.1 Machine Translation<br/>🤖 <i>Big Transformer achieves<br/>28.4 BLEU on EN-DE...</i>"]
        
        CH1["📝 Chunk: <i>An attention function can<br/>be described as mapping a<br/>query and set of key-value<br/>pairs to an output...</i>"]
        CH2["📝 Chunk: <i>We compute the attention<br/>function on a set of queries<br/>simultaneously, packed into<br/>a matrix Q...</i>"]
        CH3["📝 Chunk: <i>We trained on the WMT<br/>2014 English-German dataset<br/>consisting of about 4.5<br/>million sentence pairs...</i>"]
        
        R --> T1 & T2 & T3
        T1 --> S1 & S2
        T2 --> S3
        T3 --> S4
        S2 --> CH1 & CH2
        S3 --> CH3
    end

    style R fill:#4a148c,color:#fff,stroke:#4a148c
    style T1 fill:#1565c0,color:#fff,stroke:#1565c0
    style T2 fill:#1565c0,color:#fff,stroke:#1565c0
    style T3 fill:#1565c0,color:#fff,stroke:#1565c0
    style S1 fill:#2e7d32,color:#fff,stroke:#2e7d32
    style S2 fill:#2e7d32,color:#fff,stroke:#2e7d32
    style S3 fill:#2e7d32,color:#fff,stroke:#2e7d32
    style S4 fill:#2e7d32,color:#fff,stroke:#2e7d32
    style CH1 fill:#ef6c00,color:#fff,stroke:#ef6c00
    style CH2 fill:#ef6c00,color:#fff,stroke:#ef6c00
    style CH3 fill:#ef6c00,color:#fff,stroke:#ef6c00
```

### Component Interaction Map

```mermaid
graph LR
    subgraph API["FastAPI Backend"]
        MCP["mcp_server.py<br/>/retrieve · /prompt · /llm"]
        CHAT["chat.py<br/>/chat · /chat/session"]
        FB["feedback.py<br/>/feedback · /feedback/stats"]
        RET["retrieve.py<br/>/retrieve/tree · /retrieve/papers"]
    end

    subgraph CORE["Core Logic"]
        RI["raptor_index.py<br/>Tree Operations"]
        RT["retrieval.py<br/>Hybrid Retriever"]
        VDB["vector_db.py<br/>ChromaDB Wrapper"]
        PB["prompt_builder.py<br/>4 Task Types"]
        PT["prompt.py<br/>Templates"]
        LLM["llm_client.py<br/>Model Registry"]
        SM["session.py<br/>Session Store"]
        FBS["feedback.py<br/>JSONL Store"]
    end

    subgraph UI_LAYER["Frontend"]
        GR["ui.py<br/>Gradio Interface"]
    end

    subgraph EXTERNAL["External Services"]
        OL["Ollama<br/>localhost:11435"]
        GQ["Groq API<br/>Cloud"]
        CH[("ChromaDB<br/>148K chunks")]
        FS[("Paper Trees<br/>204 .gpickle")]
    end

    GR --> RT & PB & LLM & SM & FBS
    MCP --> RT & PB & LLM
    CHAT --> RT & PB & LLM & SM
    FB --> FBS
    RET --> RT

    RT --> VDB & RI
    PB --> PT
    VDB --> CH
    RI --> FS
    LLM --> OL & GQ

    style API fill:#e3f2fd,stroke:#1565c0
    style CORE fill:#f3e5f5,stroke:#7b1fa2
    style UI_LAYER fill:#e0f2f1,stroke:#00695c
    style EXTERNAL fill:#fff3e0,stroke:#ef6c00
```

### Feedback → Fine-Tuning Pipeline (Planned)

```mermaid
flowchart LR
    A["💬 Chat Response"] --> B{"👤 User Rates"}
    B -->|"👍 Helpful"| C["✅ Chosen"]
    B -->|"👎 Incorrect"| D["❌ Rejected"]
    B -->|"🚫 Hallucination"| D
    B -->|"✏️ Correction"| E["✅ User's corrected<br/>text = Chosen<br/>❌ Original = Rejected"]
    
    C --> F["📊 Preference Dataset<br/><i>(prompt, chosen, rejected)</i>"]
    D --> F
    E --> F
    
    F --> G["🎯 DPO Training<br/><i>TRL + PEFT + LoRA</i>"]
    G --> H["🤖 Fine-Tuned Mistral"]
    H --> I["🔄 Deploy & Serve"]
    I -.->|"Better answers<br/>next time"| A

    style C fill:#c8e6c9,stroke:#2e7d32
    style D fill:#ffcdd2,stroke:#c62828
    style E fill:#fff9c4,stroke:#f9a825
    style F fill:#e3f2fd,stroke:#1565c0
    style G fill:#f3e5f5,stroke:#7b1fa2
    style H fill:#fce4ec,stroke:#c62828
```

---

## Project Structure

```
raptor-research-assistant/
│
├── app/
│   ├── api/                    # FastAPI endpoints
│   │   ├── mcp_server.py       # Main server: /retrieve, /prompt, /llm
│   │   ├── chat.py             # /chat endpoints with session support
│   │   ├── feedback.py         # /feedback endpoints
│   │   ├── retrieve.py         # /retrieve router (hybrid search)
│   │   └── train.py            # /train endpoint (planned)
│   │
│   ├── core/                   # Business logic
│   │   ├── raptor_index.py     # RAPTOR tree operations (NetworkX)
│   │   ├── retrieval.py        # Hybrid retriever (vector + tree)
│   │   ├── vector_db.py        # ChromaDB wrapper
│   │   ├── prompt_builder.py   # Prompt assembly (4 task types)
│   │   ├── prompt.py           # Prompt templates
│   │   ├── llm_client.py       # Multi-model LLM client
│   │   ├── session.py          # In-memory session manager
│   │   ├── feedback.py         # Feedback storage (JSONL)
│   │   ├── embedding.py        # SentenceTransformers embeddings
│   │   ├── ingestion.py        # arXiv paper fetching
│   │   ├── pdf_processing.py   # PDF text extraction
│   │   ├── preference.py       # Preference dataset (planned)
│   │   ├── finetune.py         # DPO fine-tuning (planned)
│   │   └── evaluation.py       # RAGAS evaluation (planned)
│   │
│   ├── frontend/
│   │   └── ui.py               # Gradio chat interface
│   │
│   └── utils/
│       └── helpers.py          # Shared utilities
│
├── scripts/                    # Pipeline & utility scripts
│   ├── build_index.py          # Build RAPTOR trees + summaries
│   ├── ingest_papers.py        # Fetch papers from arXiv
│   ├── process_pdfs.py         # Extract text from PDFs
│   ├── generate_embeddings.py  # Create embeddings
│   ├── store_embeddings_in_chroma.py
│   ├── walkthrough_example.py  # Demo: full pipeline walkthrough
│   └── ...                     # Various utility scripts
│
├── tests/
│   └── test_raptor_index.py    # 27 tests for tree operations
│
├── data/                       # (gitignored — local data)
│   ├── raw/                    # PDFs, metadata, paper trees
│   ├── embeddings/             # Cached embeddings
│   ├── processed/              # Processed text
│   └── feedback/               # User feedback JSONL
│
├── config.yaml                 # All configuration
├── requirements.txt            # Python dependencies
└── PROJECT_PLAN.md             # Full 18-section project plan
```

---

## How It Works

### 1. Data Pipeline (Sections 1–4)

Papers are fetched from **arXiv** (categories: cs.AI, cs.LG, stat.ML), PDFs are extracted and split into 300–500 token chunks, each chunk is embedded using **SentenceTransformers** (`all-MiniLM-L6-v2`, 384-dim vectors), and stored in **ChromaDB** (148,986 chunks from 204 papers).

### 2. RAPTOR Hierarchical Index (Section 5)

Each paper is organized into a **4-level tree** using NetworkX:

```
Paper (root) → Topics (clustered) → Sections → Chunks
```

- **Topic clustering**: Sections with similar embeddings are grouped using Agglomerative Clustering (scikit-learn)
- **Summaries**: Each section and topic gets an LLM-generated summary for richer context
- **Tree traversal**: When a chunk is retrieved, the system walks up the tree to get section → topic → paper context

Currently: 204 papers indexed, 105 with full 4-level trees, 1,349 sections, 659 topics.

### 3. Hybrid Retrieval Engine (Section 6)

On every query, two retrieval strategies run in parallel:

1. **Vector search**: Embed the query → cosine similarity search in ChromaDB → top-K chunks
2. **RAPTOR tree walk-up**: For each retrieved chunk, walk up the tree to attach hierarchical context (section summary, topic summary, paper title)

This gives the LLM both the specific text AND the broader context of where that text fits in the paper's structure.

### 4. Prompt Construction (Section 7)

Prompts are assembled with 4 task-specific templates:

| Task | Purpose | Temperature |
|------|---------|-------------|
| **Q&A** | Answer questions from retrieved context | 0.3 |
| **Summarize** | Summarize papers or topics | 0.2 |
| **Compare** | Compare findings across papers | 0.3 |
| **Explain** | Step-by-step concept explanation | 0.4 |

Each prompt includes: System instruction → Hierarchical context blocks → Chat history → User question → Task-specific instructions.

### 5. Multi-Model LLM Reasoning (Section 8)

Supports switching between models per request:

| Model | Type | Use Case |
|-------|------|----------|
| **Mistral** (Ollama) | Local, on-device | Default for all inference, will be fine-tuned later |
| **Groq Llama 3.3 70B** | Cloud API | Bulk summarization, high-quality answers |

Task-specific generation parameters (temperature, max_tokens) are automatically applied based on the task type.

### 6. Two-Way Chatbot (Section 9)

- **Session management**: Each chat gets a unique session ID with stored history (questions, answers, citations, timestamps)
- **Multi-turn context**: Last 10 conversation turns are passed to the LLM for context-aware follow-up responses
- **Gradio UI**: Chat window + settings panel + citation display + session management
- **FastAPI endpoints**: `POST /chat`, `GET /chat/sessions`, `GET /chat/session/{id}`

### 7. Feedback System (Section 10)

After each answer, users can rate the response:

| Feedback | Meaning | Used For |
|----------|---------|----------|
| **Helpful** | Accurate and useful | "Chosen" in preference pairs |
| **Incorrect** | Factual errors | "Rejected" in preference pairs |
| **Hallucination** | Made up info not in sources | "Rejected" in preference pairs |
| **Correction** | User writes corrected answer | Corrected text becomes "chosen" |

Feedback is stored in JSONL format with full context (question, answer, citations, model, session) — ready for Section 11's preference dataset creation.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/retrieve` | POST | Hybrid vector + tree retrieval |
| `/retrieve/tree` | POST | Browse by topic/section |
| `/retrieve/papers` | GET | List all 204 paper IDs |
| `/retrieve/paper/{id}` | GET | Paper tree overview |
| `/prompt` | POST | Retrieve + build prompt |
| `/llm` | POST | Full pipeline: retrieve → prompt → LLM answer |
| `/llm/models` | GET | List available models |
| `/llm/health` | GET | Check if LLM is responding |
| `/chat` | POST | Session-aware chat (auto-creates session) |
| `/chat/session` | POST | Create new session |
| `/chat/session/{id}` | GET | Get session history |
| `/chat/sessions` | GET | List all sessions |
| `/chat/session/{id}` | DELETE | Delete session |
| `/feedback` | POST | Submit feedback |
| `/feedback` | GET | Get all feedback |
| `/feedback/stats` | GET | Feedback statistics |
| `/feedback/session/{id}` | GET | Feedback for a session |
| `/feedback/type/{type}` | GET | Filter by feedback type |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) with Mistral model pulled (`ollama pull mistral`)

### Setup

```bash
# Clone the repo
git clone https://github.com/priyankmistry21699-web/raptor-research-assistant.git
cd raptor-research-assistant

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo LLM_API_KEY=ollama > .env
echo LLM_API_URL=http://localhost:11434/v1/chat/completions >> .env
echo LLM_MODEL=mistral:latest >> .env
```

### Run

```bash
# Option 1: Gradio Chat UI (interactive)
python -m app.frontend.ui
# Opens at http://localhost:7860

# Option 2: FastAPI Server (API)
uvicorn app.api.mcp_server:app --port 8000
# API docs at http://localhost:8000/docs
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB |
| **Tree Index** | NetworkX (DiGraph) |
| **Clustering** | scikit-learn (Agglomerative) |
| **LLM (local)** | Ollama + Mistral |
| **LLM (cloud)** | Groq API + Llama 3.3 70B |
| **Backend** | FastAPI + Pydantic |
| **Frontend** | Gradio |
| **Feedback** | JSONL file storage |
| **Config** | YAML |
| **Data Source** | arXiv API |

---

## Roadmap

- [x] **Section 1** — Data Ingestion (arXiv API, 204 papers)
- [x] **Section 2** — PDF Processing & Chunking
- [x] **Section 3** — Embedding Generation (384-dim)
- [x] **Section 4** — Vector Database (ChromaDB, 148K chunks)
- [x] **Section 5** — RAPTOR Hierarchical Index (NetworkX trees)
- [x] **Section 6** — Hybrid Retrieval Engine
- [x] **Section 7** — Prompt Construction (4 task types)
- [x] **Section 8** — Multi-Model LLM Reasoning
- [x] **Section 9** — Two-Way Chatbot with Sessions
- [x] **Section 10** — Feedback System
- [ ] **Section 11** — Preference Dataset Creation (DPO pairs)
- [ ] **Section 12** — Model Fine-Tuning (TRL/PEFT + DPO)
- [ ] **Section 13** — Continuous Learning Loop
- [ ] **Section 14** — Evaluation System (RAGAS)
- [ ] **Section 15** — Backend System (unified FastAPI)
- [ ] **Section 16** — Frontend Interface (full Gradio UI)
- [ ] **Section 17** — DevOps & Scalability
- [ ] **Section 18** — Paper-Specific Learning & Debate

---

## Data Stats

| Metric | Value |
|--------|-------|
| Papers indexed | 204 |
| Total chunks | 148,986 |
| Sections | 1,349 |
| Topics | 659 |
| Papers with 4-level trees | 105 |
| Embedding dimensions | 384 |
| Categories | cs.AI, cs.LG, stat.ML |

---

## License

This project is for educational and research purposes.

---

*Built with RAPTOR RAG, Ollama, ChromaDB, and a lot of research papers.*
