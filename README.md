# RAPTOR Research Assistant

> **Status: In Progress** — Sections 1–14 of 18 complete. Core RAG pipeline, RLHF/DPO fine-tuning loop, continuous learning, and RAGAS evaluation system all operational.

A modular AI research assistant that reads, summarizes, compares, and reasons over 200+ ML/DL research papers using **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) — a hierarchical RAG approach that organizes papers into tree structures for deeper context-aware retrieval.

The system features multi-model LLM reasoning (local Ollama + cloud APIs), a two-way chatbot with session memory, user feedback collection, preference-based DPO fine-tuning with LoRA adapters, a continuous learning loop, and RAGAS-powered evaluation.

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
        E5["🧬 Fine-Tuned Mistral<br/><i>Base + LoRA Adapter</i><br/><i>Local PeftModel inference</i>"]
        E4["💬 Answer + Citations"]
        E1 --> E2 & E3 & E5
        E2 --> E4
        E3 --> E4
        E5 --> E4
    end

    subgraph INTERFACE["🖥️ User Interface Layer"]
        direction LR
        F1["💬 Gradio Chat UI<br/>Multi-turn · Sessions"]
        F2["⚡ FastAPI Server<br/>39 REST endpoints"]
    end

    subgraph LEARNING["🔄 Continuous Learning Loop"]
        direction LR
        G1["👍👎 User Feedback<br/>helpful · incorrect<br/>hallucination · correction"]
        G2["📊 Preference Pairs<br/><i>chosen vs rejected</i>"]
        G3["🎯 DPO Training<br/><i>TRL + PEFT + LoRA</i><br/><i>4-bit quantization</i>"]
        G4["📦 Register Model<br/><i>LoRA adapter → registry</i>"]
        G1 --> G2 --> G3 --> G4
    end

    subgraph EVAL["📈 Evaluation System (RAGAS)"]
        direction TB
        H1["🧪 Faithfulness<br/><i>grounded in context?</i>"]
        H2["🎯 Answer Relevancy<br/><i>addresses the question?</i>"]
        H3["📐 Context Precision<br/><i>retrieved chunks relevant?</i>"]
        H4["📊 Model Comparison<br/><i>side-by-side scoring</i>"]
        H1 & H2 & H3 --> H4
    end

    INGESTION --> STORAGE
    STORAGE --> INDEX
    INDEX --> RETRIEVAL
    RETRIEVAL --> REASONING
    REASONING --> INTERFACE
    INTERFACE --> LEARNING
    LEARNING -->|"LoRA adapter<br/>registered"| REASONING
    REASONING --> EVAL
    EVAL -.->|"quality signals<br/>guide model selection"| LEARNING

    style INGESTION fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style STORAGE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style INDEX fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style RETRIEVAL fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style REASONING fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style INTERFACE fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style LEARNING fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style EVAL fill:#ede7f6,stroke:#9c27b0,stroke-width:2px
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
    participant LLM as 🤖 LLM Client
    participant Feedback as 📊 Feedback Store
    participant Pref as 📋 Preference Builder

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

    alt Base Model (Ollama/Groq)
        UI->>LLM: POST /v1/chat/completions
        LLM-->>UI: generated answer
    else Fine-Tuned Model (LoRA)
        UI->>LLM: _run_finetuned_inference()
        Note over LLM: Load base + LoRA adapter<br/>via PeftModel (cached)
        LLM-->>UI: generated answer
    end

    UI->>Session: store(user_msg + assistant_msg + citations)
    UI-->>User: answer + citations panel

    User->>Feedback: 👍 Helpful / 👎 Incorrect / ✏️ Correction
    Feedback->>Feedback: append to feedback.jsonl
    Feedback->>Pref: auto-build preference pair
    Note over Pref: (prompt, chosen, rejected)<br/>stored for DPO training
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
        TRAIN["train.py<br/>/train/preferences · /train/finetune<br/>/train/loop"]
        EVALAPI["eval.py<br/>/eval/single · /eval/batch<br/>/eval/pipeline · /eval/compare"]
    end

    subgraph CORE["Core Logic"]
        RI["raptor_index.py<br/>Tree Operations"]
        RT["retrieval.py<br/>Hybrid Retriever"]
        VDB["vector_db.py<br/>ChromaDB Wrapper"]
        PB["prompt_builder.py<br/>4 Task Types"]
        PT["prompt.py<br/>Templates"]
        LLM["llm_client.py<br/>Model Registry +<br/>Fine-tuned Inference"]
        SM["session.py<br/>Session Store"]
        FBS["feedback.py<br/>JSONL Store"]
        PREF["preference.py<br/>DPO Pair Builder"]
        FT["finetune.py<br/>DPO + LoRA Training"]
        LL["learning_loop.py<br/>Auto Learning Cycle"]
        EV["evaluation.py<br/>RAGAS Metrics"]
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
    FB --> FBS & PREF
    RET --> RT
    TRAIN --> PREF & FT & LL
    EVALAPI --> EV

    RT --> VDB & RI
    PB --> PT
    VDB --> CH
    RI --> FS
    LLM --> OL & GQ
    LL --> FBS & PREF & FT & LLM
    EV --> RT & PB & LLM

    style API fill:#e3f2fd,stroke:#1565c0
    style CORE fill:#f3e5f5,stroke:#7b1fa2
    style UI_LAYER fill:#e0f2f1,stroke:#00695c
    style EXTERNAL fill:#fff3e0,stroke:#ef6c00
```

### Feedback → Fine-Tuning Pipeline

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

### Continuous Learning Loop — Automated Improvement Cycle

```mermaid
flowchart TB
    subgraph TRIGGER["⏰ Trigger"]
        T1["Manual<br/><i>POST /train/loop/trigger</i>"]
        T2["Automatic<br/><i>background thread<br/>checks every N sec</i>"]
    end

    CHECK["📋 Check Feedback Volume<br/><i>min_new_feedback threshold</i>"]
    BUILD["📊 Build Preference Pairs<br/><i>feedback → (prompt, chosen, rejected)</i>"]
    EXPORT["📤 Export DPO Dataset<br/><i>filter valid pairs</i>"]
    TRAIN["🎯 DPO Fine-Tuning<br/><i>TRL + PEFT + LoRA</i><br/><i>4-bit quantization</i>"]
    REGISTER["📦 Register Model<br/><i>add to MODEL_REGISTRY</i><br/><i>is_finetuned: true</i>"]
    SERVE["🤖 Serve via LLM Client<br/><i>Base Mistral + LoRA adapter</i><br/><i>PeftModel · cached</i>"]
    HISTORY["📝 Log to History<br/><i>loop_history.jsonl</i>"]

    T1 --> CHECK
    T2 --> CHECK
    CHECK -->|"enough feedback"| BUILD
    CHECK -->|"not enough"| SKIP["⏭️ Skip<br/><i>wait for more feedback</i>"]
    BUILD --> EXPORT --> TRAIN --> REGISTER --> SERVE
    REGISTER --> HISTORY

    style TRIGGER fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style CHECK fill:#fff3e0,stroke:#ef6c00
    style BUILD fill:#e3f2fd,stroke:#1565c0
    style EXPORT fill:#e3f2fd,stroke:#1565c0
    style TRAIN fill:#f3e5f5,stroke:#7b1fa2
    style REGISTER fill:#fce4ec,stroke:#c62828
    style SERVE fill:#e8f5e9,stroke:#2e7d32
    style SKIP fill:#f5f5f5,stroke:#9e9e9e
```

### Evaluation Pipeline — RAGAS Quality Measurement

```mermaid
flowchart LR
    subgraph INPUT["📥 Input"]
        direction TB
        I1["Single Q&A Pair<br/><i>POST /eval/single</i>"]
        I2["Batch Samples<br/><i>POST /eval/batch</i>"]
        I3["End-to-End Pipeline<br/><i>POST /eval/pipeline</i>"]
        I4["Model Comparison<br/><i>POST /eval/compare</i>"]
    end

    subgraph PIPELINE["⚙️ Pipeline Evaluation"]
        direction TB
        P1["🔍 Retrieve<br/><i>query → top-K chunks</i>"]
        P2["📋 Build Prompt<br/><i>context + question</i>"]
        P3["🤖 LLM Inference<br/><i>generate answer</i>"]
        P1 --> P2 --> P3
    end

    subgraph RAGAS["📈 RAGAS Scoring"]
        direction TB
        R1["🧪 Faithfulness<br/><i>grounded in context?</i>"]
        R2["🎯 Answer Relevancy<br/><i>addresses question?</i>"]
        R3["📐 Context Precision<br/><i>chunks relevant?</i>"]
        R4["✅ Factual Correctness<br/><i>matches reference?</i><br/><i>(optional)</i>"]
    end

    subgraph OUTPUT["📊 Results"]
        direction TB
        O1["Per-Sample Scores"]
        O2["Aggregate Means"]
        O3["Model Comparison<br/><i>side-by-side table</i>"]
        O4["History + Stats<br/><i>eval_results.jsonl</i>"]
    end

    I1 --> RAGAS
    I2 --> RAGAS
    I3 --> PIPELINE --> RAGAS
    I4 -->|"same queries ×<br/>multiple models"| PIPELINE
    RAGAS --> OUTPUT

    style INPUT fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style PIPELINE fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style RAGAS fill:#ede7f6,stroke:#9c27b0,stroke-width:2px
    style OUTPUT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## Project Structure

```
raptor-research-assistant/
│
├── app/
│   ├── api/                    # FastAPI endpoints (39 routes)
│   │   ├── mcp_server.py       # Main server: /retrieve, /prompt, /llm
│   │   ├── chat.py             # /chat endpoints with session support
│   │   ├── feedback.py         # /feedback endpoints
│   │   ├── retrieve.py         # /retrieve router (hybrid search)
│   │   ├── train.py            # /train: preferences, finetune, learning loop
│   │   └── eval.py             # /eval: RAGAS evaluation endpoints
│   │
│   ├── core/                   # Business logic
│   │   ├── raptor_index.py     # RAPTOR tree operations (NetworkX)
│   │   ├── retrieval.py        # Hybrid retriever (vector + tree)
│   │   ├── vector_db.py        # ChromaDB wrapper
│   │   ├── prompt_builder.py   # Prompt assembly (4 task types)
│   │   ├── prompt.py           # Prompt templates
│   │   ├── llm_client.py       # Multi-model LLM client + fine-tuned inference
│   │   ├── session.py          # In-memory session manager
│   │   ├── feedback.py         # Feedback storage (JSONL)
│   │   ├── embedding.py        # SentenceTransformers embeddings
│   │   ├── ingestion.py        # arXiv paper fetching
│   │   ├── pdf_processing.py   # PDF text extraction
│   │   ├── preference.py       # Preference dataset (DPO pairs from feedback)
│   │   ├── finetune.py         # DPO fine-tuning (TRL + PEFT + LoRA)
│   │   ├── learning_loop.py    # Continuous learning loop orchestrator
│   │   └── evaluation.py       # RAGAS evaluation system
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
│   ├── feedback/               # User feedback + loop history JSONL
│   ├── preference/             # DPO preference pairs JSONL
│   └── evaluation/             # RAGAS evaluation results JSONL
│
├── models/                     # Fine-tuned model outputs (LoRA adapters)
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

| Task          | Purpose                                 | Temperature |
| ------------- | --------------------------------------- | ----------- |
| **Q&A**       | Answer questions from retrieved context | 0.3         |
| **Summarize** | Summarize papers or topics              | 0.2         |
| **Compare**   | Compare findings across papers          | 0.3         |
| **Explain**   | Step-by-step concept explanation        | 0.4         |

Each prompt includes: System instruction → Hierarchical context blocks → Chat history → User question → Task-specific instructions.

### 5. Multi-Model LLM Reasoning (Section 8)

Supports switching between models per request:

| Model                  | Type             | Use Case                                            |
| ---------------------- | ---------------- | --------------------------------------------------- |
| **Mistral** (Ollama)   | Local, on-device | Default for all inference, will be fine-tuned later |
| **Groq Llama 3.3 70B** | Cloud API        | Bulk summarization, high-quality answers            |

Task-specific generation parameters (temperature, max_tokens) are automatically applied based on the task type.

### 6. Two-Way Chatbot (Section 9)

- **Session management**: Each chat gets a unique session ID with stored history (questions, answers, citations, timestamps)
- **Multi-turn context**: Last 10 conversation turns are passed to the LLM for context-aware follow-up responses
- **Gradio UI**: Chat window + settings panel + citation display + session management
- **FastAPI endpoints**: `POST /chat`, `GET /chat/sessions`, `GET /chat/session/{id}`

### 7. Feedback System (Section 10)

After each answer, users can rate the response:

| Feedback          | Meaning                      | Used For                        |
| ----------------- | ---------------------------- | ------------------------------- |
| **Helpful**       | Accurate and useful          | "Chosen" in preference pairs    |
| **Incorrect**     | Factual errors               | "Rejected" in preference pairs  |
| **Hallucination** | Made up info not in sources  | "Rejected" in preference pairs  |
| **Correction**    | User writes corrected answer | Corrected text becomes "chosen" |

Feedback is stored in JSONL format with full context (question, answer, citations, model, session) — ready for preference dataset creation.

### 8. Preference Dataset (Section 11)

Feedback entries are automatically converted into **DPO preference pairs** (chosen vs. rejected):

- **Helpful** feedback → the original answer becomes the "chosen" response
- **Incorrect / Hallucination** → the original answer becomes "rejected"
- **Correction** → the user's corrected text becomes "chosen", original becomes "rejected"

Each pair stores the full prompt (system + context + question) so it can be used directly for DPO fine-tuning. Auto-build triggers on every feedback submission.

**Endpoints**: `POST /train/preferences/build`, `GET /train/preferences`, `GET /train/preferences/stats`, `GET /train/preferences/export`

### 9. DPO Fine-Tuning (Section 12)

Full fine-tuning pipeline using **TRL** (DPOTrainer) with **PEFT** (LoRA adapters) and 4-bit quantization:

- **Base model**: Mistral-7B-v0.1 (configurable)
- **LoRA config**: r=16, alpha=32, targets q/k/v/o projections
- **Quantization**: 4-bit NF4 via BitsAndBytes
- **Training**: DPO with β=0.1, auto gradient accumulation
- **Output**: LoRA adapters saved to `models/dpo_<timestamp>/`

Registered fine-tuned models are automatically available for inference — `llm_client.py` detects `is_finetuned` and loads the base model + LoRA adapter locally using PeftModel.

**Endpoints**: `POST /train/finetune`, `GET /train/finetune/status`, `GET /train/finetune/models`, `POST /train/finetune/register`

### 10. Continuous Learning Loop (Section 13)

Automated orchestrator that closes the loop: feedback → preferences → training → deployment:

1. Check if enough new feedback has accumulated (configurable threshold)
2. Build preference pairs from recent feedback
3. Export training-ready DPO data
4. Run DPO fine-tuning
5. Register the new model and set as active

Supports both **manual triggering** and **automatic mode** (background thread checks every N seconds). History of all loop runs is stored in JSONL.

**Endpoints**: `POST /train/loop/trigger`, `GET /train/loop/status`, `POST /train/loop/auto`, `GET /train/loop/history`, `GET /train/loop/model`, `PUT /train/loop/config`

### 11. Evaluation System (Section 14)

RAGAS-powered evaluation for measuring RAG quality:

| Metric                          | What It Measures                              |
| ------------------------------- | --------------------------------------------- |
| **Faithfulness**                | Is the answer grounded in retrieved context?  |
| **Answer Relevancy**            | Does the answer address the question?         |
| **Context Precision**           | Are the retrieved chunks relevant?            |
| **Factual Correctness**         | Does the answer match a reference? (optional) |

Supports single evaluation, batch evaluation, end-to-end pipeline evaluation (query → retrieve → answer → score), and multi-model comparison (same queries across different models).

**Endpoints**: `POST /eval/single`, `POST /eval/batch`, `POST /eval/pipeline`, `POST /eval/compare`, `GET /eval/history`, `GET /eval/stats`

---

## API Endpoints (39 routes)

| Endpoint                       | Method | Description                                        |
| ------------------------------ | ------ | -------------------------------------------------- |
| `/retrieve`                    | POST   | Hybrid vector + tree retrieval                     |
| `/retrieve/tree`               | POST   | Browse by topic/section                            |
| `/retrieve/papers`             | GET    | List all 204 paper IDs                             |
| `/retrieve/paper/{id}`         | GET    | Paper tree overview                                |
| `/prompt`                      | POST   | Retrieve + build prompt                            |
| `/llm`                         | POST   | Full pipeline: retrieve → prompt → LLM answer      |
| `/llm/models`                  | GET    | List available models (incl. fine-tuned)           |
| `/llm/health`                  | GET    | Check if LLM is responding                         |
| `/chat`                        | POST   | Session-aware chat (auto-creates session)          |
| `/chat/session`                | POST   | Create new session                                 |
| `/chat/session/{id}`           | GET    | Get session history                                |
| `/chat/sessions`               | GET    | List all sessions                                  |
| `/chat/session/{id}`           | DELETE | Delete session                                     |
| `/feedback`                    | POST   | Submit feedback                                    |
| `/feedback`                    | GET    | Get all feedback                                   |
| `/feedback/stats`              | GET    | Feedback statistics                                |
| `/feedback/session/{id}`       | GET    | Feedback for a session                             |
| `/feedback/type/{type}`        | GET    | Filter by feedback type                            |
| `/train/preferences/build`     | POST   | Build preference pairs from feedback               |
| `/train/preferences`           | GET    | Get all preference pairs                           |
| `/train/preferences/stats`     | GET    | Preference dataset statistics                      |
| `/train/preferences/export`    | GET    | Export DPO training pairs                          |
| `/train/finetune`              | POST   | Start DPO fine-tuning run                          |
| `/train/finetune/status`       | GET    | Check training progress                            |
| `/train/finetune/models`       | GET    | List fine-tuned models                             |
| `/train/finetune/register`     | POST   | Register fine-tuned model for inference            |
| `/train/loop/trigger`          | POST   | Manually trigger learning loop cycle               |
| `/train/loop/status`           | GET    | Learning loop state                                |
| `/train/loop/auto`             | POST   | Enable/disable automatic loop                      |
| `/train/loop/history`          | GET    | History of all loop runs                           |
| `/train/loop/model`            | GET    | Currently selected best model                      |
| `/train/loop/config`           | PUT    | Update loop configuration                          |
| `/eval/single`                 | POST   | Evaluate a single Q&A pair (RAGAS)                 |
| `/eval/batch`                  | POST   | Evaluate a batch of Q&A samples                    |
| `/eval/pipeline`               | POST   | End-to-end RAG pipeline evaluation                 |
| `/eval/compare`                | POST   | Compare multiple models side-by-side               |
| `/eval/history`                | GET    | Recent evaluation results                          |
| `/eval/stats`                  | GET    | Aggregate evaluation statistics                    |

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

| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| **Embeddings**   | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector DB**    | ChromaDB                                |
| **Tree Index**   | NetworkX (DiGraph)                      |
| **Clustering**   | scikit-learn (Agglomerative)            |
| **LLM (local)**  | Ollama + Mistral                        |
| **LLM (cloud)**  | Groq API + Llama 3.3 70B                |
| **Fine-Tuning**  | TRL (DPOTrainer) + PEFT (LoRA) + BitsAndBytes |
| **Evaluation**   | RAGAS (Faithfulness, Relevancy, Precision) |
| **Backend**      | FastAPI + Pydantic                      |
| **Frontend**     | Gradio                                  |
| **Feedback**     | JSONL file storage                      |
| **Config**       | YAML                                    |
| **Data Source**   | arXiv API                               |

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
- [x] **Section 11** — Preference Dataset Creation (DPO pairs from feedback)
- [x] **Section 12** — Model Fine-Tuning (TRL/PEFT + DPO + LoRA)
- [x] **Section 13** — Continuous Learning Loop (auto feedback → train → deploy)
- [x] **Section 14** — Evaluation System (RAGAS metrics)
- [ ] **Section 15** — Backend System (unified FastAPI)
- [ ] **Section 16** — Frontend Interface (full Gradio UI)
- [ ] **Section 17** — DevOps & Scalability
- [ ] **Section 18** — Paper-Specific Learning & Debate

---

## Data Stats

| Metric                    | Value                 |
| ------------------------- | --------------------- |
| Papers indexed            | 204                   |
| Total chunks              | 148,986               |
| Sections                  | 1,349                 |
| Topics                    | 659                   |
| Papers with 4-level trees | 105                   |
| Embedding dimensions      | 384                   |
| Categories                | cs.AI, cs.LG, stat.ML |

---

## License

This project is for educational and research purposes.

---

_Built with RAPTOR RAG, Ollama, ChromaDB, and a lot of research papers._
