# RAPTOR Research Assistant — Project Plan

## 1. Data Ingestion & Collection

- Fetch research papers from arXiv API.
- Store metadata: title, authors, abstract, category, pdf_url, published_date.
- Download and organize PDFs.

## 2. PDF Processing & Text Chunking

- Extract text from PDFs using PyMuPDF/Unstructured.
- Parse into sections, paragraphs, and 300–500 token chunks.
- Store processed text and chunk metadata.

## 3. Embedding Generation

- Use SentenceTransformers/BGE to embed each chunk.
- Store: chunk_text, embedding_vector, paper_metadata.

## 4. Vector Database

- Set up Chroma to store embeddings and metadata.
- Enable fast semantic search and retrieval.

## 5. RAPTOR Hierarchical Index

- Build a tree structure for each paper (topic → summary → section → chunk).
- Use NetworkX for tree management and traversal.

## 6. Retrieval Engine

- On user query:
  - Vector search for relevant chunks.
  - RAPTOR tree traversal for hierarchical context.
  - Return top relevant nodes/chunks.
  - (MCP server can orchestrate hybrid retrieval: vector search + RAPTOR tree traversal)

## 7. Prompt Construction

- Assemble prompt for LLM using retrieved context and user question.
- (MCP server can standardize prompt/context assembly for different models)

## 8. LLM Reasoning

- Use Llama 3, Mixtral, or similar for:
  - Summarization, comparison, Q&A, idea generation.
- Return answer to user.
- (MCP server can route inference to multiple models and manage responses)

## 9. Two-Way Chatbot (with Session Support)

- Interactive chat interface (Gradio/Streamlit).
- Session management:
  - Assign unique session ID per chat.
  - Store chat history (questions, answers, paper references) per session.
  - Pass session history to LLM for context-aware responses.
- Display citations and context for each answer.

## 10. Feedback System

- Collect user feedback after each answer (helpful, incorrect, hallucination, correction).
- Store feedback with context and answer.

## 11. Preference Dataset Creation

- Convert feedback into preference data (prompt, chosen, rejected).
- Store for RLHF training.

## 12. Model Fine-Tuning

- Use TRL/PEFT for preference-based fine-tuning (DPO).
- Train LLM to improve reasoning and explanation.
- (MCP server can serve both base and fine-tuned models for RLHF/DPO workflows)

## 13. Continuous Learning Loop

- Automate: user question → retrieval → LLM response → feedback → preference dataset → fine-tuning → improved model.
- (MCP server can automate feedback collection, model selection, and evaluation)

## 14. Evaluation System

- Use RAGAS to evaluate faithfulness, answer accuracy, context relevance, hallucination rate.
- (MCP server can run queries across multiple models for evaluation/comparison)

## 15. Backend System

- FastAPI server with endpoints:
  - /chat (with session support)
  - /retrieve
  - /feedback
  - /train
- (Backend can call MCP server for all model and retrieval tasks)

## 16. Frontend Interface

- Gradio/Streamlit UI:
  - Chat window with session support
  - Paper viewer
  - Citation display
  - Feedback buttons
  - Upload new papers directly through the UI for ingestion, learning, and debate

## 18. Paper-Specific Learning & Debate

- When a new paper is added, the system should:
  - Ingest and process the paper as usual.
  - Allow the model to answer questions and engage in debate using only the knowledge learned from that specific paper (isolate context to the new paper).
  - Fine-tune the model on the new paper, so it can learn and adapt its responses based on the content of the paper.
  - Enable users to test, query, and challenge the model specifically on the newly ingested paper, ensuring the model's understanding and reasoning are grounded in that paper's content.

## 17. DevOps & Scalability

- Use config.yaml for settings.
- Organize code for scaling and extension.
- Write tests for each module.
