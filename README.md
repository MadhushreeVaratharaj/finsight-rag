# FinSight — Financial Document Q&A

## Overview

This repository contains a Retrieval-Augmented Generation (RAG) application for querying financial documents using natural language. The system is designed to return accurate, source-cited answers grounded strictly in uploaded documents — with no hallucination of figures, dates, or regulatory requirements.

The motivation came from a real problem. During my time as a Programmer Analyst at Cognizant, I worked on Temenos T24 — a core banking system used by financial institutions globally. Learning the system meant navigating the Temenos Learning 
Community (TLC), a knowledge base with hundreds of pages of module documentation, transaction workflows, and system configurations. Finding one specific answer meant manually searching through all of it. When I learned about RAG during my 
MSc in Data Science at SUTD, that experience came back immediately. FinSight is what I wish I had then — upload the document, ask your question, get the answer with the exact page it came from.

---

## Problem Statement

Given one or more financial PDFs — annual reports, MAS regulatory circulars, loan agreements, or technical banking documentation — the system should:

- Accept natural language questions about document contents
- Retrieve only the most relevant sections of the document
- Generate a precise answer grounded in the retrieved text
- Cite the source document and page number for every answer
- Track query latency and retrieval quality for observability

---

## System Design

The pipeline is structured as a sequential, independently testable set of stages:

1. PDF ingestion using PyPDF — one Document object per page, with metadata preserved
2. Text chunking using RecursiveCharacterTextSplitter (chunk size 1000, overlap 150)
3. Embedding using sentence-transformers/all-MiniLM-L6-v2 — runs fully locally
4. Vector indexing using FAISS — persisted to disk after first build
5. Semantic retrieval of the top-4 most relevant chunks per query
6. Answer generation using Llama 3.1 8B Instant via Groq at temperature 0
7. Source citation extraction — document name and page number returned with every answer
8. Query logging to JSONL with latency, source count, and relevance score
9. Live monitoring dashboard showing latency trends and retrieval patterns per session

Each stage is contained in a separate, clearly named module.

---

## Why RAG for Financial Documents

Financial documents change constantly — new MAS circulars, quarterly reports, updated loan terms. A fine-tuned model would require retraining every time  the knowledge base changes. RAG requires no retraining — swap the document, knowledge updates instantly.

More importantly, financial and regulatory Q&A demands traceability. Every answer must be traceable to a source. This is not a nice-to-have in banking environments — it is a compliance requirement.

---

## Key Design Decisions

**Chunk size 1000 with overlap 150** — Financial clauses and regulatory requirements tend to span multiple sentences. Smaller chunks lose context 
mid-clause. Larger chunks dilute retrieval relevance by mixing unrelated content. 150-character overlap ensures clause-boundary splits do not lose critical context.

**Local embeddings (MiniLM-L6-v2)** — Embedding runs entirely on the local machine. No document content is sent to an external API during indexing. 
For financial and regulatory documents, data confidentiality during processing is non-negotiable.

**temperature=0** — Financial Q&A requires deterministic output. Any non-zero temperature introduces variation in figures, dates, and regulatory requirements across repeated queries — unacceptable in a compliance context.

**FAISS persisted to disk** — The vector index is saved after first build and reloaded on subsequent runs. This avoids re-embedding on every session
restart, which is both slow and unnecessary when the document set has not changed.

**Keyword-based relevance scoring** — After evaluation, cosine similarity on re-embedded chunks produced inconsistent scores because it re-embeds 
outside of FAISS's internal distance calculations. Switched to keyword overlap across retireved chunks — simpler, faster, and more interpretable. A score 1.0 means all four retrieved chunks contained keywords from the query.

---

## MLOps Layer

The monitoring tab exposes live session metrics that reflect how a production AI system would be instrumented:

- Per-query latency tracking with colour-coded thresholds
- Chunks retrieved per query — a drop signals retrieval quality issues
- Relevance score per query — keyword overlap across retrieved chunks
- Full query log in JSONL format — one record per query, timestamped
- Pipeline configuration exposed — chunk size, overlap, model, top-k

In a production banking environment, every AI interaction requires an audit trail. The logging layer reflects that requirement directly.

---

## Stack

| Layer | Technology | Reason |
|---|---|---|
| LLM | Llama 3.1 8B Instant (Groq) | Fast, free tier, sufficient for document Q&A |
| Orchestration | LangChain 0.1.20 | Production RAG patterns |
| Embeddings | all-MiniLM-L6-v2 | Free, local, no data leaves machine |
| Vector store | FAISS | No infrastructure needed, fast retrieval |
| Frontend | Streamlit | Clean UI, fast deployment |
| Containerisation | Docker | Reproducible deployment |

---

## Project Structure

finsight-rag/

- app.py              # Streamlit frontend — chat, monitoring, about tabs

- rag_pipeline.py     # Core RAG engine — load, chunk, embed, retrieve, generate

- logger.py           # MLOps logging — query tracking and session statistics

- requirements.txt    # Pinned dependencies for reproducibility

- Dockerfile          # Container definition for deployment

- README.md

---

## Assumptions and Limitations

- **PDF only**: Current implementation supports PDF uploads only. Plain text and HTML documents are not yet handled.
- **In-session context**: The system answers based on uploaded documents only. It has no memory of previous sessions.
- **Relevance score**: Keyword overlap is a proxy metric, not a formal retrieval quality measure. It indicates whether retrieved chunks contained query terms — not whether the answer is correct.
- **Single-user design**: FAISS index is built per session. Multi-user concurrent access is not supported in this implementation.
- **Groq free tier**: Subject to rate limits on the free tier. Response may queue under heavy use.

---

## How to Run

```bash
git clone https://github.com/MadhushreeVaratharaj/finsight-rag
cd finsight-rag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your-key-here" > .env
streamlit run app.py
```

Upload any financial PDF from the sidebar and start asking questions.

---

## Author

Built as a practical exploration of retrieval-augmented generation for financial document analysis, motivated by real experience navigating large technical documentation in a banking environment. Feedback and questions welcome.