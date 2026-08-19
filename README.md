#  AI FAQ Assistant — Modular RAG System

> A **production-grade Retrieval-Augmented Generation (RAG) chatbot** that answers employee FAQ questions strictly from a curated knowledge base — powered by **Google Gemini** for embeddings & generation and **FAISS** for blazing-fast vector similarity search.

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Module Breakdown](#-module-breakdown)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Environment Configuration](#-environment-configuration)
- [Usage Guide](#-usage-guide)
- [Sample Interactions](#-sample-interactions)
- [How RAG Works Here](#-how-rag-works-here)
- [Dependencies](#-dependencies)

---

##  Project Overview

The **AI FAQ Assistant** is a modular Retrieval-Augmented Generation (RAG) system built for enterprise internal use. It processes a rich knowledge base (`knowledge.txt`) and answers employee questions with high accuracy — referencing only the grounded context, never hallucinating.

It covers **seven company policy domains**:

| # | Domain |
|---|---|
| 1 |  Working Hours & Attendance |
| 2 |  Travel Reimbursement |
| 3 |  Equipment & Technology Allowance |
| 4 |  Code of Conduct & Workplace Ethics |
| 5 |  Health Insurance & Wellness Benefits |
| 6 |  Performance Reviews & Compensation |
| 7 |  Onboarding & Offboarding |

---

##  Architecture

### RAG Pipeline — Indexing Phase (one-time)

```
knowledge.txt
     │
     ▼
embeddings.py ──► chunk_text()
                  (sliding-window, 500 chars, 50 overlap)
     │
     ▼
embeddings.py ──► get_embeddings_batch()
                  (Gemini text-embedding-004)
     │
     ▼
vectorstore.py ─► FAISSVectorStore.build_index()
                  (IndexFlatIP + L2 normalisation = cosine similarity)
     │
     ▼
  faiss_store/index.faiss  +  faiss_store/chunks.json  (cached to disk)
```

### RAG Pipeline — Inference Phase (per query)

```
User Query  (CLI)
     │
     ▼  embed query with text-embedding-004
     │
     ▼
vectorstore.py ─► similarity_search(query, top_k=3)
                  cosine similarity → Top-K relevant chunks
     │
     ▼
rag.py ─────────► _build_prompt()
                  inject context + strict system prompt
     │
     ▼
rag.py ─────────► generate_answer()
                  Gemini gemini-2.5-flash  (temperature=0.1)
     │
     ▼
Grounded Answer ──► CLI Output
```

### Component Diagram

```
              ┌─────────┐
              │  app.py │   ← Orchestrator / Entry point
              └────┬────┘
                   │ coordinates
        ┌──────────┼──────────┐
        │          │          │
 ┌──────▼───┐ ┌────▼────┐ ┌──▼─────┐
 │embed-    │ │vector-  │ │rag.py  │
 │dings.py  │ │store.py │ │        │
 └──────┬───┘ └────┬────┘ └──┬─────┘
        │          │          │
   Gemini      FAISS       Gemini
   Embeddings  Index       Chat Model
   (004)                  (2.5-flash)
```

---

##  Key Features

| Feature | Description |
|---|---|
|  **Zero-Hallucination Guard** | Strict system prompt enforces answers only from retrieved context |
|  **FAISS Vector Search** | Sub-millisecond cosine similarity via L2-normalized `IndexFlatIP` |
|  **Gemini Embeddings** | `text-embedding-004` for high-quality semantic representations |
|  **Modular Architecture** | Clean separation: chunking → embedding → vector store → generation |
|  **Index Caching** | FAISS index persisted to disk — rebuilds only when needed |
|  **Batch Embedding** | Rate-limit-safe batching with exponential back-off retry logic |
|  **Debug Mode** | `--debug` flag exposes retrieved chunks for full transparency |
|  **Force Rebuild** | `--rebuild` flag to regenerate the index from scratch |
|  **Low Temperature** | Gemini at `temperature=0.1` for factual, low-variance responses |
|  **Model Fallback** | Auto-falls back from `gemini-2.5-flash` → `gemini-1.5-flash` |

---

##  Project Structure

```
DAY12_KAIEF26042/
│
├── app.py              # Main CLI entry point & pipeline orchestrator
├── embeddings.py       # Text chunking + Gemini embedding generation
├── vectorstore.py      # FAISSVectorStore class (build/save/load/search)
├── rag.py              # RAG prompt builder + Gemini generation layer
├── knowledge.txt       # Multi-topic company FAQ knowledge base (7 sections)
│
├── requirements.txt    # Production Python dependencies
├── .env.example        # API key template — copy to .env
├── .gitignore          # Excludes .env, venv/, faiss_store/, __pycache__/
└── README.md           # This file
```

> **Auto-generated on first run** (gitignored):
> ```
> faiss_store/
> ├── index.faiss     # Binary FAISS vector index
> └── chunks.json     # Serialised text chunks metadata
> ```

---

##  Module Breakdown

### `knowledge.txt`
A richly detailed **7-section company policy document** covering leave entitlements, travel reimbursement limits, equipment allowances, code of conduct, health insurance, performance ratings, salary increases, and offboarding procedures.

---

### `embeddings.py`

| Function | Purpose |
|---|---|
| `chunk_text(text, chunk_size, chunk_overlap)` | Paragraph-aware sliding-window text splitter |
| `get_embedding(text, model, task_type)` | Single-text embedding via Gemini API with retry/back-off |
| `get_embeddings_batch(texts, batch_size, ...)` | Batch embedding with inter-batch rate-limit delays |

> **Design note**: Chunking respects paragraph boundaries first, only falling back to character-level splitting for oversized paragraphs — preserving semantic coherence within each chunk.

---

### `vectorstore.py` — `FAISSVectorStore`

| Method | Purpose |
|---|---|
| `build_index(chunks)` | Embeds all chunks, L2-normalises, builds `IndexFlatIP` |
| `save(index_path, chunks_path)` | Writes `.faiss` binary + `.json` chunk list to disk |
| `load(index_path, chunks_path)` | Restores index from disk; returns `False` on cache miss |
| `similarity_search(query, top_k)` | Embeds query, normalises, returns top-k chunk strings |

> **Design note**: `IndexFlatIP` on L2-normalised vectors is mathematically equivalent to cosine similarity — more semantically meaningful than raw L2 distance for text retrieval.

---

### `rag.py`

| Component | Detail |
|---|---|
| `RAG_SYSTEM_PROMPT` | Strict 6-rule instruction preventing hallucination |
| `initialise_gemini(api_key)` | Configures Gemini SDK; falls back from Flash 2.5 → 1.5 |
| `_build_prompt(query, chunks)` | Formats numbered context sections + question delimiters |
| `generate_answer(query, chunks, model)` | Calls Gemini API, returns cleaned answer string |

---

### `app.py`

Orchestrates the full lifecycle:
1. Parses CLI flags (`--debug`, `--rebuild`)
2. Loads `.env` and validates `GEMINI_API_KEY`
3. Initialises Gemini model
4. Loads/builds the FAISS vector store
5. Runs the interactive loop: **input → retrieve → generate → display**

---

##  Prerequisites

- **Python 3.10+** — required for built-in `list[str]` type hints
- **Google Gemini API Key** — get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Internet connection for API calls

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/Habibxcode/DAY12_KAIEF26042.git
cd DAY12_KAIEF26042
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Environment Configuration

Copy the example file and add your API key:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your key:

```env
GEMINI_API_KEY=AIzaSy...your_actual_key_here...
```

>  **Security**: Never commit `.env` to version control. It is already listed in `.gitignore`.

---

##  Usage Guide

### Standard run

```bash
python app.py
```

**On first run**, the assistant will automatically:
1. Read `knowledge.txt`
2. Split it into overlapping text chunks
3. Call the Gemini Embedding API to vectorise all chunks
4. Build and cache the FAISS index to `faiss_store/`

**On subsequent runs**, the cached index loads instantly — no API calls needed for indexing.

---

### Debug mode — see retrieved context

```bash
python app.py --debug
```

---

### Force rebuild the FAISS index

```bash
python app.py --rebuild
```

> Use this whenever you update `knowledge.txt`.

---

##  Sample Interactions

### Grounded policy answer

```
❓ Your question: How many days of annual leave do employees get?

🤖 Answer:
According to Section 1.4 (Leave Entitlement), full-time employees are
entitled to 25 working days of Annual Leave per calendar year,
pro-rated for employees who join mid-year.
```

---

###  Multi-fact health benefits query

```
❓ Your question: What does the health plan cover for dental and vision?

🤖 Answer:
Based on Section 5.1, the company health plan includes:
- Dental coverage: annual maximum of $2,000 per employee.
- Vision coverage: $300 per year for glasses or contact lenses.
```

---

###  Travel reimbursement boundary

```
❓ Your question: Can I expense alcohol during a business trip?

🤖 Answer:
No. Per Section 2.2 (Non-Reimbursable Expenses), alcohol and personal
entertainment expenses are explicitly not covered and will not be
reimbursed by the company.
```

---

### Graceful refusal — out-of-scope question

```
❓ Your question: What is the company's stock price today?

🤖 Answer:
I am sorry, but that information is not available in the knowledge base.
```

---

##  How RAG Works Here

```
User Query
    │
    ▼  (1) Embed query  →  text-embedding-004
    │
    ▼  (2) Cosine similarity search in FAISS
    │       Returns top-3 most relevant knowledge chunks
    │
    ▼  (3) Inject chunks as context into strict RAG prompt
    │       System prompt: "Answer ONLY from the context below"
    │
    ▼  (4) Gemini gemini-2.5-flash generates the answer
    │       temperature=0.1 → factual, low-variance output
    │
    ▼  (5) No relevant context found?
            → "I am sorry, but that information is not available
               in the knowledge base."
```

The key principle: the model **never answers from its pre-trained knowledge alone**. Every response is anchored to retrieved text from `knowledge.txt` — ensuring accuracy, auditability, and safety.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `google-generativeai` | ≥ 0.8.0 | Gemini embedding + generation API |
| `faiss-cpu` | ≥ 1.8.0 | High-speed vector similarity search |
| `numpy` | ≥ 1.26.0 | Matrix operations for embedding vectors |
| `python-dotenv` | ≥ 1.0.0 | Secure `.env` file loading |

---

*Built for **ACME Corporation** — Knots AI Engineering Foundation Cohort 1, Day 12.*
