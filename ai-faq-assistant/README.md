# 🤖 AI FAQ Assistant — Modular RAG System

> **A production-grade Retrieval-Augmented Generation (RAG) chatbot** that answers employee FAQ questions strictly from a curated knowledge base — powered by **Google Gemini** for embeddings & generation and **FAISS** for blazing-fast vector similarity search.

---

## 📋 Table of Contents

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

---

## 🌟 Project Overview

The **AI FAQ Assistant** is a modular Retrieval-Augmented Generation (RAG) system built for enterprise internal use. It processes a rich knowledge base (`knowledge.txt`) and answers employee questions with high accuracy, referencing only the grounded context — never hallucinating.

It covers seven policy domains:
- 🕐 Working Hours & Attendance
- ✈️ Travel Reimbursement
- 💻 Equipment & Technology Allowance
- ⚖️ Code of Conduct & Workplace Ethics
- 🏥 Health Insurance & Wellness Benefits
- 📊 Performance Reviews & Compensation
- 🚪 Onboarding & Offboarding

---

## 🏗️ Architecture

### RAG Pipeline Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    INDEXING PHASE (one-time)                 │
│                                                              │
│  knowledge.txt                                               │
│      │                                                       │
│      ▼                                                       │
│  embeddings.py ──→ chunk_text()                              │
│      │              (sliding-window, 500 chars, 50 overlap)  │
│      │                                                       │
│      ▼                                                       │
│  embeddings.py ──→ get_embeddings_batch()                    │
│      │              (Gemini text-embedding-004)              │
│      │                                                       │
│      ▼                                                       │
│  vectorstore.py ─→ FAISSVectorStore.build_index()            │
│      │              (IndexFlatIP + L2 normalisation)         │
│      │                                                       │
│      ▼                                                       │
│  💾 faiss_store/index.faiss + chunks.json  (cached)          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE (per query)               │
│                                                              │
│  User Query (CLI)                                            │
│      │                                                       │
│      ▼                                                       │
│  vectorstore.py ─→ similarity_search(query, top_k=3)         │
│      │              embed query → cosine similarity          │
│      │                                                       │
│      ▼                                                       │
│  Top-K Relevant Context Chunks                               │
│      │                                                       │
│      ▼                                                       │
│  rag.py ─────────→ _build_prompt()                           │
│      │              (inject context + strict system prompt)  │
│      │                                                       │
│      ▼                                                       │
│  rag.py ─────────→ generate_answer()                         │
│      │              (Gemini gemini-2.5-flash)                │
│      │                                                       │
│      ▼                                                       │
│  Grounded Answer ──→ CLI Output                              │
└──────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
                ┌─────────┐
                │  app.py │  ← Entry point / orchestrator
                └────┬────┘
                     │ coordinates
          ┌──────────┼──────────┐
          │          │          │
   ┌──────▼──┐  ┌────▼────┐  ┌─▼──────┐
   │embed-   │  │vector-  │  │rag.py  │
   │dings.py │  │store.py │  │        │
   └──────┬──┘  └────┬────┘  └─┬──────┘
          │           │         │
          │    FAISS Index      │
          │           │         │
   Gemini Embeddings  │    Gemini Chat
   (text-embedding-004)    (gemini-2.5-flash)
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🛡️ **Zero-Hallucination Guard** | Strict system prompt enforces answers only from retrieved context |
| ⚡ **FAISS Vector Search** | Sub-millisecond cosine similarity search via L2-normalized IndexFlatIP |
| 🔢 **Gemini Embeddings** | `text-embedding-004` for high-quality semantic vector representations |
| 🧩 **Modular Architecture** | Clean separation: chunking, embedding, vector store, generation |
| 💾 **Index Caching** | FAISS index persisted to disk — rebuilds only when needed |
| 🔄 **Batch Embedding** | Rate-limit-safe batching with exponential back-off retry logic |
| 🐛 **Debug Mode** | `--debug` flag exposes retrieved chunks for transparency |
| 🔁 **Force Rebuild** | `--rebuild` flag to regenerate the index from scratch |
| 🌡️ **Low Temperature** | Gemini configured at `temperature=0.1` for factual, grounded responses |

---

## 📁 Project Structure

```
ai-faq-assistant/
│
├── app.py              # Main CLI entry point & pipeline orchestrator
├── knowledge.txt       # Multi-topic company FAQ knowledge base (7 sections)
├── embeddings.py       # Text chunking + Gemini embedding generation
├── vectorstore.py      # FAISSVectorStore class (build/save/load/search)
├── rag.py              # RAG prompt construction + Gemini generation
│
├── faiss_store/        # Auto-generated on first run (gitignored)
│   ├── index.faiss     # Binary FAISS vector index
│   └── chunks.json     # Serialised text chunks
│
├── requirements.txt    # Production Python dependencies
├── .env.example        # API key template (copy to .env)
├── .gitignore          # Excludes .env, faiss_store/, __pycache__/, etc.
└── README.md           # This file
```

---

## 🔍 Module Breakdown

### `knowledge.txt`
A richly detailed 7-section company policy document (~7,000 words) covering:
- **Section 1**: Working Hours, Remote Work, Overtime, Leave Types, Public Holidays
- **Section 2**: Travel Reimbursement, Per Diems, Expense Procedures, International Travel
- **Section 3**: Equipment Provision, Home Office Allowance, BYOD, Software Budget
- **Section 4**: Code of Conduct, Anti-Harassment, Social Media Policy, IP Ownership
- **Section 5**: Health Insurance, Wellness Stipend, EAP, Dental/Vision, Preventive Care
- **Section 6**: Performance Ratings, Salary Increases, Bonuses, Promotion Policy
- **Section 7**: Onboarding Checklist, Probationary Period, Resignation & Offboarding

---

### `embeddings.py`

| Function | Purpose |
|---|---|
| `chunk_text(text, chunk_size, chunk_overlap)` | Sliding-window text splitter with paragraph-aware boundary detection |
| `get_embedding(text, model, task_type)` | Single-text embedding via Gemini API with retry/back-off |
| `get_embeddings_batch(texts, batch_size, ...)` | Batch embedding with inter-batch rate-limit delays |

**Design note**: Chunking respects paragraph boundaries first, only falling back to character-level splitting for oversized paragraphs. This preserves semantic coherence within chunks.

---

### `vectorstore.py` — `FAISSVectorStore`

| Method | Purpose |
|---|---|
| `build_index(chunks)` | Embeds chunks, L2-normalises, builds `IndexFlatIP` |
| `save(index_path, chunks_path)` | Writes `.faiss` binary + `.json` chunk list |
| `load(index_path, chunks_path)` | Restores from disk; returns `False` on cache miss |
| `similarity_search(query, top_k)` | Embeds query, normalises, returns top-k chunk strings |

**Design note**: `IndexFlatIP` (inner product) on L2-normalised vectors = cosine similarity. This is more semantically meaningful than raw L2 distance for text retrieval.

---

### `rag.py`

| Component | Detail |
|---|---|
| `RAG_SYSTEM_PROMPT` | Strict 6-rule system instruction preventing hallucination |
| `initialise_gemini(api_key)` | Configures SDK, returns model (falls back from Flash 2.5 → 1.5) |
| `_build_prompt(query, chunks)` | Formats numbered context sections + question delimiters |
| `generate_answer(query, chunks, model)` | Calls Gemini, returns cleaned answer string |

**Design note**: `temperature=0.1` keeps the model anchored to facts. The system prompt explicitly forbids external knowledge and mandates a graceful refusal phrase.

---

### `app.py`

Orchestrates the full lifecycle:
1. Parses CLI flags (`--debug`, `--rebuild`)
2. Loads `.env` and validates `GEMINI_API_KEY`
3. Initialises Gemini model
4. Loads/builds the FAISS vector store
5. Runs the interactive loop: input → retrieve → generate → display

---

## ⚙️ Prerequisites

- Python **3.10+** (required for `list[str]` built-in generics)
- A **Google Gemini API Key** — get one at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Internet connection for API calls

---

## 🚀 Installation

### 1. Clone or navigate to the project directory

```bash
cd ai-faq-assistant
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

> **Note**: `faiss-cpu` requires a 64-bit Python installation. On Windows, ensure you have the Microsoft C++ Build Tools if installation fails.

---

## 🔑 Environment Configuration

Copy the example environment file and add your API key:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=AIzaSy...your_actual_key_here...
```

> ⚠️ **Security**: Never commit `.env` to version control. It is already listed in `.gitignore`.

---

## 💬 Usage Guide

### Standard run (builds index on first launch)

```bash
python app.py
```

On first run, the assistant will:
1. Read `knowledge.txt`
2. Split it into ~N text chunks
3. Call the Gemini Embedding API to vectorise all chunks
4. Build and save the FAISS index to `faiss_store/`

On subsequent runs, the cached index is loaded in seconds.

---

### Debug mode (shows retrieved context chunks)

```bash
python app.py --debug
```

---

### Force rebuild the FAISS index

```bash
python app.py --rebuild
```

Use this if you update `knowledge.txt` and want the index refreshed.

---

## 💡 Sample Interactions

### ✅ Grounded answer

```
❓ Your question: How many days of annual leave do employees get?

🤖 Answer:
According to Section 1.4 (Leave Entitlement), full-time employees at Acme
Corporation are entitled to 25 working days of Annual Leave per calendar
year. This is pro-rated for employees who join mid-year.
```

---

### ✅ Multi-fact answer

```
❓ Your question: What does the health plan cover for dental and vision?

🤖 Answer:
Based on Section 5.1, the company health plan includes:
- Dental coverage with an annual maximum of $2,000 per employee.
- Vision coverage of $300 per year for glasses or contact lenses.
```

---

### ✅ Graceful refusal (out-of-scope question)

```
❓ Your question: What is the company's stock price today?

🤖 Answer:
I am sorry, but that information is not available in the knowledge base.
```

---

### ✅ Policy boundary query

```
❓ Your question: Can I expense alcohol during a business trip?

🤖 Answer:
No. According to Section 2.2 (Non-Reimbursable Expenses), alcohol and
personal entertainment expenses are explicitly listed as non-reimbursable
items and will not be covered by the company.
```

---

## 🔄 How RAG Works Here

```
User Query
    │
    ▼ (1) Embed query with text-embedding-004
    │
    ▼ (2) Cosine similarity search in FAISS index
    │       → Returns top-3 most relevant knowledge chunks
    │
    ▼ (3) Inject chunks as context into strict RAG prompt
    │       → System prompt enforces ground-truth-only answers
    │
    ▼ (4) Gemini gemini-2.5-flash generates answer
    │       → temperature=0.1 for factual, low-variance output
    │
    ▼ (5) If no relevant context → graceful refusal message
```

The key insight: the model **never answers from its pre-trained knowledge alone**. Every response is anchored to the retrieved text from `knowledge.txt` — ensuring accuracy, auditability, and safety.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `google-generativeai` | ≥ 0.8.0 | Gemini embedding + generation API |
| `faiss-cpu` | ≥ 1.8.0 | High-speed vector similarity search |
| `numpy` | ≥ 1.26.0 | Matrix operations for embedding vectors |
| `python-dotenv` | ≥ 1.0.0 | `.env` file loading for secret management |

---

*Built for ACME Corporation — Knots AI Engineering Foundation, Day 12.*
