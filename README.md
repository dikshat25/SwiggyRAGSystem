# # Swiggy RAG — Intelligent Document Q&A System

> A fully **local** Retrieval-Augmented Generation (RAG) system built to intelligently query the **Swiggy Annual Report FY 2023-24** using open-source models — no external APIs, no cloud LLM costs.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Why Local RAG?](#why-local-rag)
- [Installation](#installation)
- [Usage](#usage)
- [Batch Testing](#batch-testing)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements an **RAG pipeline** that allows users to ask natural language questions about Swiggy's Annual Report and receive precise, context-aware answers. The system extracts structured and unstructured content from a 170-page PDF, embeds it locally, and uses a two-stage retrieval-reranking pipeline to fetch the most relevant context for each query.

The entire pipeline runs **100% locally** using open-source HuggingFace models — making it privacy-preserving, free to run, and fully reproducible.

---

## How It Works

The RAG pipeline follows a **5-stage process**:

```
PDF Input
   │
   ▼
[Stage 1] Document Processing
   └── pdfplumber extracts text + tables from all 170 pages
   └── RecursiveCharacterTextSplitter → 232 optimized chunks (size=1100, overlap=75)
   │
   ▼
[Stage 2] Embedding & Indexing
   └── sentence-transformers/all-MiniLM-L6-v2 generates 384-dim vectors locally
   └── ChromaDB stores vectors on disk (./chroma_db)
   │
   ▼
[Stage 3] Retrieval
   └── Cosine similarity search → Top-8 candidate chunks retrieved
   │
   ▼
[Stage 4] Reranking
   └── cross-encoder/ms-marco-MiniLM-L-6-v2 scores each (query, chunk) pair
   └── Top-3 most relevant chunks selected
   │
   ▼
[Stage 5] Answer Generation
   └── EnhancedAnswerGenerator extracts structured data via regex patterns
   └── Returns answer + confidence score + source page citations
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   EnhancedSwiggyRAG                     │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ DocumentProcessor│    │   ConversationHistory    │   │
│  │  - pdfplumber    │    │  - Session tracking      │   │
│  │  - Text + Tables │    │  - Q&A history log       │   │
│  │  - 232 chunks    │    └──────────────────────────┘   │
│  └────────┬─────────┘                                   │
│           │                                             │
│  ┌────────▼─────────┐    ┌──────────────────────────┐   │
│  │    Retriever     │    │        Reranker           │   │
│  │  - MiniLM-L6-v2  │───▶│  - ms-marco-MiniLM-L-6   │   │
│  │  - ChromaDB      │    │  - CrossEncoder scoring  │   │
│  │  - Top-8 results │    │  - Top-3 selected        │   │
│  └──────────────────┘    └────────────┬─────────────┘   │
│                                       │                 │
│                          ┌────────────▼─────────────┐   │
│                          │  EnhancedAnswerGenerator  │   │
│                          │  - Pattern extraction    │   │
│                          │  - Structured data       │   │
│                          │  - Confidence scoring    │   │
│                          └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

##  Key Features

| Feature | Details |
|---|---|
| PDF Parsing | Extracts both **text and tables** from every page using `pdfplumber` |
| Smart Chunking | `chunk_size=1100`, `overlap=75` — tuned for 232 optimal chunks |
| Local Embeddings | `all-MiniLM-L6-v2` — 384-dim vectors, runs fully offline after first download |
| Vector Store | ChromaDB with **disk persistence** (`./chroma_db`) — no re-embedding on restart |
| Two-Stage Retrieval | Bi-encoder for speed → Cross-encoder for precision |
| Structured Extraction | Regex patterns for Revenue, Profit, Employees, Growth, Cities, Users |
| Conversation History | Tracks full Q&A session with timestamps and confidence scores |
| Source Citations | Every answer includes page number + relevance score |

---

## Why Local RAG?

Most RAG tutorials rely on cloud LLMs (OpenAI, Anthropic, Gemini). This project deliberately avoids that:

- **Privacy**: Your document never leaves your machine
- **Cost**: Zero inference cost — run thousands of queries for free
- **No Rate Limits**: Query as much as needed without throttling
- **Reproducibility**: Identical results every run, no model versioning issues
- **Offline Capability**: Works without internet after initial model download

The only internet usage is the **one-time model download** from HuggingFace on first run. After that, the system is fully air-gapped.

## Installation

Run in **Google Colab** or a local Python environment:

```bash
pip install langchain langchain-community langchain-chroma
pip install langchain-huggingface sentence-transformers
pip install pdfplumber pypdf langchain-text-splitters
pip install torch numpy tqdm
```

---

##  Usage

### Step 1 — Upload PDF
```python
from google.colab import files
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
```

### Step 2 — Initialize System
```python
rag = EnhancedSwiggyRAG(pdf_path)
```

### Step 3 — Ask Questions
```python
result = rag.answer_question("What was Swiggy's revenue in FY24?")
rag.display_answer(result)
```

### Sample Output
```
================================================================================
ANSWER FROM SWIGGY ANNUAL REPORT
================================================================================

Revenue: ₹11,247 Crores
Profit: ₹(2,350) Crores

Full Context:
[Relevant passage from the annual report...]

Confidence: 87.3%

Found in 3 location(s):
[Source 1] Page 42 | Relevance: 87.3%
[Source 2] Page 78 | Relevance: 74.1%
[Source 3] Page 91 | Relevance: 68.9%
================================================================================
```

## Batch Testing

To run multiple queries at once:

```python
test_queries = [
    "What was Swiggy's revenue in FY2024?",
    "How many cities does Swiggy operate in?",
    "What are the key risks mentioned in the report?",
    "How many restaurant partners does Swiggy have?",
    "What is Swiggy's net loss for FY24?",
    "What is Swiggy's Instamart business?",
    "How many active users does Swiggy have?",
    "What were Swiggy's key highlights of FY2024?"
]

results = []
for query in test_queries:
    result = rag.answer_question(query)
    results.append({
        "query": query,
        "confidence": result["confidence"],
        "query_type": result["query_type"],
        "has_structured_data": bool(result["structured_data"])
    })
    rag.display_answer(result)

# Summary Report
print(f"\n{'='*60}")
print(f"BATCH TEST SUMMARY")
print(f"{'='*60}")
print(f"Total Queries   : {len(results)}")
print(f"Avg Confidence  : {sum(r['confidence'] for r in results)/len(results):.1%}")
high_conf = sum(1 for r in results if r['confidence'] > 0.7)
print(f"High Confidence : {high_conf}/{len(results)} queries")
```

## Tech Stack

| Component | Library / Model |
|---|---|
| PDF Extraction | `pdfplumber` |
| Text Splitting | `langchain-text-splitters` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | `ChromaDB` (local disk) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| RAG Framework | `LangChain` |
| Deep Learning | `PyTorch` |
| Runtime | Google Colab / Jupyter |
| Language | Python 3.12 |

---
