# P000233DS — AI-Powered Automated Assessment System
**EdGenAI Technologies (Spectrum Tuition) × RMIT University**

An AI-powered exam question generation system using GraphRAG (Graph-based Retrieval Augmented Generation) and large language models. Generates pedagogically grounded exam questions from university-level IT course material across 5 subject domains.

---

## Table of Contents
- [What This System Does](#what-this-system-does)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Adding New Documents](#adding-new-documents)
- [Known Limitations](#known-limitations)

---

## What This System Does

Educators provide a **learning objective**, **difficulty level**, and **question type**. The system:

1. Routes the query to the correct subject domain using semantic embedding similarity
2. Traverses a domain-specific knowledge graph to retrieve enriched context
3. Generates an exam question grounded in that context using Qwen 2.5 7B
4. Generates a sample answer for educator reference
5. Evaluates question quality using an LLM-as-Judge and optionally regenerates

**Supported domains:** Deep Learning · Database Systems · Software Engineering · Operating Systems · Computer Architecture

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11, Ubuntu 22.04, macOS |
| Python | 3.10.x |
| GPU | NVIDIA GPU with 8GB+ VRAM (recommended) |
| CUDA | 12.4+ |
| RAM | 16GB+ |
| Disk | 20GB+ free space |
| VRAM | 8GB (Qwen 2.5 7B uses ~6GB in 4-bit) |

> **Note:** CPU-only mode is supported but generation takes 5–10 minutes per request.

---

## Quick Start

### Step 1 — Clone the repository
```bash
git clone https://github.com/danish1804/graphrag-qgen.git
cd graphrag-qgen
git checkout danish-develop
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 4 — Configure environment
```bash
cp .env.example .env
# Edit .env and set GRAPHRAG_BASE_DIR to your project path
```

Windows example:
```
GRAPHRAG_BASE_DIR=D:\Projects\graphrag_qgen
```

Linux/Mac example:
```
GRAPHRAG_BASE_DIR=/home/user/graphrag_qgen
```

### Step 5 — Start the server
```bash
# Full mode — loads all 5 domain GraphRAG indexes (recommended)
python main_gr.py --mode full

# Standard mode — loads only Standard RAG (faster startup ~15s)
python main_gr.py --mode standard
```

Wait for:
```
[Server] Judge model ready.
[Server] Server is live.
```

### Step 6 — Test with health check
```bash
curl http://localhost:8000/health | python -m json.tool
```

Expected: `"status": "ok"` with all 5 domains `true`.

### Step 7 — Generate your first question
```bash
curl -X POST http://localhost:8000/graphrag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "learning_objective": "Explain how CNNs use parameter sharing",
    "difficulty": "medium",
    "question_type": "conceptual"
  }' | python -m json.tool
```

---

## Project Structure

```
graphrag_qgen/
├── main_gr.py                  # FastAPI server — primary entry point
├── main.py                     # Legacy single-doc server
├── main_multi.py               # Legacy multi-doc server
├── config.yml                  # All system parameters (edit here)
├── prepare_finetune_data.py    # Fine-tuning data pipeline
├── requirements.txt
├── .env                        # Local environment (not committed)
├── .env.example                # Template for .env
│
├── standard_rag/               # Standard RAG pipeline
│   └── README.md              → See standard_rag/README.md
│
├── graphrag/                   # GraphRAG pipeline (main system)
│   └── README.md              → See graphrag/README.md
│
├── utils/                      # Shared utilities
│   └── README.md              → See utils/README.md
│
├── evaluation/                 # LLM-as-Judge evaluation
│   └── README.md              → See evaluation/README.md
│
├── scripts/                    # Utility and maintenance scripts
│   └── README.md              → See scripts/README.md
│
├── data/
│   ├── raw/                    # Incoming PDFs (before renaming)
│   ├── tested/                 # Validated, renamed PDFs (master library)
│   ├── uploads/                # Staging area for admin uploads
│   └── document_registry.json # Registry of all PDFs and their domains
│
├── indexes/                    # Built knowledge graphs (not committed)
│   ├── graphrag_deep_learning.*
│   ├── graphrag_database.*
│   ├── graphrag_software_engineering.*
│   ├── graphrag_operating_systems.*
│   └── graphrag_computer_architecture.*
│
├── outputs/
│   └── evaluations/            # LLM judge evaluation JSON records
│
└── postman/                    # Postman collection for API testing
    └── P000233DS_GraphRAG.postman_collection.json
```

---

## Architecture Overview

```
Educator Request
      ↓
FastAPI Server (main_gr.py)
      ↓
Domain Router (embedding-based)
      ↓
      ├── Easy/Medium → Single Domain GraphRAG
      └── Hard → Multi-Domain GraphRAG (3+ domains combined)
            ↓
      Knowledge Graph Retrieval
      (seed nodes + neighbour traversal)
            ↓
      Qwen 2.5 7B Instruct (4-bit quantised)
      ├── generate_question()
      └── generate_answer()
            ↓
      LLM-as-Judge Evaluation
      (quality score → regenerate if below threshold)
            ↓
      JSON Response
      (question + sample_answer + judge_scores + retrieved_chunks)
```

**Detailed architecture:** See [graphrag/README.md](graphrag/README.md)

---

## API Endpoints

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server and domain status |
| GET | `/graphrag/status` | Domain graph statistics |
| GET | `/multi/status` | Multi-doc index statistics |

### GraphRAG (Primary)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/graphrag/generate` | Generate question via GraphRAG |
| POST | `/graphrag/upload` | Upload single doc for GraphRAG |
| POST | `/graphrag/rebuild/{domain}` | Rebuild a domain graph |

### Standard RAG
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload single doc for Standard RAG |
| POST | `/generate` | Generate question via Standard RAG |
| POST | `/multi/generate` | Generate from multi-doc index |
| POST | `/multi/rebuild` | Rebuild multi-doc index |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/admin/upload-document` | Upload PDF to a domain |
| POST | `/admin/rebuild-domain` | Incremental graph update |
| POST | `/admin/add-domain` | Register new subject domain |
| GET | `/admin/new-documents` | Show unindexed documents |

### Request Body — `/graphrag/generate`
```json
{
  "learning_objective": "Explain how CNNs use parameter sharing",
  "difficulty": "easy | medium | hard",
  "question_type": "recall | conceptual | application",
  "domain": "deep_learning | database | software_engineering | operating_systems | computer_architecture | null"
}
```

> If `domain` is omitted — auto-routed by embedding similarity.
> If `difficulty` is `hard` and `domain` is omitted — multi-domain retrieval activated.

### Response Structure
```json
{
  "question": "Generated exam question...",
  "metadata": {
    "learning_objective": "...",
    "difficulty": "medium",
    "question_type": "conceptual",
    "domain": "deep_learning",
    "sources": ["source_50.pdf", "source_62.pdf"],
    "retrieval_method": "GraphRAG",
    "key_concepts": ["parameter sharing", "feature extraction"],
    "concept_relationships": ["parameter sharing -> feature extraction"],
    "retrieved_chunks": {
      "chunk_1": {"text": "...", "source": "source_62.pdf", "type": "seed"},
      "chunk_2": {"text": "...", "source": "source_50.pdf", "type": "neighbour"}
    },
    "sample_answer": "Sample answer for educator reference...",
    "judge_scores": {
      "question_overall": 4.8,
      "answer_overall": 4.5,
      "retrieval_overall": 3.2,
      "attempts": 1,
      "eval_id": "eval_2026-05-03_..."
    }
  }
}
```

---

## Configuration

All system parameters are in `config.yml`. No code changes needed for tuning.

### Key Parameters

**Difficulty → Chunks Mapping**
```yaml
difficulty:
  easy:   { max_chunks: 3, seed_ratio: 0.80 }  # 2 seed + 1 neighbour
  medium: { max_chunks: 5, seed_ratio: 0.60 }  # 3 seed + 2 neighbour
  hard:   { max_chunks: 7, seed_ratio: 0.40,   # 3 seed + 4 neighbour
             multi_domain: true,
             domain_score_threshold: 0.5 }
```

**LLM as Judge**
```yaml
judge:
  judge_name: "Qwen/Qwen2.5-3B-Instruct"
  score_threshold: 3.5      # below this → regenerate
  max_regenerations: 2      # max retry attempts
```

**Graph Construction**
```yaml
graph:
  alpha:
    max_doc_freq: 0.85      # entity frequency threshold
    min_entity_freq: 2
  beta:
    min_edge_weight: 2      # minimum co-occurrence for edge
```

Full reference: see `config.yml` — every parameter is documented inline.

---

## Rebuilding Indexes From Scratch

If you need to rebuild the knowledge graphs (required on first run or after adding documents):

```bash
# Build all 5 domain graphs (takes 30-60 minutes)
python graphrag/graphrag_domain_indexer.py --domain all

# Build single domain
python graphrag/graphrag_domain_indexer.py --domain database

# Build multi-doc Standard RAG index
python standard_rag/multi_indexer.py
```

> **Important:** PDFs must be in `data/tested/` before building indexes.
> See [data/README.md](data/README.md) for PDF setup instructions.

---

## Adding New Documents

### Admin API Flow (Recommended)
```bash
# Step 1: Check what is already indexed
curl http://localhost:8000/admin/new-documents

# Step 2: Upload new PDF to a domain
curl -X POST "http://localhost:8000/admin/upload-document?domain=database" \
  -F "file=@/path/to/new_textbook.pdf"

# Step 3: Trigger incremental update (~2-5 minutes)
curl -X POST http://localhost:8000/admin/rebuild-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "database"}'
```

### Manual Flow
```bash
# Step 1: Place PDF in data/raw/
# Step 2: Rename
python scripts/rename_docs.py

# Step 3: Validate and move to tested/
python scripts/validate_docs.py

# Step 4: Upload via admin API (as above)
```

See [scripts/README.md](scripts/README.md) for details.

---

## Domain Knowledge Graph Statistics

| Domain | Documents | Nodes | Edges | Cross-Source Edges |
|--------|-----------|-------|-------|-------------------|
| Deep Learning | 15 | 5,330 | 17,256 | 3,255 |
| Database | 9 | 27,284 | 142,193 | 41,403 |
| Software Engineering | 14 | 7,782 | 24,428 | 6,861 |
| Operating Systems | 11 | 7,671 | 22,525 | 3,161 |
| Computer Architecture | 14 | 15,824 | 64,777 | 7,465 |

---

## Known Limitations

| Issue | Status | Fix |
|-------|--------|-----|
| MCQ format in some questions | Known | Fine-tuning planned |
| Low retrieval quality scores | Known | OCR artifacts in PDFs |
| Judge runs on CPU if VRAM < 1.5GB | Known | Reuse generator as judge |
| Multi-domain only for hard difficulty | By design | Samar's requirement |
| No per-educator isolated indexes | Planned | Educator copy API |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen 2.5 7B Instruct (4-bit QLoRA) |
| Judge LLM | Qwen 2.5 7B (reused) or 3B Instruct |
| Embeddings | BAAI/bge-large-en-v1.5 (1024-dim) |
| Vector Store | FAISS (IndexIVFFlat) |
| Knowledge Graph | NetworkX + spaCy NER |
| API Framework | FastAPI + Uvicorn |
| Entity Extraction | spaCy en_core_web_sm |

---

## Team

| Member | Role |
|--------|------|
| Mohammed Danish Alam | GraphRAG pipeline, FastAPI, domain routing, admin API |
| Aiswarya Sudhir | LLM-as-Judge, evaluation framework |
| Shrinidhi Sushil Kukde | Frontend UI, backlog management |
| Arun Attri | Data preprocessing, testing |
| Dharshini Arunraj | Data preprocessing, testing |

**Supervisor:** Ke Deng (RMIT) | **Industry:** Samar Shailendra (EdGenAI Technologies)

---

## Related Documentation

- [graphrag/README.md](graphrag/README.md) — GraphRAG pipeline internals
- [standard_rag/README.md](standard_rag/README.md) — Standard RAG pipeline
- [utils/README.md](utils/README.md) — Shared utilities
- [evaluation/README.md](evaluation/README.md) — LLM-as-Judge system
- [scripts/README.md](scripts/README.md) — Utility scripts
- [data/README.md](data/README.md) — Data structure and PDF management
