# standard_rag/ : Standard RAG Pipeline

A flat chunk retrieval pipeline using FAISS vector search. No knowledge graph : retrieves chunks by embedding similarity only. Used as baseline comparison against GraphRAG and for single-document quick queries.

---

## GraphRAG vs Standard RAG : When to Use Which

| Use Case | Recommended Pipeline |
|----------|---------------------|
| Educator uploads their own PDF for one-off questions | Standard RAG (`/upload` + `/generate`) |
| Questions from EdGenAI's curated domain corpus | GraphRAG (`/graphrag/generate`) |
| Quick baseline comparison | Standard RAG multi-doc (`/multi/generate`) |
| Hard cross-domain questions | GraphRAG multi-domain (`/graphrag/generate` hard) |

**Key difference:**
```
Standard RAG:
  Query -> FAISS similarity -> top-k chunks -> LLM
  Retrieves: chunks similar to the query

GraphRAG:
  Query -> FAISS similarity -> seed nodes -> graph traversal
         -> seed chunks + neighbour chunks -> LLM
  Retrieves: similar chunks PLUS related concept chunks
```

---

## Files

```
standard_rag/
├── __init__.py
├── pipeline.py         # Single-document RAG pipeline
├── multi_indexer.py    # Builds combined multi-doc FAISS index
└── multi_pipeline.py   # Multi-document RAG pipeline
```

---

## pipeline.py : Single Document RAG

Handles upload and question generation for a single educator-uploaded PDF. Index is built in memory : not saved to disk.

### Class: RAGPipeline

```python
from standard_rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load_models()          # loads BAAI embedder + Qwen 7B
pipeline.ingest("my_notes.pdf") # chunks + embeds into memory
result = pipeline.generate(
    learning_objective = "Explain virtual memory",
    difficulty         = "medium",
    question_type      = "conceptual"
)
```

#### load_models()

Loads BAAI/bge-large-en-v1.5 and Qwen 2.5 7B. Called once at server startup : shared with GraphRAG pipeline.

#### ingest(file_path) -> dict

Loads PDF/PPTX, chunks it, embeds chunks, builds temporary FAISS index in memory.

```python
result = pipeline.ingest("data/uploads/lecture.pdf")
# {
#     "file_name":   "lecture.pdf",
#     "total_chunks": 87,
#     "index_name":  "lecture_rag"
# }
```

#### generate(learning_objective, difficulty, question_type) -> dict

Retrieves top-k chunks by FAISS similarity and generates question.

```python
result = pipeline.generate(
    learning_objective = "Explain SQL joins",
    difficulty         = "hard",
    question_type      = "application"
)
# {
#     "question": "...",
#     "metadata": {
#         "retrieval_method": "Standard RAG (single doc)",
#         "chunks_retrieved": 5,
#         "avg_retrieval_score": 0.82,
#         "sources": ["lecture.pdf"]
#     }
# }
```

**Chunks retrieved by difficulty:**
```
easy:   3 chunks
medium: 5 chunks
hard:   7 chunks
```

**Question type** (`recall`, `conceptual`, `application`, `case_study`) changes the LLM prompt only — not chunk counts. Same as GraphRAG; see [README.md](../README.md#difficulty-vs-question-type-important).

> **Note:** Single-doc RAG does NOT run LLM judge evaluation. Judge only runs in the GraphRAG pipeline.

---

## multi_indexer.py : Multi-Document Index Builder

Builds a combined FAISS index from multiple PDFs. Saved to disk and loaded at server startup.

### build_multi_index(docs_dir, index_dir)

Loads all PDFs from `docs_dir`, chunks them, embeds all chunks, builds combined FAISS index.

```bash
# Build multi-doc index:
python standard_rag/multi_indexer.py
```

**Output saved to:**
```
indexes/
├── multi_doc_index.faiss    # FAISS index
├── multi_doc_chunks.pkl     # chunk texts
└── multi_doc_meta.json      # document metadata
```

**Current index stats:**
```
34,468 chunks across 22 documents
Avg retrieval score: 0.80+
```

### What Documents Are Included

The multi-doc index is built from the 22 PDFs in `data/tested/` that were available during initial setup. It does NOT update automatically when new PDFs are added via admin API.

To rebuild after adding documents:
```bash
# Via API:
curl -X POST http://localhost:8000/multi/rebuild

# Or directly:
python standard_rag/multi_indexer.py
```

---

## multi_pipeline.py : Multi-Document RAG Pipeline

Uses the combined FAISS index to retrieve from all 22 documents simultaneously.

### Class: MultiDocPipeline

```python
from standard_rag.multi_pipeline import MultiDocPipeline

pipeline = MultiDocPipeline()
pipeline.load_models()   # shared with other pipelines
pipeline.load_index()    # loads multi_doc_index from disk
```

#### load_index(index_dir)

Loads the pre-built multi-doc FAISS index from `indexes/`.

```python
pipeline.load_index()
# Combined index loaded : 34468 vectors, 22 documents
```

#### generate(learning_objective, difficulty, question_type) -> dict

Retrieves top-k chunks from across all 22 documents and generates question.

```python
result = pipeline.generate(
    learning_objective = "Explain virtual memory management",
    difficulty         = "medium",
    question_type      = "conceptual"
)
# {
#     "question": "What is the primary purpose of virtual memory...",
#     "metadata": {
#         "retrieval_method": "Standard RAG (multi doc)",
#         "chunks_retrieved": 5,
#         "avg_retrieval_score": 0.804,
#         "sources": ["source_6.pdf"]
#     }
# }
```

#### status() -> dict

Returns index statistics.

```python
status = pipeline.status()
# {
#     "loaded": true,
#     "total_chunks": 34468,
#     "documents": 22
# }
```

---

## API Endpoints

### Single Document Flow

```bash
# Step 1 : Upload PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@my_lecture.pdf"

# Step 2 : Generate question from uploaded PDF
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "learning_objective": "Explain SQL transactions",
    "difficulty": "hard",
    "question_type": "application"
  }'
```

### Multi-Document Flow

```bash
# Generate from all 22 documents:
curl -X POST http://localhost:8000/multi/generate \
  -H "Content-Type: application/json" \
  -d '{
    "learning_objective": "Explain virtual memory",
    "difficulty": "medium",
    "question_type": "conceptual"
  }'

# Check index status:
curl http://localhost:8000/multi/status

# Rebuild index after adding documents:
curl -X POST http://localhost:8000/multi/rebuild
```

---

## Response Structure

Standard RAG responses are simpler than GraphRAG : no graph metadata, no judge scores:

```json
{
    "question": "What is the primary purpose of virtual memory...",
    "metadata": {
        "learning_objective":   "Explain virtual memory",
        "difficulty":           "medium",
        "question_type":        "conceptual",
        "retrieval_method":     "Standard RAG (multi doc)",
        "sources":              ["source_6.pdf"],
        "chunks_retrieved":     5,
        "avg_retrieval_score":  0.804
    }
}
```

**What is missing vs GraphRAG:**
```
 retrieved_chunks    (no chunk visibility)
 key_concepts        (no graph entities)
 concept_relationships (no graph edges)
 sample_answer       (not generated)
 judge_scores        (no evaluation)
 domain routing      (no domain awareness)
```

This is intentional : Standard RAG is the baseline. For production use, `/graphrag/generate` is recommended.

---

## Retrieval Comparison : Standard RAG vs GraphRAG

For the same query `"Explain SQL joins"` at `hard` difficulty:

**Standard RAG:**
```
Retrieved: 7 chunks most similar to "SQL joins"
All chunks: direct SQL join definitions and examples
Context: flat, focused, but limited concept coverage
```

**GraphRAG:**
```
Retrieved: 3 seed chunks (direct SQL joins)
         + 4 neighbour chunks (hash join, merge join,
           query optimisation, join ordering)
Context: richer : captures concept relationships
         produces more complex multi-concept questions
```

---

## Testing

```bash
# Test single-doc pipeline:
python standard_rag/pipeline.py

# Build and test multi-doc index:
python standard_rag/multi_indexer.py

# Test multi-doc pipeline:
python standard_rag/multi_pipeline.py
```

---

## Related

- [../graphrag/README.md](../graphrag/README.md) : GraphRAG pipeline (recommended for production)
- [../utils/README.md](../utils/README.md) : shared document_loader, embedder, generator
- [../README.md](../README.md) : Main project README
