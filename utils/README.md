# utils/ - Shared Utilities

Shared utility modules used across all pipelines. No pipeline-specific logic lives here - these are pure building blocks.

---

## Files

```
utils/
├── __init__.py
├── config.py                 # YAML config loader
├── document_loader.py        # PDF/PPTX -> text chunks
├── embedder.py               # BAAI embeddings + FAISS index
├── generator.py              # Qwen 2.5 7B inference
├── judge.py                  # LLM-as-Judge model loader
├── registry.py               # Document registry (PDF tracking)
├── prompts.py                # Question/answer prompt templates
├── question_type_prompts.py  # Per-type instructions (incl. case_study)
├── question_types.py         # Valid types from config
├── retrieval_config.py       # top_k_nodes / max_chunks per difficulty
├── multi_domain_retrieval.py # Multi-domain chunk budget (aligns with max_chunks)
├── content_hash.py           # Dedup byte/text hashing
├── text_normalization.py     # Text normalisation for similarity
├── text_similarity.py        # Near-duplicate detection
├── registry_content.py       # Registry fingerprints for dedup
├── atomic_io.py              # Atomic JSON write (registry, runtime state)
└── runtime_state.py          # Server runtime flags (e.g. rebuild in progress)
```

---

## Difficulty vs question type

| Module | Role |
|--------|------|
| `retrieval_config.py` | Maps **difficulty** → `top_k_nodes`, `neighbour_depth`, `max_chunks` |
| `multi_domain_retrieval.py` | Ensures multi-domain hard uses same per-domain budget as single-domain hard |
| `question_type_prompts.py` | Maps **question_type** → extra LLM instructions (case_study = SCENARIO + QUESTION) |
| `registry_content.py` | Store content hashes / duplicate flags in `document_registry.json` |
| `atomic_io.py` | Safe atomic writes for JSON state files |
| `runtime_state.py` | Lightweight server state (avoid concurrent rebuild conflicts) |
| `generator.py` | Uses `question_type` for prompts; per-type `max_new_tokens` from config |

**Question type does not affect retrieval.** Only `cfg["difficulty"][difficulty]` drives chunk counts and routing (see `graphrag/graphrag_pipeline.py`).

---

## config.py - Configuration Loader

Loads `config.yml` from project root once at startup. Cached after first load - subsequent calls return the same object.

### Usage

```python
from utils.config import cfg

# Difficulty parameters
max_chunks    = cfg["difficulty"]["hard"]["max_chunks"]    # 7
seed_ratio    = cfg["difficulty"]["medium"]["seed_ratio"]  # 0.6

# Model names
llm_name      = cfg["model"]["llm_name"]        # Qwen/Qwen2.5-7B-Instruct
embed_model   = cfg["model"]["embedding_model"] # BAAI/bge-large-en-v1.5

# Graph parameters
max_doc_freq  = cfg["graph"]["alpha"]["max_doc_freq"]   # 0.85
min_edge_weight = cfg["graph"]["beta"]["min_edge_weight"] # 2

# Judge parameters
threshold     = cfg["judge"]["score_threshold"]  # 3.5
judge_name    = cfg["judge"]["judge_name"]       # Qwen/Qwen2.5-3B-Instruct
```

### config.yml Location

Always resolved relative to project root via `GRAPHRAG_BASE_DIR` env var or `__file__` dirname. No hardcoded paths.

### Test

```bash
python -c "from utils.config import cfg; print(cfg['difficulty'])"
```

---

## document_loader.py - PDF/PPTX to Chunks

Loads PDF and PPTX files and splits them into overlapping text chunks.

### Functions

#### load_and_chunk(file_path, chunk_size=None, chunk_overlap=None)

Main function. Reads chunk size from `config.yml` if not explicitly passed.

```python
from utils.document_loader import load_and_chunk

result = load_and_chunk("data/tested/source_7.pdf")

# Returns:
{
    "file_name":    "source_7.pdf",
    "file_type":    ".pdf",
    "total_chars":  142853,
    "total_chunks": 312,
    "chunks":       ["First 500 chars...", "Next chunk..."]
}
```

**Chunk parameters (from config.yml):**
```yaml
document:
  chunk_size: 500       # characters per chunk
  chunk_overlap: 50     # overlap between consecutive chunks
  min_chunk_chars: 200  # minimum chunk length to keep
```

#### load_pdf(file_path) -> str

Extracts raw text from PDF using `pypdf`.

#### load_pptx(file_path) -> str

Extracts text from each slide of a PPTX file. Prefixes each slide with `[Slide N]`.

#### chunk_text(text, chunk_size, chunk_overlap) -> List[str]

Splits text using `RecursiveCharacterTextSplitter` with separators: `["\n\n", "\n", ". ", " ", ""]`.

### Supported Formats

| Format | Extension | Extractor |
|--------|-----------|-----------|
| PDF | .pdf | pypdf PdfReader |
| PowerPoint | .pptx | python-pptx Presentation |

### Test

```bash
python utils/document_loader.py path/to/file.pdf
# or
python utils/document_loader.py  # runs on sample text
```

---

## embedder.py - BAAI Embeddings + FAISS

Loads the BAAI/bge-large-en-v1.5 embedding model and provides FAISS index operations.

### Class: Embedder

```python
from utils.embedder import Embedder

embedder = Embedder()
# Loads: BAAI/bge-large-en-v1.5 (1024-dim)
```

#### embedder.embed(texts: List[str]) -> np.ndarray

Encodes a list of strings to 1024-dimensional normalised embeddings.

```python
embeddings = embedder.embed(["SQL joins", "database transactions"])
# Shape: (2, 1024), normalised to unit length
```

### FAISS Functions

#### build_index(chunks, embedder) -> faiss.Index

Builds a FAISS IndexFlatIP (inner product) index from text chunks.

```python
from utils.embedder import build_index
index = build_index(chunks, embedder)
```

#### search(query, index, nodes, embedder, top_k) -> List[dict]

Searches FAISS index for top-k most similar chunks.

```python
results = search("SQL joins", index, nodes, embedder, top_k=3)
# Returns: [{"chunk": "...", "score": 0.85, "source": "..."}]
```

#### save_index / load_index

```python
from utils.embedder import save_index, load_index
save_index(index, nodes, "indexes/my_index")
index, nodes = load_index("indexes/my_index")
```

### Model Details

| Property | Value |
|----------|-------|
| Model | BAAI/bge-large-en-v1.5 |
| Dimensions | 1024 |
| Normalisation | L2 normalised |
| Device | CUDA if available, else CPU |
| Download size | ~1.3GB |

### Test

```bash
python utils/embedder.py
```

---

## generator.py - Qwen 2.5 7B Inference

Loads Qwen 2.5 7B Instruct (4-bit quantised) and provides question and answer generation.

### Functions

#### load_model() -> (model, tokenizer)

Loads Qwen 2.5 7B in 4-bit NF4 quantisation using BitsAndBytes.

```python
from utils.generator import load_model
model, tokenizer = load_model()
# Downloads ~4GB on first run
# Subsequent runs load from cache (~15 seconds)
```

**Quantisation config:**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

#### generate_question(context, learning_objective, model, tokenizer, difficulty, question_type) -> str

Generates one exam question. **Retrieval is already done** — this only formats the LLM call.

- **difficulty** — passed to prompt; answer depth for `generate_answer()` uses this.
- **question_type** — selects prompt template via `question_type_prompts.py`:
  - `recall`, `conceptual`, `application` — standard single question
  - `case_study` — requires **SCENARIO:** block + **QUESTION:** line; uses `max_new_tokens: 280` from config

```python
from utils.generator import generate_question

question = generate_question(
    context            = retrieval["context"],
    learning_objective = "Explain deadlocks",
    model              = model,
    tokenizer          = tokenizer,
    difficulty         = "hard",
    question_type      = "case_study"   # retrieval unchanged vs "application"
)
```

Generates one exam question from retrieved context.

```python
from utils.generator import generate_question

question = generate_question(
    context            = "Parameter sharing means...",
    learning_objective = "Explain CNN parameter sharing",
    model              = model,
    tokenizer          = tokenizer,
    difficulty         = "medium",
    question_type      = "conceptual"
)
```

**System prompt rules enforced:**
- Generate exactly ONE question
- Open-ended only - no MCQ, no A) B) C) D)
- Do not reference "the context" or "the passage"
- Question must be standalone and self-contained
- Do not invent values not present in context

#### generate_answer(context, question, model, tokenizer, difficulty) -> str

Generates a sample answer using the same context as question generation.

```python
from utils.generator import generate_answer

answer = generate_answer(
    context    = "Parameter sharing means...",
    question   = "Explain how parameter sharing works...",
    model      = model,
    tokenizer  = tokenizer,
    difficulty = "hard"
)
```

**Answer depth by difficulty:**
```
easy:   2-3 sentences - brief and direct
medium: 3-5 sentences - covers key points
hard:   5-8 sentences - concepts, relationships, examples
```

### Model Details

| Property | Value |
|----------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Quantisation | 4-bit NF4 (BitsAndBytes) |
| VRAM usage | ~6GB |
| Max new tokens | 512 (question), 512 (answer) |
| Temperature | 0.7 |
| Download size | ~4GB |

### Test

```bash
python utils/generator.py
```

---

## judge.py - LLM-as-Judge Model Loader

Loads the judge LLM with automatic GPU/CPU device selection based on available VRAM.

### Functions

#### load_judge() -> (model, tokenizer, device)

Loads the judge model. Checks free VRAM first:

```python
from utils.judge import load_judge

model, tokenizer, device = load_judge()
# [Judge] GPU free: 2.1 GB / 8.0 GB total
# [Judge] Free VRAM sufficient - using GPU.
# OR
# [Judge] Free VRAM < 1.5 GB - running on CPU.
```

**Device selection:**
```python
MIN_GPU_FREE_GB = cfg["judge"]["min_gpu_free_gb"]  # 1.5

if free_vram >= MIN_GPU_FREE_GB:
    device = "cuda"   # 4-bit quantised on GPU
else:
    device = "cpu"    # fp32 on CPU (slower)
```

> **For RTX 4060 8GB:** After Qwen 7B loads (~6GB), only ~2GB free. Judge loads on GPU if `min_gpu_free_gb ≤ 2.0`. To reuse Qwen 7B as judge instead (recommended), see `graphrag/graphrag_pipeline.py -> load_judge_model()`.

#### run_judge(model, tokenizer, system_prompt, user_prompt) -> str

Runs one inference call on the judge model.

```python
from utils.judge import run_judge

output = run_judge(
    model         = model,
    tokenizer     = tokenizer,
    system_prompt = "You are an expert evaluator...",
    user_prompt   = "Evaluate this question: ..."
)
# Returns raw text with REASONING + SCORES sections
```

#### unload_model(model, tokenizer)

Frees GPU/CPU memory. Called when judge is used in standalone mode.

```python
from utils.judge import unload_model
unload_model(model, tokenizer)
# Calls del model, gc.collect(), torch.cuda.empty_cache()
```

### Judge Config (config.yml)

```yaml
judge:
  judge_name:      "Qwen/Qwen2.5-3B-Instruct"
  max_new_tokens:  512
  temperature:     0.1      # low for consistent scoring
  load_in_4bit:    true
  min_gpu_free_gb: 1.5
  score_threshold: 3.5      # below -> regenerate question
  max_regenerations: 2
```

### Test

```bash
python utils/judge.py
# Loads judge, runs smoke test, unloads
```

---

## registry.py - Document Registry

Tracks all PDFs in the system - their domain assignments and indexing status. Single source of truth replacing hardcoded file lists.

### Registry File Location

```
data/document_registry.json
```

### Registry Structure

```json
{
  "last_source_number": 64,
  "domains": {
    "deep_learning": {
      "description": "Deep Learning - CNNs, pooling, backprop",
      "files": {
        "source_50.pdf": {"indexed": true, "added": "2026-03-01"},
        "source_64.pdf": {"indexed": true, "added": "2026-05-02"}
      }
    },
    "database": { ... }
  }
}
```

### Functions

#### load_registry() -> dict

Loads registry JSON from disk. Raises `FileNotFoundError` if not found.

```python
from utils.registry import load_registry
registry = load_registry()
```

#### get_domain_files(domain, indexed_only=True) -> List[str]

Returns filenames for a domain. Used by `build_domain_graph()` instead of hardcoded DOMAIN_MAP lists.

```python
from utils.registry import get_domain_files
files = get_domain_files("database")
# ["source_7.pdf", "source_8.pdf", ..., "source_64.pdf"]
```

#### register_document(filename, domain, indexed=False)

Registers a new PDF under a domain. Called by admin upload endpoint.

```python
from utils.registry import register_document
register_document("source_65.pdf", domain="database", indexed=False)
```

#### mark_as_indexed(filename, domain)

Marks a document as indexed after rebuild completes.

```python
from utils.registry import mark_as_indexed
mark_as_indexed("source_65.pdf", domain="database")
```

#### get_unindexed_files(domain=None) -> List[dict]

Returns files registered but not yet indexed.

```python
from utils.registry import get_unindexed_files
pending = get_unindexed_files(domain="database")
# [{"filename": "source_65.pdf", "domain": "database", "added": "2026-05-03"}]
```

#### get_registry_summary() -> dict

Returns counts per domain. Used by `/admin/new-documents` endpoint.

```python
from utils.registry import get_registry_summary
summary = get_registry_summary()
# {
#   "last_source_number": 64,
#   "domains": {
#     "deep_learning": {"total_files": 15, "indexed_files": 15, "pending_files": 0}
#   },
#   "total_files": 63, "total_indexed": 63
# }
```

#### get_next_source_number() -> int

Returns next available source number and increments registry counter. Used by `rename_docs.py`.

```python
from utils.registry import get_next_source_number
next_num = get_next_source_number()  # returns 65, registry updated to 65
```

### Initialise Registry

Run once after setting up the project:

```bash
python scripts/create_registry.py
```

This reads `DOMAIN_MAP` and creates `data/document_registry.json` with all existing PDFs marked as indexed.

### Test

```bash
python utils/registry.py
# Runs 9 tests: load, summary, get files,
# unindexed check, register, verify, mark, verify, cleanup
```

---

## Import Patterns

All utils files handle both direct execution and import gracefully:

```python
# Works when imported by main_gr.py (project root on path):
from utils.config import cfg

# Works when run directly (utils/ on path):
try:
    from utils.config import cfg
except ImportError:
    from config import cfg
```

---

## Related

- [../graphrag/README.md](../graphrag/README.md) - uses document_loader, embedder, generator, judge
- [../standard_rag/README.md](../standard_rag/README.md) - uses document_loader, embedder, generator
- [../evaluation/README.md](../evaluation/README.md) - uses judge, config
- [../README.md](../README.md) - Main project README
