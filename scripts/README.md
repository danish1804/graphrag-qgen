# scripts/ : Utility and Maintenance Scripts

Standalone scripts for setup, validation, testing, and comparison. These are run directly : nothing imports FROM this folder.

---

## Files

```
scripts/
├── check_doc_topics.py         # Check what topic each PDF covers
├── check_setup.py              # Verify environment is correctly installed
├── check_top_entities.py       # Inspect top entities in a domain graph
├── compare_domain_graphrag.py  # Compare domain-routed GraphRAG outputs
├── compare_rag_vs_graphrag.py  # Compare Standard RAG vs GraphRAG
├── create_registry.py          # Initialise document_registry.json
├── get_sample_chunks.py        # Preview chunks from a PDF
├── rename_docs.py              # Rename PDFs to source_N format
├── test_chunk_filter.py        # Test quality chunk filter
├── test_document_batch.py      # Batch test document loading
├── test_new_features.py        # case_study + dedup smoke (offline + --live API)
└── verify_refactor_changes.py  # Post-refactor API verification script
```

---

## check_setup.py : Environment Verification

Run this first after installing dependencies to confirm everything is correctly set up.

```bash
python scripts/check_setup.py
```

**Checks:**
- Python version (3.10.x required)
- CUDA availability and GPU name
- PyTorch version and CUDA build
- spaCy model (`en_core_web_sm`) installed
- FAISS importable
- HuggingFace transformers available
- BitsAndBytes (4-bit quantisation) available
- Project folder structure correct
- `data/tested/` exists and contains PDFs
- `indexes/` exists

**Expected output:**
```
 Python 3.10.11
 CUDA available : NVIDIA GeForce RTX 4060
 PyTorch 2.6.0+cu124
 spaCy en_core_web_sm loaded
 FAISS imported
 Transformers available
 BitsAndBytes available
 data/tested/ : 63 PDFs found (subject to change)
 indexes/ : 15 index files found (subject to change)
Setup verified. Ready to run.
```

---

## rename_docs.py : PDF Renaming

Renames all PDFs in `data/raw/` to the standard `source_N.pdf` format and moves them to `data/tested/`.

```bash
# Preview what would be renamed (no changes):
python scripts/rename_docs.py --dry-run

# Apply renames:
python scripts/rename_docs.py
```

**What it does:**
1. Reads `last_source_number` from `data/document_registry.json`
2. Finds all un-renamed PDFs/PPTX in `data/raw/`
3. Renames each to `source_{N}.pdf` continuing from last number
4. Moves renamed file to `data/tested/`
5. Updates `last_source_number` in registry
6. Logs all renames to `data/rename_log.txt`

**Example:**
```
[Registry] Last source number: 63
New files will start from: source_64

  [ 64] DeepLearning_Lecture9.pdf
        -> source_64.pdf
        -> moved to tested/
        -> registry counter: 64

Done - 1 files renamed successfully.
```

**Notes:**
- Skips files already in `source_N` format
- Supports `.pdf` and `.pptx`
- Does NOT register domain : use admin API after renaming

---

## create_registry.py : Initialise Document Registry

Run **once** when setting up the project for the first time. Creates `data/document_registry.json` from the existing `DOMAIN_MAP` in `graphrag_domain_indexer.py`.

```bash
python scripts/create_registry.py
```

**Output:**
```
Registry created: data/document_registry.json
Total documents:  63
Last number:      63
  deep_learning: 13 files
  database: 9 files
  software_engineering: 14 files
  operating_systems: 11 files
  computer_architecture: 14 files
```

> **When to run:** Only needed on first setup OR if `document_registry.json` is lost or corrupted. Running again will overwrite the existing registry : any admin-uploaded documents not in DOMAIN_MAP will be lost.

---

## check_doc_topics.py : PDF Topic Inspector

Checks what topic each PDF actually covers by extracting and displaying a text preview. Useful for verifying PDFs are assigned to the correct domain.

```bash
python scripts/check_doc_topics.py
```

Or check a specific PDF:
```bash
python scripts/check_doc_topics.py --file source_50.pdf
```

**Output:**
```
source_50.pdf:
  First 300 chars: "Deep Learning COSC2779 Lecture 4
  Convolutional Neural Networks Parameter sharing..."
  Suggested domain: deep_learning
```

---

## check_top_entities.py : Domain Graph Entity Inspector

Shows the top N entities in a domain knowledge graph by frequency. Useful for:
- Verifying graph built correctly
- Identifying noise entities that slipped through filters
- Understanding what concepts dominate each domain

```bash
python scripts/check_top_entities.py --domain database --top 20
```

**Output:**
```
Top 20 entities in database domain:
  1. transaction          (freq=1842, sources=8)
  2. relation             (freq=1205, sources=7)
  3. sql                  (freq=987,  sources=9)
  4. index                (freq=876,  sources=6)
  5. deadlock             (freq=743,  sources=5)
  ...
```

**Use case:** If you see noise entities like `figure`, `table`, `chapter` in the top entities, the alpha threshold (`max_doc_freq`) may need adjusting in `config.yml`.

---

## check_top_entities.py : Domain Graph Entity Inspector

Shows the top N entities in a domain knowledge graph by frequency. Useful for verifying graph quality and identifying noise.

```bash
python scripts/check_top_entities.py --domain database --top 20
```

---

## compare_rag_vs_graphrag.py : Standard RAG vs GraphRAG Comparison

Runs the same learning objective through both Standard RAG and GraphRAG pipelines and prints both questions side by side for quality comparison.

```bash
python scripts/compare_rag_vs_graphrag.py
```

**Output:**
```
Learning objective: Explain parameter sharing in CNNs
Difficulty: medium | Type: conceptual

Standard RAG question:
  What is parameter sharing in CNNs?

GraphRAG question:
  Explain how parameter sharing in CNNs enables
  efficient feature extraction and how it relates
  to sparse connectivity and translation equivariance.

GraphRAG retrieved concepts:
  parameter sharing -> feature extraction
  parameter sharing -> sparse connectivity
  parameter sharing -> same operation
```

This directly demonstrates the GraphRAG advantage : richer multi-concept questions from graph traversal.

---

## compare_domain_graphrag.py : Domain Routing Comparison

Tests the embedding-based domain router on a set of queries and shows routing scores for all domains.

```bash
python scripts/compare_domain_graphrag.py
```

**Output:**
```
Query: "Explain SQL joins"
  database:             0.609  <- winner
  software_engineering: 0.446
  operating_systems:    0.404
  computer_architecture: 0.416
  deep_learning:        0.417

Query: "Explain CNN parameter sharing"
  deep_learning:        0.726  <- winner
  computer_architecture: 0.436
  database:             0.395
  ...
```

Use this to verify routing accuracy after adding new domains or updating domain keywords.

---

## get_sample_chunks.py : Chunk Preview

Loads a PDF and displays sample chunks to verify chunking quality.

```bash
python scripts/get_sample_chunks.py --file data/tested/source_7.pdf
```

**Output:**
```
Processing: source_7.pdf
  Extracted 89,432 characters
  Created 197 chunks (size=500, overlap=50)

Chunk 1:
  "A transaction is a unit of program execution
   that accesses and possibly updates various data
   items. To ensure data integrity..."

Chunk 2:
  "..."
```

Useful for checking if chunks are clean or contain OCR artifacts before building the graph.

---

## test_chunk_filter.py : Quality Filter Test

Tests the `is_quality_chunk()` filter that removes boilerplate content (title pages, table of contents, copyright notices, formula dumps).

```bash
python scripts/test_chunk_filter.py
```

**Output:**
```
Testing quality chunk filter...

PASS (should reject):
   "Copyright © 2023 Pearson Education..." -> rejected
   "Table of Contents\n1. Introduction..." -> rejected 
   "ISBN 978-0-13-..." -> rejected 

PASS (should keep):
   "A transaction is a unit of program execution..." -> kept 
   "Parameter sharing reduces learnable weights..." -> kept 

Filter working correctly.
```

---

## test_document_batch.py : Batch Document Loading Test

Tests loading and chunking of multiple PDFs in sequence. Used to verify all PDFs in a domain are readable before building the graph.

```bash
python scripts/test_document_batch.py --domain database
```

**Output:**
```
Testing batch load for domain: database
  source_7.pdf    -> 197 chunks 
  source_8.pdf    -> 312 chunks 
  source_13.pdf   -> 89 chunks  
  source_14.pdf   -> 421 chunks 
  source_17.pdf   -> 156 chunks 
  source_18.pdf   -> 203 chunks 
  source_31.pdf   -> 67 chunks  
  source_36.pdf   -> 178 chunks 
  source_49.pdf   -> 298 chunks 

Total: 9 PDFs, 1921 chunks loaded successfully.
```

Run this before building a domain graph to catch any corrupt or unreadable PDFs early.

---

## Recommended Setup Order

When setting up the project from scratch:

```bash
# 1. Verify environment
python scripts/check_setup.py

# 2. Place PDFs in data/raw/ then rename them
python scripts/rename_docs.py --dry-run   # preview first
python scripts/rename_docs.py             # apply

# 3. Verify PDFs load correctly
python scripts/test_document_batch.py --domain deep_learning

# 4. Create document registry
python scripts/create_registry.py

# 5. Build domain graphs (takes 40-60 minutes)
python graphrag/graphrag_domain_indexer.py --domain all

# 6. Start the server
python main_gr.py --mode full
```

---

## test_new_features.py : Feature smoke tests

Runs unit tests for dedup hashing, static checks for `case_study` in config/prompts, and optional live API tests.

```bash
# Offline only
python scripts/test_new_features.py

# With API on port 8000
python scripts/test_new_features.py --live --generate
```

---

## Related

- [../data/README.md](../data/README.md) : PDF structure and document flow
- [../graphrag/README.md](../graphrag/README.md) : graph building details
- [../README.md](../README.md) : Main project README
