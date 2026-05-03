# graphrag/ - GraphRAG Pipeline

The core of the system. Builds domain-specific knowledge graphs from IT textbooks and uses graph-aware retrieval to generate pedagogically grounded exam questions.

---

## Why GraphRAG Instead of Standard RAG?

Standard RAG finds chunks **similar to the query**. GraphRAG finds chunks similar to the query **and** chunks about concepts related to the query via the knowledge graph.

```
Standard RAG query: "SQL joins"
  -> Retrieves: chunks about SQL joins

GraphRAG query: "SQL joins"
  -> Retrieves: chunks about SQL joins (seed nodes)
  -> Also retrieves: hash join, merge join, query
    optimisation, join ordering, cost estimation
    (neighbour nodes - connected via graph edges)
```

This produces richer multi-concept questions that test deeper understanding.

---

## Files

```
graphrag/
├── __init__.py
├── graphrag_indexer.py         # Core graph builder (two-pass)
├── graphrag_retriever.py       # Graph traversal retrieval
├── graphrag_pipeline.py        # Orchestrator - routes and generates
├── graphrag_multi_indexer.py   # Combined graph (all domains, unused)
└── graphrag_domain_indexer.py  # Per-domain graphs + embedding router
```

---

## graphrag_domain_indexer.py - Domain Definitions and Routing

### DOMAIN_MAP

Defines which PDFs belong to which domain and the keywords used to build domain embedding profiles:

```python
DOMAIN_MAP = {
    "deep_learning": {
        "files": ["source_50.pdf", ..., "source_64.pdf"],
        "description": "Deep Learning - CNNs, pooling, backprop",
        "keywords": ["cnn", "convolutional", "neural network", ...]
    },
    "database": { ... },
    "software_engineering": { ... },
    "operating_systems": { ... },
    "computer_architecture": { ... }
}
```

> **Note:** File lists in DOMAIN_MAP are the fallback. The system reads from `data/document_registry.json` first (more up to date).

### build_domain_profiles(embedder)

Called once at server startup. Builds one embedding per domain from its description + keywords. Cached in `_domain_profiles`.

### route_by_embedding(query, embedder, return_all_scores=False)

Routes a learning objective to the correct domain using cosine similarity:

```python
# Returns winning domain name:
domain = route_by_embedding("Explain SQL joins", embedder)
# -> "database"

# Returns all scores:
scores = route_by_embedding("Explain SQL joins", embedder,
                             return_all_scores=True)
# -> {"database": 0.609, "deep_learning": 0.417, ...}
```

### build_domain_graph(domain_name, embedder=None)

Builds the full knowledge graph for one domain from scratch. Reads files from `document_registry.json` (fallback: DOMAIN_MAP).

```bash
# Build single domain:
python graphrag/graphrag_domain_indexer.py --domain database

# Build all domains:
python graphrag/graphrag_domain_indexer.py --domain all
```

---

## graphrag_indexer.py - Core Graph Builder

### Two-Pass Architecture

**Pass 1 - Entity extraction and filtering:**

```
For each chunk:
  -> Extract named entities + noun phrases (spaCy)
  -> Count document frequency (DF) per entity
  -> Filter noise:
      entity in >85% of chunks -> too generic -> remove
      entity appears < 2 times -> too rare -> remove
      entity length < 3 chars -> remove
      entity contains ©, numbers only -> remove
```

**Pass 2 - Graph construction:**

```
For each chunk:
  -> For every pair of entities in same chunk:
      -> Create edge between them
      -> Edge weight = number of co-occurrences
      -> Mark cross-source = True if entities
        from different PDFs
  -> Filter edges: weight < 2 -> remove
```

### build_graphrag_index(chunks, sources, index_dir, graph_name, embedder=None)

Main function. Takes chunks and builds FAISS + NetworkX graph.

```
Outputs saved to indexes/:
  {graph_name}.faiss      <- FAISS vector index of node embeddings
  {graph_name}_graph.pkl  <- NetworkX graph (nodes + edges)
  {graph_name}_meta.json  <- Metadata: node list, counts, sources
```

Pass `embedder=graph_pipeline.embedder` to reuse the server's GPU embedder - avoids reloading on CPU.

### incremental_update_graph(new_chunks, new_sources, index_dir, graph_name, embedder=None)

Updates existing graph with new document chunks **without full rebuild**:

```
1. Load existing graph from disk
2. Extract entities from NEW chunks only
3. Compare against existing nodes:
   - Already exists -> update frequency + sources
   - Truly new -> add as new node
4. Add new edges (new-to-new + new-to-existing)
5. Embed ONLY new nodes (not all 27,000+)
6. Add new embeddings to existing FAISS index
7. Save updated graph

Time: ~8 seconds per new PDF vs ~40 minutes for full rebuild
```

---

## graphrag_retriever.py - Graph Traversal Retrieval

### retrieve_graph_context(query, index, graph, nodes, embedder, ...)

Five-step retrieval process:

```
Step 1 - Seed node selection:
  Embed query -> FAISS similarity search
  -> top_k_nodes most similar concept nodes

Step 2 - Graph traversal:
  For each seed node:
    -> Find all neighbour nodes (BFS, depth=neighbour_depth)
    -> Sort neighbours by edge weight (co-occurrence strength)

Step 3 - Chunk collection:
  Seed chunks:     chunks stored in seed node data
  Neighbour chunks: chunks stored in neighbour node data

Step 4 - Seed/Neighbour ratio allocation:
  seed_limit      = max_chunks × seed_ratio
  neighbour_limit = max_chunks - seed_limit

  Difficulty ratios:
    easy:   80% seed, 20% neighbour (max 3 chunks)
    medium: 60% seed, 40% neighbour (max 5 chunks)
    hard:   40% seed, 60% neighbour (max 7 chunks)

  More neighbour chunks = more concept relationships
  = higher cognitive complexity in generated question

Step 5 - Deduplication:
  Remove duplicate chunks by content hash
  Assemble final context string
```

### Returns

```python
{
    "context":         "assembled text for LLM prompt",
    "seed_nodes":      [{"entity": "SQL join", "score": 0.8}],
    "neighbour_nodes": [{"entity": "hash join", "weight": 12}],
    "relationships":   [("SQL join", "hash join"), ...],
    "sources":         ["source_7.pdf", "source_8.pdf"],
    "chunk_pool":      [{"text": "...", "type": "seed", ...}]
}
```

---

## graphrag_pipeline.py - Orchestrator

### Class: GraphRAGPipeline

Main pipeline class. Loaded once at server startup.

#### Startup sequence (called by main_gr.py):

```python
pipeline = GraphRAGPipeline()
pipeline.load_models()       # loads BAAI embedder + Qwen 7B
pipeline.load_judge_model()  # reuses Qwen 7B as judge
```

#### generate_with_routing(learning_objective, difficulty, question_type, domain=None)

Main generation method. Handles all 5 routing cases:

```
Case 1: domain specified + valid
  -> use that domain directly

Case 2: domain specified + invalid
  -> warn + fall back to embedding router

Case 3: no domain + hard difficulty
  -> multi-domain retrieval

Case 4: no domain + easy/medium
  -> embedding router -> single best domain

Case 5: domain specified + easy/medium
  -> use that domain directly
```

#### generate_with_multi_domain(learning_objective, difficulty, question_type)

Hard difficulty multi-domain retrieval:

```
1. Get embedding scores for ALL domains
2. Filter domains above domain_score_threshold (0.5)
3. Retrieve chunks_per_domain chunks from each qualifying domain
4. Combine contexts: [DEEP_LEARNING DOMAIN]\n...\n---\n[DATABASE DOMAIN]\n...
5. Single LLM call with combined context
6. LLM judge evaluation
```

#### generate(learning_objective, difficulty, question_type, ...)

Core single-domain generation:

```
1. Load difficulty config from config.yml
   (max_chunks, seed_ratio, top_k_nodes)
2. retrieve_graph_context() - graph traversal
3. generate_question() - first Qwen call
4. generate_answer() - second Qwen call
5. LLM judge evaluation (if judge loaded)
6. Return question + full metadata
```

---

## Building Domain Graphs - Step by Step

### Prerequisites

1. All PDFs must be in `data/tested/`
2. `document_registry.json` must exist (run `scripts/create_registry.py` first)
3. Virtual environment activated

### Build Command

```bash
# Single domain (~5-10 minutes):
python graphrag/graphrag_domain_indexer.py --domain deep_learning

# All domains (~40-60 minutes total):
python graphrag/graphrag_domain_indexer.py --domain all
```

### What Gets Built

```
indexes/
├── graphrag_deep_learning.faiss        # FAISS node embeddings
├── graphrag_deep_learning_graph.pkl    # NetworkX graph
├── graphrag_deep_learning_meta.json    # Metadata
├── graphrag_database.*
├── graphrag_software_engineering.*
├── graphrag_operating_systems.*
└── graphrag_computer_architecture.*
```

### Verify Build

```bash
python -c "
from graphrag.graphrag_domain_indexer import load_domain_graph
index, graph, nodes = load_domain_graph('database')
print(f'Nodes: {graph.number_of_nodes()}')
print(f'Edges: {graph.number_of_edges()}')
print(f'FAISS vectors: {index.ntotal}')
"
```

Expected output:
```
Nodes: 27284
Edges: 142193
FAISS vectors: 27284
```

---

## Domain Graph Statistics

| Domain | Files | Nodes | Edges | Cross-Source |
|--------|-------|-------|-------|-------------|
| deep_learning | 15 | 5,330 | 17,256 | 3,255 |
| database | 9 | 27,284 | 142,193 | 41,403 |
| software_engineering | 14 | 7,782 | 24,428 | 6,861 |
| operating_systems | 11 | 7,671 | 22,525 | 3,161 |
| computer_architecture | 14 | 15,824 | 64,777 | 7,465 |

---

## Alpha and Beta Parameters

Alpha and beta control knowledge graph density.

**Alpha - Node creation (in `config.yml`):**
```yaml
graph:
  alpha:
    max_doc_freq: 0.85      # entity in >85% chunks = noise -> removed
    min_entity_freq: 2      # entity must appear ≥2 times
```

**Beta - Edge creation:**
```yaml
graph:
  beta:
    min_edge_weight: 2      # co-occurrence must be ≥2 to create edge
```

> **Samar's insight:** If `min_edge_weight = 0` -> no edges created -> GraphRAG degrades to standard RAG because graph traversal finds no neighbours.

---

## Testing

```bash
# Test graph pipeline directly:
python graphrag/graphrag_pipeline.py

# Test domain indexer + routing:
python graphrag/graphrag_domain_indexer.py --domain deep_learning

# Test retriever:
python graphrag/graphrag_retriever.py
```

---

## Related

- [../utils/README.md](../utils/README.md) - document_loader, embedder, generator, judge
- [../evaluation/README.md](../evaluation/README.md) - LLM-as-Judge system
- [../README.md](../README.md) - Main project README
