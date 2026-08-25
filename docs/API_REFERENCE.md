# API Reference — AI Exam Question Generator

**Project:** P000233DS (EdGenAI) · **Service:** `main_gr.py` · **Version:** 2.0.0
**Base URL:** `http://<host>:8000` · **Interactive docs:** `/docs` (Swagger), `/redoc`
**Prepared for:** integration planning · **Maintainer:** Mohammed Danish Alam

---

## 1. Integration summary (read this first)

The service exposes **14 active endpoints** across three functional groups. For
integration you only need the **3 core endpoints**; the admin group is for content
operations, and the standard-RAG group is a research baseline that should **not** be
integrated.

| Priority | Group | Endpoints | Integrate? |
|---|---|---|---|
| **P0 — Core** | Product surface | `POST /graphrag/generate`, `GET /health`, `GET /graphrag/status` | **Yes** |
| P1 — Content ops | Admin / corpus management | 6 endpoints (`/admin/*`, `/graphrag/upload`, `/graphrag/rebuild/*`) | Behind admin auth |
| P2 — Baseline | Standard RAG (single + multi doc) | 5 endpoints (`/upload`, `/generate`, `/multi/*`) | **No — research baseline only** |

**The one endpoint that matters:** `POST /graphrag/generate`. Everything else is
health, observability, or corpus administration.

Server runs in two modes (`config.yml → server.mode`):

- `full` — GraphRAG loaded, all endpoints live *(use this)*
- `standard` — GraphRAG skipped, `/graphrag/*` returns **503**

---

## 2. Core endpoints (P0)

### 2.1 `POST /graphrag/generate` — generate an exam question

The primary product endpoint. Takes a learning outcome, returns a generated exam
question plus a reference answer and full provenance metadata.

**Request**

```json
{
  "learning_objective": "Explain how normalisation reduces redundancy in relational databases",
  "difficulty": "medium",
  "question_type": "conceptual",
  "top_k_nodes": 3,
  "neighbour_depth": 1,
  "domain": null
}
```

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `learning_objective` | string | **yes** | — | min length 3 — **this is the query** |
| `difficulty` | enum | no | `medium` | `easy` \| `medium` \| `hard` |
| `question_type` | enum | no | `conceptual` | `recall` \| `conceptual` \| `application` \| `case_study` |
| `top_k_nodes` | int | no | 3 | 1–20 — ⚠️ **currently ignored**, see below |
| `neighbour_depth` | int | no | 1 | 1–3 — ⚠️ **currently ignored**, see below |
| `domain` | string \| null | no | `null` | `null` = auto-route (recommended) |

Valid `domain` values: `database`, `deep_learning`, `software_engineering`,
`operating_systems`, `computer_architecture`.

**There is no separate `query` field — `learning_objective` *is* the query.** The
string is used twice in the pipeline:

1. **As the retrieval query.** It is embedded with `bge-large-en-v1.5` and used for
   the FAISS vector search that selects seed concept nodes, and again to re-rank the
   retrieved chunks by cosine similarity.
2. **As the generation instruction.** It is inserted into the prompt so the produced
   question targets that specific outcome.

> ⚠️ **`top_k_nodes` and `neighbour_depth` are accepted and validated, but do not
> take effect.** `GraphRAGPipeline.generate()` overwrites both from
> `config.yml → difficulty.<level>` immediately after entry, and the multi-domain path
> never receives them. **Retrieval breadth is controlled solely by `difficulty`:**
>
> | difficulty | max_chunks | top_k_nodes | neighbour_depth | seed_ratio | multi-domain |
> |---|---|---|---|---|---|
> | `easy` | 3 | 2 | 1 | 0.80 | no |
> | `medium` | 5 | 3 | 1 | 0.60 | no |
> | `hard` | 7 | 5 | 2 | 0.40 | yes |
>
> Treat `difficulty` as the retrieval-tuning lever. The two fields should either be
> wired through or removed from the request model — tracked as gap #8 in §5.

**Domain routing behaviour**

| Input | Behaviour |
|---|---|
| Valid `domain` given | Uses that domain directly |
| Invalid `domain` given | Warns, falls back to auto-routing (does **not** error) |
| `null` + `hard` | Multi-domain retrieval across all domains above score threshold |
| `null` + `easy`/`medium` | Embedding-based auto-routing to the single best domain |

**Response** — `{ question, metadata }`, `extra = allow` (metadata keys may grow).

```json
{
  "question": "Explain why it is critical to achieve 2NF before proceeding to 3NF...",
  "metadata": {
    "sample_answer": "...",
    "domain_routed_to": "database",
    "domains_used": ["database"],
    "routing_confidence": 0.34,
    "retrieved_chunks": ["..."],
    "concept_relationships": [["normalisation", "redundancy", 12]],
    "difficulty": "medium",
    "question_type": "conceptual",
    "judge_scores": {
      "question_overall": 4.6,
      "answer_overall": 4.8,
      "retrieval_overall": 3.3,
      "attempts": 1,
      "eval_id": "eval_20260819_..."
    }
  }
}
```

**Integration notes**

- **Latency is significant.** The request runs retrieval → generation → LLM-as-judge,
  and regenerates up to **3 attempts** if the judge score is below threshold
  (`judge.score_threshold` = 3.5). Worst case is 3 full generate+judge cycles in one
  synchronous HTTP call. Budget accordingly, or wrap in an async job for production.
- `judge_scores` may contain `null` values if judge output fails to parse — handle it.
- Errors return `500` with `{"detail": "..."}`; validation errors return `400`.

---

### 2.2 `GET /health` — liveness and readiness

No parameters. Use for load-balancer health checks and to confirm which pipelines
loaded.

```json
{
  "status": "ok",
  "models_loaded": true,
  "standard_rag": { "...": "..." },
  "graphrag": { "...": "..." }
}
```

---

### 2.3 `GET /graphrag/status` — index observability

No parameters. Returns per-domain index state including node and edge counts. Use to
verify the graphs are loaded and to expose corpus size in an admin UI.

---

## 3. Admin / content-ops endpoints (P1)

These mutate the corpus and the knowledge graph. **They must sit behind admin
authentication before any deployment** (see §5).

| Method | Path | Purpose | Body / Params |
|---|---|---|---|
| `POST` | `/admin/upload-document` | Upload a PDF/PPTX into a domain | `domain` (query), `file` (multipart) |
| `POST` | `/admin/rebuild-domain` | Rebuild one domain's graph; supports incremental update + dedup | `{ "domain": "database" }` |
| `POST` | `/admin/add-domain` | Register a new subject domain at runtime | `{ "domain_name", "description", "keywords": [] }` |
| `GET` | `/admin/new-documents` | List files in `tested/` not yet assigned to a domain | — |
| `POST` | `/graphrag/upload` | Upload a PDF/PPTX; domain auto-detected from filename + content | `file` (multipart) |
| `POST` | `/graphrag/rebuild/{domain_name}` | Full rebuild of one domain's graph | path param |

**`POST /admin/rebuild-domain` response** (`AdminRebuildResponse`):

```json
{
  "status": "ok", "domain": "database", "update_type": "incremental",
  "message": "...", "new_nodes_added": 412, "new_edges_added": 1843,
  "nodes_updated": 96, "graph_nodes": 20274, "graph_edges": 101340,
  "elapsed_seconds": 84.2, "new_files": ["source_81.pdf"],
  "skipped_duplicates": [{ "file": "source_82.pdf", "duplicate_of": "source_7" }]
}
```

**Integration note — rebuilds are long-running.** A full-domain rebuild takes minutes;
all five domains take roughly an hour. These are synchronous HTTP calls and **will
exceed typical gateway timeouts**. For production, run them as background jobs with a
status-polling endpoint.

---

## 4. Standard RAG baseline (P2 — do not integrate)

Retained for the GraphRAG-vs-RAG comparison in the research write-up. Same generation
contract, no graph.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Index a single document |
| `POST` | `/generate` | Generate from that single document (requires prior `/upload`) |
| `GET` | `/multi/status` | Combined multi-document index stats |
| `POST` | `/multi/generate` | Generate from the combined index across all documents |
| `POST` | `/multi/rebuild` | Rebuild the combined index |

`POST /generate` and `/multi/generate` take `GenerateRequest`:
`learning_objective` (required), `difficulty`, `question_type`, `top_k` (1–10, default 3).

---

## 5. Known gaps to resolve before deployment

Flagged honestly for integration planning — none are blockers for a demo, all matter
for production.

| # | Gap | Impact | Suggested fix |
|---|---|---|---|
| 1 | **No authentication on any endpoint**, and CORS is `allow_origins=["*"]` | Admin endpoints can rebuild graphs or add domains from any origin | API key or JWT at the gateway; restrict CORS to known origins |
| 2 | **No rate limiting** | A generate call is GPU-expensive; trivial to overload | Gateway-level throttle per client |
| 3 | **Judge runs synchronously** inside `/graphrag/generate`, up to 3 attempts | High and variable latency | Async job + polling, or make judging opt-in via a request flag |
| 4 | **Rebuild endpoints are synchronous** and can run for minutes to an hour | Gateway timeouts | Background worker + job-status endpoint |
| 5 | Startup log advertises `GET /evaluate` — **no such route exists** | Misleading; integrators may code against it | Remove the stale log line |
| 6 | `/graphrag/rebuild/{domain_name}` docstring lists only 4 domains; 5 exist (`computer_architecture` missing) | Documentation drift | Update docstring |
| 7 | No API versioning in paths | Breaking changes will hit consumers directly | Prefix `/v1/` before external consumers integrate |
| 8 | **`top_k_nodes` / `neighbour_depth` are accepted but ignored** — overwritten by the difficulty preset | Callers believe they are tuning retrieval when they are not | Either wire them through as overrides, or remove them from `GraphRAGGenerateRequest` |

---

## 6. Quick reference — all 14 endpoints

| # | Method | Path | Group | Integrate |
|---|---|---|---|---|
| 1 | GET | `/health` | System | **P0** |
| 2 | POST | `/upload` | Standard RAG | P2 |
| 3 | POST | `/generate` | Standard RAG | P2 |
| 4 | GET | `/multi/status` | Standard RAG | P2 |
| 5 | POST | `/multi/generate` | Standard RAG | P2 |
| 6 | POST | `/multi/rebuild` | Standard RAG | P2 |
| 7 | GET | `/graphrag/status` | GraphRAG | **P0** |
| 8 | POST | `/graphrag/upload` | GraphRAG | P1 |
| 9 | POST | `/graphrag/generate` | GraphRAG | **P0** |
| 10 | POST | `/graphrag/rebuild/{domain_name}` | GraphRAG | P1 |
| 11 | POST | `/admin/upload-document` | Admin | P1 |
| 12 | POST | `/admin/rebuild-domain` | Admin | P1 |
| 13 | POST | `/admin/add-domain` | Admin | P1 |
| 14 | GET | `/admin/new-documents` | Admin | P1 |

---

## 7. Running the service

```bash
python main_gr.py                  # defaults from config.yml (server.host / port / mode)
uvicorn main_gr:app --port 8000    # alternative

# Then open the interactive contract:
#   http://localhost:8000/docs
```

`config.yml → server.mode: full` is required for the `/graphrag/*` endpoints to be live.
