# GraphRAG Question-Generation — Changes & Results Log

**Project:** P000233DS continuation (EdGenAI) · **Maintainer:** Mohammed Danish Alam
**Scope:** all changes made while carrying forward the existing capstone codebase.
**Status:** living document — append a dated entry under [Changelog](#changelog-by-date) for each new change.

> This is a *technical* record of what changed, why, which files, and the measured
> effect. Numbers are reproducible via the commands in [How to verify](#how-to-verify).

---

## 1. Executive summary of results

| Metric | Before | After | Notes |
|---|---|---|---|
| Retrieval **content coverage** (cov@text, judge-free) | **0.81** | **0.91** | primary result; measured on the 1.5 gold set |
| Retrieval **node coverage** (cov@node) | 0.51 | 0.51 | flat — graphs got leaner after cleanup (see §5) |
| Mean chunk relevance | ~0.75 | ~0.75 | retrieved text stays on-topic |
| Database graph **cross-source edges** | 289,405 | **32,508** | ~90% was a duplicate document inflating the count |

Headline: retrieval **content quality improved measurably (81 → 91%)** and the graph
statistics are now **honest** (the duplicate that fabricated most cross-source edges is
gone). Node-level coverage stayed flat because the graphs are now leaner/cleaner — the
remaining lever is **entity-extraction quality** (see [Open items](#6-known-issues--open-items)).

---

## 2. Starting point (what was carried forward)

The capstone system was intact and functional: 5-domain GraphRAG (NetworkX + spaCy NER +
FAISS + `bge-large-en-v1.5`), Qwen 2.5 7B Instruct generator with QLoRA question/answer
LoRA adapters, an LLM-as-judge evaluation loop, a FastAPI service layer, and an admin/dedup
pipeline. Nothing was rebuilt from scratch.

Problems identified at handover (from Bani's review + a code audit), which drove the work
below:

- **LLM-as-judge self-bias** — the same Qwen model generated *and* graded questions,
  inflating scores (audit of 590 old records: 52% rated the question ≥4.5/5 while rating
  retrieval ≤2.5/5).
- **Garbled text** in chunks (`conne cts`, `wi th`) — assumed OCR, actually a stale PDF lib.
- **Low retrieval quality** — off-topic neighbours, split concepts, boilerplate as content.
- **A duplicate document** inflating the database graph.

---

## 3. Changes — retrieval quality (Bani's list 1.1–1.6 + review)

All are committed on `feature/retrieval-ranking`. Each has a `scripts/test_*.py`.

### 3.1 Query-time chunk re-ranking  *(addresses 1.4 + review Problem B)*
- **File:** `graphrag/graphrag_retriever.py` (`rerank` param, default on).
- **What:** retrieved chunks are re-ordered by cosine similarity to the live query
  (within the seed/neighbour split) before the budget is applied, instead of graph
  insertion order.
- **Why/result:** on the inner-join example this drops off-topic chunks (`LongString`,
  `DELETE…cascade`) and surfaces the join-relevant ones. Single biggest quality win.

### 3.2 Capped, weight-sorted depth-2 traversal  *(review Problem B)*
- **File:** `graphrag/graphrag_retriever.py` (`depth2_cap`, default **50**).
- **What:** depth-2 neighbour expansion was unbounded (741 candidates on the DB graph);
  now sorted by edge weight and capped.
- **Why/result:** compute/robustness bound. Tuned across all 5 domains — cap=50 preserves
  full neighbour relevance while shrinking the candidate pool ~6×.

### 3.3 Synonym / alias merging  *(1.1 — "biggest lever")*
- **File:** `graphrag/graphrag_indexer.py` (`normalize_entity`, `ALIAS_MAP`, hyphen
  normalisation, plural-abbreviation fix).
- **What:** abbreviation/expansion pairs collapse to one canonical node
  (`sql` = `structured query language`, `cnn` = `convolutional neural network`, …). Also
  fixed a pre-existing singulariser bug (`CNNs` → `cnns` instead of `cnn`).
- **Why/result:** stops a concept's signal being split across two nodes. Consolidated ~215
  fragmented concepts on the database graph (incl. `dbms`+`dbmss`+`database management
  system`). Takes effect on re-index.

### 3.4 Degree down-weighting of neighbour selection  *(1.2 — retrieval half)*
- **File:** `graphrag/graphrag_retriever.py` (`degree_penalty`, default **0.5**).
- **What:** neighbour ranking uses `edge_weight / degree^penalty` so generic hub terms
  (e.g. `attribute`, degree 3699) stop dominating.
- **Why/result:** at 0.5, mean neighbour degree halved (34 → 16) with a small relevance
  gain. (The community-detection half of 1.2 is deferred — see open items.)

### 3.5 Confidence-margin routing  *(1.3)*
- **Files:** `graphrag/graphrag_pipeline.py`, `config.yml` (`routing.margin_threshold`
  = 0.05); also dropped filenames from domain profiles in
  `graphrag/graphrag_domain_indexer.py`.
- **What:** auto-routed queries compute the top-2 domain-score margin; if within the
  threshold the query is treated as boundary-spanning and falls back to multi-domain.
  `routing_confidence` is now recorded in response metadata.
- **Why/result:** boundary queries (e.g. "paging maps virtual to physical memory") no
  longer silently lose cross-domain content — confirmed firing live.

### 3.6 Score-weighted multi-domain merge  *(1.6)*
- **File:** `graphrag/graphrag_pipeline.py` (`generate_with_multi_domain`).
- **What:** the multi-domain chunk budget is split proportional to each domain's routing
  score instead of an equal share (total budget unchanged).
- **Why/result:** the dominant domain gets a bigger slice (e.g. database 8 / CA 6 rather
  than 7 / 7).

---

## 4. Changes — data & extraction quality

### 4.1 Extraction: pypdf → pdfplumber
- **File:** `utils/document_loader.py` (`load_pdf`), `requirements.txt`.
- **What:** replaced pypdf with pdfplumber (MIT).
- **Why/result:** the garbled text was **not OCR** — the PDFs aren't scanned. It was the
  pinned **pypdf 4.3.1** mis-decoding kerning (confirmed: 4.3.1 reproduces `conne cts`,
  current extractors don't). Chose pdfplumber (MIT) over PyMuPDF/fitz (AGPL) to avoid a
  copyleft dependency. One residual: pdfplumber run-togethers `source_49` (logged).

### 4.2 Exclude duplicate documents from indexing
- **File:** `utils/registry.py` (`get_domain_files`, `include_duplicates` flag).
- **What:** the indexer honoured `indexed` but ignored the registry's `duplicate_of` flag,
  so `source_73` (== `source_7`) and 2 DL duplicates were indexed.
- **Why/result:** 3 duplicates now excluded. On re-index this dropped the database
  cross-source edge count **289k → 32k** — the inflated figure was ~90% duplicate artifact.

### 4.3 Boilerplate / reference-list filter
- **File:** `utils/document_loader.py` (`is_quality_chunk`, applied in `load_and_chunk`).
- **What:** revived the orphaned quality filter — drops front-matter, TOC, copyright, and
  reference-list chunks (reference lists caught by citation density, not just the word
  "bibliography").
- **Why/result:** stops "According to Lutz (1993)…" style citation questions and removes
  boilerplate nodes. ~6% of chunks dropped on a typical textbook (26% on a reference-heavy
  one). Note: over-drops a few very short/slide-style SE docs — impact confirmed negligible.

---

## 5. Changes — evaluation & the judge

### 5.1 Judge-independent retrieval eval  *(1.5)*
- **Files:** `evaluation/retrieval_eval.py`, `evaluation/retrieval_gold.json`.
- **What:** measures retrieval without the LLM judge. Per query it reports **cov@node**
  (expected concepts among retrieved nodes), **cov@text** (expected concepts in retrieved
  chunk text — the primary metric), and mean chunk relevance. Seed gold set = 3 queries per
  domain (expand toward 20–30).
- **Result:** the pre/post-batch numbers in §1.

### 5.2 Retrieval health check (label-free)
- **File:** `scripts/retrieval_health.py`.
- **What:** run *any* query and flag weak retrieval using two intrinsic signals (top seed
  score, mean chunk relevance) — no gold labels. Prints the seed concepts.
- **Result:** exposed that many seed *concepts* are junk (section numbers `8.3.1`, author
  initials `j. h`, split words, generic phrases) → the entity-quality lever (open item).

### 5.3 Retrieval-judge empty-scores bug fixed
- **File:** `evaluation/evaluator.py` (`_judge_and_parse`).
- **What:** the retrieval rubric sometimes returned empty scores because the judge wrote its
  reasoning and omitted the SCORES JSON. Now each rubric retries once with a JSON-only
  forcing prompt if the first parse yields no scores.
- **Result:** `retrieval_overall` now populates instead of `null`.

### 5.4 Pluggable, decoupled judge  *(self-bias fix — plumbing)*
- **Files:** `utils/judge.py` (`Judge`/`LocalJudge`/`APIJudge`/`build_judge`), `config.yml`
  (`judge.backend`), `graphrag/graphrag_pipeline.py`, `evaluation/evaluator.py`.
- **What:** `load_judge_model()` no longer hard-wires `self.judge = self.model`. The judge
  is config-driven: `reuse_generator` (default, self-bias), `local` (a separate model), or
  `api` (any OpenAI-compatible endpoint — Azure GPT-5-nano or an HPC vLLM server). The eval
  record now logs the **real** judge identity (`judge_model` + `judge_backend`) instead of
  the mislabelled config string.
- **Status:** plumbing complete and verified; switching to an independent judge is now a
  **config change** pending the team's endpoint decision.

---

## 6. Re-index & measured results

Rebuilt all 5 domains (`--domain all`) after the index-time changes (alias merge + dedup +
boilerplate). Graph size before → after:

| Domain | Nodes | Edges | Cross-source |
|---|---|---|---|
| database | 31,887 → **20,274** | 351,183 → 101,340 | 289,405 → **32,508** |
| deep_learning | 6,266 → 4,305 | 26,231 → 15,107 | 3,255 → 9,473 |
| software_engineering | 7,782 → 6,677 | 24,428 → 23,505 | 6,861 → 5,973 |
| operating_systems | 7,671 → 7,371 | 22,525 → 26,818 | 3,161 → 4,930 |
| computer_architecture | 15,824 → 8,687 | 64,777 → 19,129 | 7,465 → 4,889 |

Graphs are leaner (noise removed); where cross-source rose it is genuine cross-document
linking from alias merging (not duplicate inflation). Coverage before/after in §1.

**Live sanity check** (`scripts/generate_one.py`, database learning outcome): produced a
strong question — *"Explain why it is critical to achieve 2NF before proceeding to 3NF…,
and describe what would happen if this order were reversed."* Note the self-bias is visible:
retrieval scored ~3.3 while the question/answer self-scored 5.0/5.0.

---

## 7. Tooling added (`scripts/` and `evaluation/`)

`test_reranking.py`, `test_depth2_cap.py`, `test_alias_merge.py`, `test_routing_margin.py`,
`test_degree_penalty.py`, `test_multidomain_weighting.py`, `test_dedup_exclusion.py`,
`test_boilerplate_filter.py`, `retrieval_health.py`, `generate_one.py`,
`evaluation/retrieval_eval.py` (+ `retrieval_gold.json`).

---

## 8. Known issues / open items

| Item | Status |
|---|---|
| **Independent judge model** (Azure GPT-5-nano vs HPC-hosted) | plumbing done; pending team decision, then flip `judge.backend: api` |
| **Entity-extraction quality** — junk concepts (section nums, author initials, split words, generic phrases) | logged; index-time, needs a re-index; strengthen `is_valid_entity` |
| **1.2 community detection** (Louvain/Leiden) | deferred; optional given cov@text 0.91 |
| **AI-solvability harness** (independent model answers question-only) | not built; judge-free; Samar-requested |
| **Human calibration set** (validate any judge vs humans) | not built; none currently exists |
| **CA `pipelining` retrieval** (cov@text ~0.6) | likely a corpus-coverage gap, not graph |
| **pdfplumber run-together** on `source_49` | logged; per-file extraction repair |
| **CLO / learning-outcome tagging** feature | Samar-sanctioned; to plan post-GraphRAG |
| Full-vs-incremental `cross_source` definition mismatch | pre-existing; align later |

---

## 9. How to verify

```bash
# Judge-free retrieval benchmark (compare cov@text/cov@node)
python evaluation/retrieval_eval.py

# Label-free health check on any queries
python scripts/retrieval_health.py "your query here"

# End-to-end single question from a learning outcome
python scripts/generate_one.py "your learning outcome" medium conceptual

# Rebuild indexes after any index-time change (alias/boilerplate/dedup/entity)
python graphrag/graphrag_domain_indexer.py --domain all
```

---

## Changelog by date

*(append newest at top)*

- **2026-08 — Evaluation & judge:** fixed retrieval-judge empty scores (JSON-only retry);
  made the judge pluggable/config-driven and fixed judge_model provenance; added the
  label-free retrieval health check; logged the entity-quality lever.
- **2026-07 — Re-index & measurement:** built the 1.5 judge-free eval + gold set; ran the
  full re-index; measured cov@text 0.81 → 0.91 and the honest graph stats.
- **2026-07 — Data/extraction:** pdfplumber extraction, duplicate exclusion, boilerplate
  filter.
- **2026-07 — Retrieval:** query-time re-ranking, depth-2 cap, alias merging, degree
  down-weighting, confidence-margin routing, score-weighted multi-domain merge.
