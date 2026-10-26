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
| Retrieval **content coverage** (cov@text, judge-free) | **0.81** | **0.91** | measured on the 1.5 gold set with the *original* matcher |
| Retrieval **node coverage** (cov@node) | 0.51 | 0.51 | flat — graphs got leaner after cleanup (see §5) |
| Mean chunk relevance | ~0.75 | ~0.75 | retrieved text stays on-topic |
| Database graph **cross-source edges** | 289,405 | **32,508** | ~90% was a duplicate document inflating the count |

### Precision-at-cutoff and the k sweep (2026-08-26)

Requested by Bani: measure P@1/@3/@5/@10 to separate "right information near the top"
from "noise further down". Added multi-cutoff precision plus a `--max-chunks` override to
`retrieval_eval.py`.

**First, the k question.** The benchmark runs **k = 7** (hard-difficulty settings).
Production varies by difficulty: **easy 3, medium 5, hard 7**. P@10 is undefined at k=7 —
dividing by 7 and calling it P@10 measures something else, dividing by 10 penalises the
system for chunks it was never asked to retrieve. The `--max-chunks 10` diagnostic
retrieves deeper than production so the cutoff becomes computable.

| k | cov@text | p@k | p@1 | p@3 | p@5 | p@10 | mean_rel |
|---|---|---|---|---|---|---|---|
| 3 | 0.86 | 0.80 | 0.93 | 0.80 | — | — | 0.762 |
| 5 | 0.92 | 0.79 | 0.93 | 0.89 | 0.79 | — | 0.758 |
| **7 (current)** | **0.94** | 0.74 | 0.93 | 0.89 | 0.84 | — | 0.752 |
| 10 | 0.94 | 0.73 | 0.93 | 0.89 | 0.84 | 0.73 | 0.742 |

**Findings:**

1. **Coverage plateaus at k=7.** k=10 adds zero coverage (0.94 → 0.94) while precision
   and mean relevance both fall. The extra chunks are pure noise.
2. **p@1 = 0.93 at every k** — the top-ranked chunk is relevant in 14 of 15 queries
   regardless of depth. Ranking is sound; dilution occurs below rank 3.
3. **k=5 may be the better operating point:** 0.92 coverage (98% of maximum) at 0.79
   precision. k=7 buys +0.02 coverage for −0.05 precision, a poor trade given that
   non-answer-bearing chunks degrade generation (Cuconasu et al., SIGIR 2024).
4. **A single global k looks wrong.** software_engineering collapses at k=3 (cov@text
   0.67, p@3 0.44) while database and operating_systems hold at 1.00 and 0.92. Optimal
   depth appears domain-dependent.
5. **cov@node is 0.55 at every k**, as expected: node coverage depends on `top_k_nodes`
   (fixed at 5), not chunk count. Confirms the node gap is an extraction problem, not a
   retrieval-depth one.

**Caveats.** Changing k also changes the seed/neighbour allocation (k=5 → 2+3, k=7 → 3+4),
so this is not pure truncation and p@5 differs between the k=5 and k=7 runs. And with 15
gold queries the 0.92/0.94 gap is one or two concepts — directional, not decisive. Expand
the gold set before acting on it.

### Like-for-like before/after (2026-08-25) — single ruler, both sides

Run by scoring the archived pre-batch indexes (`indexes/_backup_prebatch`) with the
**current** harness via `retrieval_eval.py --index-dir`. Same gold set, same metrics,
same text matcher; only the indexes differ. This supersedes the earlier 0.81 → 0.91
figure, which was measured with the older hyphen-blind matcher and is therefore not
comparable to anything measured since.

| Metric | Before | After | Change |
|---|---|---|---|
| **cov@text** (content coverage) | **0.82** | **0.94** | **+0.12** |
| cov@node (graph coverage) | 0.51 | 0.55 | +0.04 |
| mean_rel | 0.751 | 0.752 | flat |
| hit@k | 0.93 | 1.00 | +0.07 |
| mrr | 0.93 | 0.97 | +0.04 |
| ndcg | 0.91 | 0.96 | +0.05 |
| **p@k** (precision) | **0.79** | **0.74** | **−0.05** |

**Largest win — computer architecture.** On the old index the pipelining query failed
outright: cov@text 0.20, hit@k 0, mrr 0, ndcg 0, with "pipelining", "cpu", "throughput"
and "stage" all absent from the retrieved text. It now returns usable context (0.60), and
the domain went 0.57 → 0.78. Separately, "inner join" was missing from retrieval for the
inner-join query on the old index and is now present.

**Regression — software engineering precision.** p@k 0.71 → **0.48**, mrr 1.00 → 0.83,
cov@node 0.47 → 0.38, while its cov@text rose 0.85 → 1.00. The domain now finds all the
right content but surrounds it with more noise. This is the SE precision problem noted in
the current-state table, and the comparison confirms it is an effect of these changes
rather than pre-existing. Investigate before the fine-tuning phase, since noisier context
degrades generation (Cuconasu et al., SIGIR 2024).

### Current measured state (2026-08-20, post entity fixes + corrected matcher)

⚠️ **Do not compare the cov@text figures below against the 0.81 → 0.91 row above.**
Those were measured with a hyphen-blind text matcher that reported false misses on
hyphenated compounds. The 0.81 → 0.91 pair is internally valid (same matcher both
sides); the numbers below use the corrected matcher and are a different ruler. A clean
apples-to-apples restatement would require re-running the pre-batch index in
`indexes/_backup_prebatch` with the current matcher — not yet done.

| Domain | cov@node | cov@text | mean_rel | hit@k | p@k | mrr | ndcg |
|---|---|---|---|---|---|---|---|
| database | **0.80** | **1.00** | 0.763 | 1.00 | 0.81 | 1.00 | 0.98 |
| deep_learning | 0.50 | **1.00** | 0.749 | 1.00 | 0.81 | 1.00 | 1.00 |
| software_engineering | 0.38 | **1.00** | 0.719 | 1.00 | **0.48** | **0.83** | 0.86 |
| operating_systems | 0.48 | 0.92 | 0.750 | 1.00 | **0.90** | 1.00 | 0.99 |
| computer_architecture | 0.57 | **0.78** | 0.777 | 1.00 | 0.71 | 1.00 | 0.97 |
| **OVERALL** | **0.55** | **0.94** | 0.752 | 1.00 | 0.74 | 0.97 | 0.96 |

Graph sizes after the entity fixes (tasks 22 + 24): database 20,518 nodes,
computer_architecture 8,716, operating_systems 7,318, software_engineering 6,824,
deep_learning 4,259 — a net **+1,040 nodes**, the technical-numeric additions
outweighing the junk removals.

**Reading this table:**

- **cov@node (0.55) is now the bottleneck**, not cov@text (0.94). The right content
  reaches the prompt; it is the graph representation that lags.
- **software_engineering**: perfect cov@text but the worst p@k (0.48) and the only
  sub-1.00 mrr (0.83). The concepts are retrieved but heavily diluted with noise —
  a precision problem, not a coverage problem.
- **computer_architecture**: the only real *coverage* gap (cov@text 0.78). Genuine
  misses are "throughput" and "stage" (pipelining) and "register" (memory hierarchy).
- **operating_systems** has the cleanest retrieval of any domain (p@k 0.90).
- **hit@k is 1.00 everywhere and carries no signal** at k=7 on this gold set. Report
  p@k and mrr instead; keep hit@k only as a floor check.

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
| **Entity-extraction quality** — junk concepts | **implemented (task 22)** — rejects section-numbers/headings/initials + recovers soft-hyphen splits; ~690 junk nodes cleared on next re-index; cov@node impact pending measurement. Generic-phrase concepts + short surnames remain (deeper fix: prefer NER over noun-chunks) |
| **Learning-outcome exam generation (full design)** | **Researched and parked (25 Aug)** — see `docs/LEARNING_OUTCOMES_DESIGN.md`. Larger than a feature: outcome data model (CASE-aligned), batch endpoint, exam blueprint / table of specifications, coverage reporting, cross-outcome duplicate detection. Notable market finding: **no surveyed platform accepts a list of outcomes and returns a blueprint-conforming exam** — generation and assembly tools are split, joined by a human tagging step. For 1 Sept ship only the minimum (outcome model with explicit `level`, batch endpoint, per-question tagging, coverage report); full ATA optimisation deliberately out of scope since we generate on demand rather than select from a bank |
| **Multi-domain context reconstruction** | **deferred by decision (25 Aug)**. 4 of 60 corpus items have no retrieval context, so their faithfulness scored 1/5 against an empty string and their overall is depressed (~0.8 pts). Not a design constraint — `generate_with_multi_domain` is fully deterministic and replayable — but reimplementing the merge risks a context that differs from the one actually used, which would be silently wrong. Exclude these 4 from faithfulness/overall comparisons; their clarity, difficulty and type scores remain valid. Cause is fixed forward: the generator now stores `retrieved_chunks` at generation time |
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

- **2026-10-26 — v3 measured; v1 → v2 → v3 comparison (OS and DB), run by Danish:**

  **Graph quality against the textbook indexes** (`graph_quality_report.py`):

  | Measure | OS v1 | OS v2 | **OS v3** | DB v1 | DB v2 | **DB v3** |
  |---|---|---|---|---|---|---|
  | Nodes | 7,318 | 7,063 | 6,161 | 20,518 | 19,389 | 16,376 |
  | Edges | 26,776 | 29,926 | 31,629 | 101,949 | 105,636 | 108,607 |
  | **Index relations directly connected** | 17.8% | 20.0% | **21.4%** | 13.8% | 14.3% | **16.0%** |
  | Index relations within 2 hops | 41.6% | 46.5% | 48.9% | 39.0% | 42.1% | 44.1% |
  | Directly connected, of representable | 32.3% | 34.4% | 36.0% | 25.6% | 25.8% | 28.8% |
  | Index concepts found | 52.6% | 52.8% | 53.3% | 54.4% | 54.8% | 55.8% |
  | **Isolated nodes** | 28.4% | 25.2% | **20.9%** | 29.3% | 27.3% | **23.4%** |
  | Largest component | 69.4% | 73.0% | 77.6% | 68.6% | 71.2% | 75.1% |
  | Numeric-only nodes | 19 | 0 | 0 | 433 | 0 | 0 |

  Every measure improves monotonically from v1 to v3 in both subjects. v1 → v3: direct
  relation recall +3.6 points OS (+20% relative), +2.2 points DB (+16%); isolated nodes −7.5
  and −5.9 points. The 1NF/2NF/3NF regression is fixed (absent from the v3 missing list).
  Strip-and-merge removed 902 OS and 3,013 DB fragment nodes relative to v2.

  **Downstream retrieval** (`retrieval_eval.py`, 6 gold queries, 3 per domain):

  | Graph + weighting | DB cov@node | OS cov@node | Overall cov@node | Overall cov@text | Overall p@k |
  |---|---|---|---|---|---|
  | v1 + degree discount (production) | 0.80 | 0.48 | 0.64 | 0.96 | 0.86 |
  | v2 + degree discount | 0.87 | 0.55 | 0.71 | 0.96 | 0.88 |
  | v2 + npmi_logcount | 0.80 | 0.70 | 0.75 | 0.97 | 0.86 |
  | v3 + degree discount | 0.80 | 0.63 | 0.72 | 0.96 | 0.88 |
  | **v3 + npmi_logcount** | 0.80 | **0.77** | **0.78** | 0.96 | 0.86 |

  **Reading.**
  - The best downstream configuration is v3 + NPMI weighting, with OS node coverage 0.48 →
    0.77. But **the downstream benchmark cannot separate these variants with confidence.** At
    3 queries per domain a single concept moves a domain score by 0.07–0.20, cov@text is at a
    ceiling (0.96 throughout), and changes are not monotonic at query level (DB inner join
    cov@node 0.80 in v2, 0.60 in v3; v2 + NPMI dropped "inner join" from the text but was the
    only run that retrieved "demand paging" text).
  - **The intrinsic textbook-index measure is the stronger evidence here**: it is evaluated
    over 1,212 (OS) and 1,640 (DB) expert relations, not 6 queries, and it moves in one
    direction.
  - **Proposed fix for the benchmark size:** turn the index into retrieval queries. Each index
    head with sub-entries ("Deadlock": avoidance, detection, prevention, ...) becomes a query
    ("Explain deadlock") whose expected concepts are its sub-entries. That yields hundreds of
    queries per subject from an expert source not used to build the graph. Replaces the
    15-query bottleneck (flaw register A9).

  **Recommendation.** Adopt v3 extraction for the next re-index (clear intrinsic gains, no
  downstream loss). Keep npmi_logcount as the leading candidate, pending the larger
  index-derived benchmark. Production unchanged until then and until labelling resolves.
  "demand paging" is still missing (parser limit; v4 glossary-vocabulary candidate).

- **2026-10-13 — Profile v3 and a configurable neighbour weighting (task 8, code):**

  **v3 = v2 plus three fixes found while measuring v2:**
  1. Number-label entities that pass the technical-numeric allowlist are kept (fixes the v2
     loss of 1NF/2NF/3NF).
  2. Leading determiners, possessives and quantifiers are stripped ("its variable" →
     "variable", "each signal" → "signal"), so fragments merge into their concept. Words that
     change meaning are deliberately kept: "no" (no preemption), ordinals (first fit),
     "more"/"most" (most significant bit). Number words are stripped only when not followed by
     phase, tier, way, level and similar ("two phase commit" → 2pc, "three tier architecture"
     kept).
  3. "number", "size", "step", "case", "approach" removed again as non-concepts.

  **New `retrieval.neighbour_weighting`** (`degree_discount` = current, default;
  `npmi_logcount` = NPMI × log(1 + count)). `load_graphrag_index` now stores
  `total_chunks` on the graph, which NPMI needs; missing data falls back to the degree
  discount. `retrieval_eval.py --neighbour-weighting` overrides it per run.

  **Verification:** `scripts/test_entity_profile.py` 47/47 pass (13 new v3 cases, 2 retriever
  weighting cases). Regression on production nodes: v1 rejects 0, v2 rejects 1,003, both
  unchanged by this edit. Production stays on v1 and degree_discount.

  **Deferred (v4 candidate):** known-term vocabulary from textbook glossaries (not indexes, to
  avoid circular evaluation) for terms the parser misses, such as "demand paging".

- **2026-10-13 — Downstream retrieval, v1 vs v2 (OS and DB):**

  `retrieval_eval.py --domains operating_systems,database` on `indexes/` (v1) and
  `indexes_variants/v2`, run by Danish. 6 gold queries (3 per domain), hard settings.

  | | DB v1 | DB v2 | OS v1 | OS v2 | Overall v1 | Overall v2 |
  |---|---|---|---|---|---|---|
  | cov@node | 0.80 | **0.87** | 0.48 | **0.55** | 0.64 | **0.71** |
  | cov@text | 1.00 | 1.00 | 0.92 | 0.92 | 0.96 | 0.96 |
  | p@k | 0.81 | 0.81 | 0.90 | **0.95** | 0.86 | 0.88 |
  | mean_rel | 0.763 | 0.760 | 0.750 | 0.750 | 0.757 | 0.755 |

  **No downstream harm, small gains in node coverage.** Caveat: 3 queries per domain, so each
  change is one or two concepts (inner join query 0.60→0.80; semaphores 0.40→0.60).
  Directional only.

  **"demand paging" is missed by both versions** and is the only uncovered OS gold concept. It
  occurs 10 times in the OS sources (including the Stallings glossary) but never becomes a
  node: a parser limitation of `en_core_web_sm` (likely tagging "demand" as a verb), not a
  filter rule. Candidate fix: a known-term vocabulary (PhraseMatcher) built from the textbook
  **glossaries**. The vocabulary must not come from the back-of-book **index**, which is the
  evaluation gold, or the evaluation becomes circular.

- **2026-10-13 — Neighbour ranking measured against the textbook relations; a v2 regression
  (task 8, first results):**

  **Regression in v2: 1NF, 2NF and 3NF are no longer nodes** (v1 had 1nf–4nf; v2 has only
  4nf; bcnf in both). Cause: spaCy labels "3NF" as an ORDINAL/CARDINAL entity, and v2 drops
  those labels to remove "three"/"four". The drop must exempt strings that pass the technical-
  numeric allowlist. Found by Danish from the report's missing-concept list. v2 stays frozen
  as recorded; the fix goes into the next profile.

  **New measure: what retrieval actually expands to.** Edge weighting does not change which
  edges exist, so edge-presence metrics cannot see it. For every concept with index relations
  among its neighbours (OS 169, DB 199), take its top-5 neighbours under each weighting and
  report: recall@5 of the index relations, generic-word share, and rare-fragment share
  (neighbours with degree <= 2). v2 graphs, gold normalised with v2.

  | Weighting | OS recall@5 | OS generic | OS rare-fragment | DB recall@5 | DB generic | DB rare-fragment |
  |---|---|---|---|---|---|---|
  | raw count | 0.496 | 0.300 | 0.010 | 0.395 | 0.170 | 0.009 |
  | **count / deg^0.5 (current)** | **0.280** | 0.049 | **0.320** | **0.230** | 0.020 | **0.305** |
  | count / deg^0.25 | 0.373 | 0.141 | 0.111 | 0.332 | 0.086 | 0.062 |
  | **NPMI x log(1+count)** | **0.341** | **0.046** | **0.085** | **0.300** | 0.027 | **0.052** |

  **Findings.**
  1. **The current retrieval weighting is dominated.** NPMI x log(count) beats it on recall@5
     (+0.06 to +0.07) and cuts rare fragments about fourfold to sixfold, with essentially the
     same generic share, in both subjects.
  2. **Why the current discount fails:** it removes generic hubs but promotes rare noun-phrase
     fragments. Examples among top-5 neighbours: "its variable", "my availability", "each
     signal", "thread b1", "creasing order", "boundary user", "notemptyqueue".
  3. **Raw counts score highest on recall@5 but 17–30% of their top 5 are generic words**
     ("process", "time", "data"). Gold partners are mid-frequency (median degree 44 vs 2 for
     all nodes; only 3.7% are generic words), so recall@5 alone rewards frequency; it must be
     read alongside the generic and fragment shares. No single weighting wins on all three.
  4. **Fragments come from determiner- and pronoun-led noun chunks** ("its ...", "each ...",
     "two ..."), which the normaliser does not strip (it strips only the/a/an). A simulation
     that strips them and merges the node into its head phrase (approximate: weights summed)
     merges 1,443 OS and 4,582 DB nodes and lowers rare-fragment share under the current
     weighting (OS 0.32 to 0.24, DB 0.31 to 0.24), with relation recall unchanged.

  **Status:** measured on the v2 graphs (deterministic); the strip-and-merge result is a
  simulation, to be confirmed by a rebuild. Index gold relations include some compositional
  phrases ("class → priority" from "priority class"), which adds noise to recall@5.

- **2026-10-13 — v2 measured on operating systems and databases (task 7 result):**

  Built with `graphrag_domain_indexer.py --profile v2 --index-dir indexes_variants/v2`
  (production untouched). Scored with `graph_quality_report.py`, gold normalised with each
  graph's own profile.

  | Measure | OS v1 | OS v2 | DB v1 | DB v2 |
  |---|---|---|---|---|
  | Nodes | 7,318 | 7,063 | 20,518 | 19,389 |
  | Edges | 26,776 | 29,926 | 101,949 | 105,636 |
  | **Index relations directly connected** | 17.8% | **20.0%** | 13.8% | **14.3%** |
  | Index relations within 2 hops | 41.6% | 46.5% | 39.0% | 42.1% |
  | Index concepts found | 52.6% | 52.8% | 54.4% | 54.8% |
  | Isolated nodes | 28.4% | 25.2% | 29.3% | 27.3% |
  | Largest component | 69.4% | 73.0% | 68.6% | 71.2% |
  | Numeric-only nodes | 19 | 0 | 433 | 0 |
  | Max degree | 601 | **1,584** | 2,035 | **2,702** |

  **Reading.**
  - Junk removal works (numeric nodes to zero) and connectivity improves in both subjects.
  - **The relation gap is barely touched.** Fewer than 1 in 5 expert-asserted relations is a
    direct edge in either subject after v2. Extraction cleanup is not the main lever; edge
    construction is (task 8), consistent with the literature review.
  - **Concept recall is flat by construction of the metric.** OS diff: v2 gained 5 index
    concepts (process, data, networks, functions, zombies) and lost 4 marginal ones (batch
    systems, clock interrupt, Peterson's algorithm, superclasses; degree 0–2 in v1, now under
    the minimum-frequency threshold after dropped NER occurrences). Every index term counts
    equally, so restoring "process" counts the same as losing "superclasses".
  - **Use 1-hop relation recall as the headline, not 2-hop.** The restored core words become
    the largest hubs (OS "process" degree 1,584; next "system", "number", "data", "time"), and
    hubs shorten paths, so 2-hop recall partly rewards hub growth rather than meaningful edges.
  - "number" and "time" are arguably non-concepts that v2 now keeps; v2 is frozen for this
    comparison and they are deferred to the task-8 variant.

  **Status:** measured, deterministic builds, one run each. Downstream retrieval on the v2
  graphs not yet measured (`retrieval_eval.py --index-dir indexes_variants/v2
  --domains operating_systems,database`, which now supports a domain subset and reads each
  graph's entity profile).

- **2026-10-13 — Entity extraction profiles: v1 (production) and v2 (fixes) (task 7):**

  Extraction rules are now selected by a profile in `config.yml` (`entity_filter`), stored
  in each `*_meta.json`, and switchable per build (`--profile`) or per process
  (`GRAPHRAG_ENTITY_PROFILE`). **v1 reproduces the 20 Aug indexes exactly** and stays the
  active profile; v2 is built into `indexes_variants/v2` only. The domain indexer refuses to
  write a non-v1 profile into `indexes/`.

  **Defects v2 fixes, all found via the textbook-index report:**
  1. **Singulariser deleted every concept ending in "analysis".** "analysis" became "analysi",
     which a separate rule rejects as an over-singularised plural. No node ending in
     "analysis" exists in any of the five graphs. The same rule produced "statu" (status) in
     four domains and "databas". v2 singularises only the head noun and keeps -sis, -us, -ss,
     -ics endings.
  2. **Generic-word blacklist deleted core concepts:** process, network, function, data, set,
     type. v2 keeps them (genericness to be handled by edge weighting, task 8) and still
     removes true non-concepts (example, result, first, ...).
  3. **Numeric NER labels and person names** (CARDINAL "three"/"four", PERSON lecturer names)
     dropped in v2.
  4. **Page ranges admitted as concepts** by the task-24 pattern `^\d{3,5}\s+\w`; v2 requires a
     processor-family word after the number.
  5. **Numeric protocol abbreviations rejected** (2PC, 3PC, 2PL); v2 admits them and adds 13
     alias mappings (two phase commit → 2pc, MVCC, WAL, OLAP, OLTP, DMA, PCB, FCFS, ...).
  6. **Character-doubled PDF artefacts** ("rrmmiitt ...", deep_learning's #1 hub) rejected in v2.

  **Verification.** `scripts/test_entity_profile.py`: 32/32 pass (11 v1 legacy behaviours
  preserved, 21 v2 fixes). Regression on real data: every node of all five current graphs
  passes v1 (0 changes). Applied to the same nodes, v2 rejects 1,003 junk nodes (DB 618,
  SE 201, CA 135, OS 35, DL 14), before the concepts v1 wrongly deleted are added back on
  rebuild. Effect on concept and relation recall: pending the v2 rebuild of OS and DB.

- **2026-10-12 — Graph-quality measurement against textbook indexes (task 6, baseline):**

  New `scripts/graph_quality_report.py`. Answers Samar's "how do we know a graph is better"
  with references that are expert-made and independent of us: the back-of-book indexes of the
  pilot textbooks. Index **main entries** give a gold concept list; **head → sub-entry** pairs
  give a gold list of expert-asserted relations; **"see" cross-references** give synonym pairs.
  Gold lists go to `evaluation/gold_index/` (git-ignored, corpus-derived). The script also
  writes a 150-edge stratified sample sheet per domain for two-annotator edge precision.

  **Gold sources, chosen by measured parse quality** (my spot-check of 20 concepts and 15
  relations per book): Stallings (OS) ~19/20 and ~14/15 clean; Silberschatz (DB) ~19/20 and
  ~13/15; Garcia-Molina (DB) ~17/20 concepts, no sub-entry structure parsed, concepts only;
  Elmasri (DB) ~8/20 and ~5/15, **excluded**; xv6 book excluded (index of code symbols).
  The parser detects columns and indent levels per page; ligatures, inverted entries and author
  index lines are handled. Parse noise remains, so all recalls below are approximate.

  **Baseline, current indexes (built 20 Aug):**

  | Measure | Operating systems | Databases |
  |---|---|---|
  | Index concepts (gold) | 580 | 2,469 |
  | **Concept recall** (gold concept exists as a node) | **52.6%** | **54.4%** |
  | Index relations (gold) | 1,212 | 1,640 |
  | Both concepts are nodes | 668 (55%) | 883 (54%) |
  | **Relation recall, directly connected** | **17.8%** | **13.8%** |
  | Relation recall, within 2 hops | 41.6% | 39.0% |
  | Directly connected, of representable pairs | 32.3% | 25.6% |
  | Synonym pairs merged into one node | 0 of 16 | 4 of 103 |
  | Isolated nodes | 28.4% | 29.3% |
  | Hub concentration (top 1% of nodes' share of edge ends) | 30.9% | 39.1% |
  | Numeric-only nodes | 19 | 433 |

  **What it shows.** About half of the textbooks' own index concepts are not graph nodes, and
  fewer than 1 in 5 expert-asserted relations is a direct edge. Specific causes found:
  - **The generic-word blacklist deletes core concepts.** `is_valid_entity` rejects "process",
    "network", "function", "type", "set", "data". It was written for deep-learning slides;
    "process" is the central concept of operating systems and is not a node.
  - **Abbreviations with digits are rejected:** 2PC, 2PL, 3PC, J2EE (the full forms "two phase
    commit" / "two phase locking" exist, but the index's abbreviation entries do not match).
  - **Genuine gaps:** "demand paging", "journaling", "boot block", "multiversion concurrency
    control" are absent from their graphs.
  - **Synonyms are almost never merged** (4 of 119 index "see" pairs), so the static alias map
    covers very little.
  - The singulariser bug also affects matching ("zombies" → "zomby").

  **Status:** measured, single run, deterministic. Recall figures are approximate because of
  residual parse noise in the gold lists. Edge precision is pending human annotation of the
  sample sheets.

- **2026-10-12 — Two corrections to 2026-10-08 claims:**

  1. **The Mahalanobis gain is not explained by domain labels.** On 2026-10-08 I attributed it to
     the within-domain covariance turning nearest-centroid into an LDA-like supervised classifier.
     Re-run with covariance estimated from all vectors *without* labels (Ledoit-Wolf, 3 random
     splits, 1,000 train / 1,000 test per domain): cosine 86.0% (sd 0.3), Mahalanobis with
     within-domain covariance 92.8% (sd 0.3), **Mahalanobis with label-free total covariance
     92.8% (sd 0.3)**. The gain comes from the correlation structure of the embedding space, as
     Arun suggested. The 10-NN loss (84.6%) was measured with within-domain covariance only;
     label-free covariance is untested for 10-NN.
  2. **"Deadlock's strongest neighbours include three, four, time, problem" was raw edge weight,
     not what retrieval selects.** Retrieval ranks neighbours by weight / sqrt(degree). Under that
     ranking those four fall to positions 74–108. Retrieval's actual top 5 for deadlock:
     avoidance, prevention, one resource, starvation, dead lock. The discount over-corrects towards
     rare, low-degree fragments: "dead lock" (spelling variant of deadlock, degree 1) and "releas"
     (singulariser bug, degree 1) score 2.00 alongside real concepts. So raw counts favour generic
     hubs and the square-root discount favours rare fragments; edge weighting needs a principled
     fix (NPMI with a minimum count, plus entity cleanup), not a heavier or lighter discount.

- **2026-10-08 — Distance metrics other than cosine (Arun's suggestion), screening test:**

  New `scripts/compare_distance_metrics.py`. Proxy task: 1,000 held-out node vectors per
  domain (5,000 total), classify which domain graph each came from, using 1,000 training nodes
  per domain. Nearest centroid approximates routing; 10-nearest-neighbour approximates seed
  search. Mahalanobis uses a Ledoit-Wolf shrinkage estimate (shrinkage 0.038) of the pooled
  within-domain covariance.

  | Metric | Nearest centroid | 10-NN |
  |---|---|---|
  | Cosine (current) | 85.9% | 92.1% |
  | Euclidean | 85.9% | 92.1% |
  | Manhattan | 85.7% | 92.2% |
  | Mahalanobis | **92.4%** | **84.6%** |

  - **Euclidean is identical to cosine** on unit vectors, as expected (||a-b||^2 = 2 - 2cos).
  - **Manhattan makes no difference.**
  - **Mahalanobis helps the centroid task and hurts the neighbour task.** With a within-domain
    covariance, nearest-centroid Mahalanobis is effectively linear discriminant analysis, a
    classifier that uses the domain labels; that explains the gain. Whitening distorts local
    neighbourhoods, which explains the loss. Consistent with the 24 Sep finding that removing
    principal components reduced separability.
  - **Implication:** worth testing Mahalanobis/LDA for routing on the 75 labelled outcomes
    (OS is the weak domain). Not worth applying to seed search. Status: synthetic proxy, single
    split, seed 0; not yet confirmed on real queries.

- **2026-10-07 — Full repository audit for paper preparation; corrections to earlier claims:**

  Prompted by Bani's request for a complete pipeline description and results compilation.
  Deliverable: `_notes/QG_Pipeline_Results_and_Experiment_Plan_v1.docx` (git-ignored; contains
  corpus-derived statistics). Fact-checked independently against code and docs: about 230
  claims, 19 discrepancies, all corrected before sending. The audit also found that several
  earlier entries in this log are wrong or overstated. Recorded here rather than silently edited:

  1. **Bloom before/after (2026-09-17 entry) did not change only the prompt.** The baseline
     corpus used `force_domain: true`, the after run `false`; multi-domain items went from 4 to
     29 of 60, so retrieval contexts differ. Model sampling was unseeded (seed 42 controls item
     selection only). The direction of the result stands; attribution to the prompt does not.
  2. **Rules vs judge, "flagged 4.25 vs clean 4.28" (2026-08-25) cannot be reproduced** from the
     stored corpus. Recomputed with the original rules: 4.12 (n=5) vs 4.25 (n=55). The
     qualitative conclusion (judge barely separates them) holds.
  3. **`dataset/outputs/comparison_report.md` is invalid:** both sides are the BASE_MODEL file.
     Recomputed from the correct files (207 pairs, self-judged): question score 4.72 vs 4.78,
     Wilcoxon p = 0.33. There is no valid evidence yet that the question LoRA helps.
  4. **Question-LoRA eval leakage:** all 71 eval contexts also occur in the training set, so the
     eval loss of 0.125 is not held-out.
  5. **The independent judge (Claude Haiku 4.5) is the same model that generated the
     question-LoRA training data.** Unrecorded threat to any fine-tuning comparison.
  6. **Judge implementation differs from the standing decisions** (1–5 scale, several
     dimensions per call, reasoning step, no reference answer). Recorded in 2026-08-23 as a
     deliberate deferral; still not done.
  7. **Document de-duplication uses SHA-256**, not MD5/SHA-1; graph nodes are merged by
     normalised string, not hashing.
  8. **New extraction defects:** the singulariser maps `databases` to `databas` (a separate
     node, frequency 352); the technical-numeric pattern `^\d{3,5}\s+\w` admits page ranges.
  9. **Not reproducible from committed code:** graph structure (isolated nodes), vMF spread,
     86.4% separability and the leave-one-out ECE were computed ad hoc. Scripts still to be
     committed.
  10. **No quantitative standard-RAG vs GraphRAG comparison exists**;
      `scripts/compare_rag_vs_graphrag.py` only prints example questions.

- **2026-09-25 — Rater workbook reissued for round 1 (`human_evaluation_ROUND1.xlsx`):**

  **Correction to the 24 Sep groundedness entry.** I described the `groundedness` column as
  unratable because raters cannot see the source passages. Reading its definition again, it
  asks whether the question is *"answerable from the subject material, without inventing
  facts"*, which a subject expert can judge from their own knowledge. The column was
  **mislabelled, not meaningless**: it measures factual soundness, not grounding in the
  retrieved context. Renamed to `factual_soundness` with a rewritten definition, rather than
  dropped. **Consequence for analysis:** it must not be compared with the judge's
  context-based groundedness score as if the two measure the same thing.

  **Column kept in place, not deleted.** Agreement `mean_quality` averages `Rater_x!C:F`;
  deleting column D would have shifted that range without warning. Formulas before/after:
  851/851, 0 changed.

  **Scope: 30 questions, all three raters.** Correction to the "20 each" proposal I gave
  earlier. Twenty *different* items per rater breaks the Fleiss calculation (it needs three
  ratings per item), and the *same* twenty falls short of the alt-test floor of 30. Thirty
  shared items satisfies both, at roughly 45 minutes per rater. Selection is stratified by
  domain and type (OS 8, DB 7, SE 6, CA 4, DL 3, multi 2; application 9, recall 8,
  conceptual 7, case study 6), flagged in a new `rate_in_round_1` column. Rows not in the round
  are greyed on each rater tab.

  **Framing:** the instructions now call this a judge-calibration set generated on 25 August,
  not a snapshot of current system quality.

  **Verified** by filling a copy with random ratings and recalculating in LibreOffice: 30
  items counted as complete, Fleiss κ 0.06, which is what chance should give. Original kept at
  `_notes/human_evaluation_ROUND1_backup_2026-09-25.xlsx`.

- **2026-09-25 — FAISS full run and routing calibration, measured on the maintainer's machine:**

  **FAISS, `measure_scalability.py --synthetic 200` (k=3, noise 0.6, seed 42):**

  | domain | recall@3 at nprobe 16 | top-1 differs from exact | exact ms/query |
  |---|---|---|---|
  | database | 0.718 | 55/200 (27.5%) | 0.806 |
  | deep_learning | 0.820 | 33/200 (16.5%) | 0.113 |
  | software_engineering | 0.772 | 40/200 (20.0%) | 0.168 |
  | operating_systems | 0.747 | 53/200 (26.5%) | 0.188 |
  | computer_architecture | **0.703** | **57/200 (28.5%)** | 0.282 |

  Mean recall@3 **0.752**, consistent with the 24 Sep sandbox preview (0.67–0.77). Exact
  search is under 1 ms on every graph. **Correction to the 24 Sep entry:** I wrote that no
  nprobe setting is both accurate and faster than exact. On this machine that holds for four
  domains, but database at nprobe 64 reaches 0.967 recall at 1.1× exact speed — break-even,
  not slower. The accurate statement is that no setting is *meaningfully* faster while
  accurate. Timings below 0.1 ms are also noisy (database nprobe 2 ran faster than nprobe 1),
  so the speed column should be read as order-of-magnitude. The recommendation stands.

  **Routing calibration, `domain_distribution.py --calibrate` (75 labelled outcomes):**

  | method | NLL | ECE | mean confidence | mean entropy |
  |---|---|---|---|---|
  | proportional (current 1.6) | 1.396 | 0.656 | 25.1% | 0.993 |
  | softmax T=1 | 1.489 | 0.680 | 22.7% | 0.998 |
  | **softmax T=0.0247** | **0.278** | **0.089** | **91.3%** | 0.173 |

  Routing accuracy **90.7%**, which independently reproduces the `test_routing_accuracy.py`
  figure — a useful consistency check between the two scripts. The current proportional
  weighting reports 25% confidence for a router that is right 91% of the time.

  **Out-of-sample check:** T was fitted and scored on the same 75 items, so in-sample ECE is
  optimistic. Leave-one-out refitting gives ECE **0.059**, so calibration holds on held-out
  items. With 75 items in 10 bins ECE is itself noisy; treat both as "well calibrated", not as
  precise values.

  **Calibrated confidence identifies the misroutes.** Mean confidence is 0.94 on correct
  routes and 0.62 on the 7 misroutes (5 of them operating_systems). A threshold of 0.8 flags
  12 of 75 queries and catches **all 7** misroutes, with 5 false alarms. Caveat: 7 errors,
  threshold chosen on the same data. A candidate for the open-set / low-confidence fallback
  proposed on 24 Sep — validate on fresh outcomes before adopting.

- **2026-09-24 — Groundedness is currently unmeasurable, and the rater workbook asks for it anyway:**

  **Prompted by a supervisor question** on whether the generator is restricted to retrieved
  context or can draw on parametric knowledge. The prompt does constrain it — `QUESTION_SYSTEM_PROMPT`
  says the question "MUST be answerable using ONLY the provided context" and "Do NOT introduce
  concepts not present in the context". But that is an instruction, not an enforced property,
  and we have never measured compliance.

  **Attempting to measure it surfaced three defects.**

  1. **The context the model actually saw is not persisted.** `build_question_corpus.py`
     writes `md.get("context")`, and the comment directly above that line states the
     requirement plainly: *"The retrieval context MUST be stored... Without this the corpus
     cannot support post-hoc judging at all."* Measured state of the two corpora:

     | corpus | context field | chunks | usable for groundedness |
     |---|---|---|---|
     | `question_corpus.json` (60, out for labelling) | `context_source: "reconstructed"` ×56, absent ×4 | full | **No** — replayed retrieval, may differ from what was used |
     | `bloom_after.json` (60) | **empty, 60 of 60** | truncated at 200 chars (498 of 590 at the cap) | **No** |

  2. **`ANSWER_SYSTEM_PROMPT` contradicts the groundedness instruction.** It says *"Write as
     if answering from knowledge directly"* while the user prompt in the same call says *"Base
     your answer only on the provided context"*. The intent is stylistic — suppress "according
     to the context" — but as written it invites parametric recall, and it sits in the system
     prompt. Needs rewording to target the style without licensing the behaviour.

  3. **`human_evaluation_ROUND1.xlsx` asks raters to score `groundedness`** while the Questions
     tab supplies only `question_id, domain, question_type, learning_outcome, question_text`.
     Raters cannot see the source material. That column would capture perceived plausibility,
     and nothing would distinguish it from genuine grounding. This is the fourth defect found
     in this workbook before distribution. Labelling has not started, so it is fixable: either
     add the source passages or drop the column for round 1.

  **Process note — a wrong number I nearly reported.** A first pass measured lexical
  groundedness at **0.477** mean content-word coverage, which looked alarming. It was a
  morphology artefact: the matcher failed on "objects" vs "object", "executes" vs "execute".
  With stemming it moved to 0.534 — and by then the `reconstructed` flag showed neither figure
  measures anything real, because the comparison text is not what the model saw. **Neither
  number should be quoted.** Recorded here because the failure mode is instructive: the
  measurement looked like a finding about the system and was a finding about the measurement.

  **No code changed yet.** Persisting the full generation-time context is small; the workbook
  decision is needed before labelling starts. Groundedness checking must use a model other
  than the generator, per the standing decision on answerability.

- **2026-09-24 — Approximate FAISS search is costing us far more than it buys (task 36):**

  **The gap this closes.** `build_graphrag_index()` switches from exact `IndexFlatIP` to
  approximate `IndexIVFFlat` above 1,000 nodes, with `n_clusters = min(√n, 128)` and
  `nprobe = min(n_clusters, 16)`. Every domain graph is above that threshold, so **every
  retrieval in the system has always used approximate search** — and the accuracy cost was
  never measured. Flaw register item **B2**.

  **New `scripts/measure_scalability.py`.** Recovers the stored vectors from the production
  index itself (`make_direct_map` + `reconstruct_n`) and builds an exact index over the
  *same bytes*, so the comparison isolates the approximation. This matters: the indexer
  embeds `f"{node}: {first_two_chunks}"[:500]`, not the bare node name, so re-embedding by
  hand is easy to get subtly wrong. Verified on all five graphs — reconstruction succeeds,
  all 47,635 vectors unit-norm, no zero rows.

  **Index shapes (measured, not assumed):**

  | domain | nodes | n_clusters | nprobe |
  |---|---|---|---|
  | database | 20,518 | 128 | 16 |
  | computer_architecture | 8,716 | 93 | 16 |
  | operating_systems | 7,318 | 85 | 16 |
  | software_engineering | 6,824 | 82 | 16 |
  | deep_learning | 4,259 | 65 | 16 |

  **Recall@3 of approximate vs exact, at production `nprobe=16`:**

  | domain | noise 0.0 | noise 0.3 | noise 0.6 |
  |---|---|---|---|
  | database | 0.992 | 0.748 | **0.672** |
  | deep_learning | 0.998 | 0.847 | **0.772** |
  | software_engineering | 0.995 | 0.797 | **0.752** |
  | operating_systems | 1.000 | 0.803 | **0.772** |
  | computer_architecture | 0.993 | 0.762 | **0.745** |

  **nprobe sweep, database graph (200 synthetic queries, noise 0.6):**

  | nprobe | recall@3 | ms/query | vs exact |
  |---|---|---|---|
  | 1 | 0.150 | 0.022 | 24.9× faster |
  | 4 | 0.365 | 0.059 | 9.4× |
  | 8 | 0.515 | 0.113 | 4.9× |
  | **16 (production)** | **0.707** | 0.243 | 2.3× |
  | 32 | 0.870 | 0.486 | 1.1× |
  | 64 | 0.962 | 1.010 | **0.5× — slower than exact** |
  | 128 | 1.000 | 2.049 | 0.3× |

  **Exact search on the largest graph costs 0.550 ms/query.**

  **The conclusion, and it does not depend on the synthetic queries.** There is no `nprobe`
  setting on these graphs that is both accurate and faster than exact search. Recall only
  reaches 0.96 at `nprobe=64`, and at `nprobe=64` IVF is already **half the speed** of
  brute force — it is scanning half the index plus paying quantiser overhead on top. The
  approximation currently buys about **0.3 ms per query** and costs roughly **30% of the
  seed nodes**. Against a pipeline where embedding the query alone is ~10–20 ms and
  generation is measured in seconds, 0.55 ms is not a cost worth optimising.

  **Why this is worse than a recall number suggests.** Seed nodes are *ranked*, and the
  first one anchors graph traversal. A different top-1 changes the entire neighbourhood
  that gets expanded, so the error does not stay inside retrieval — it propagates into
  which chunks are even candidates. This plausibly contributes to **A1** (cov@node 0.55)
  and **A2** (SE p@1 0.67), though that link is a hypothesis, not yet measured.

  **Status of these numbers — read this before quoting them.** The queries here are
  *synthetic*: real node vectors with Gaussian noise added, then renormalised. They are not
  real questions. This was done because the gold set has only 15 queries, where a single
  query moves recall by 0.067. The noise levels are arbitrary and the absolute recall
  figures should be treated as indicative. What is *not* sensitive to that choice is the
  shape of the nprobe/latency curve and the exact-search timing, and those alone carry the
  conclusion. **Re-run with `--synthetic 0` once the gold set is expanded, and report both.**

  **Not yet changed.** No code has been switched to exact search. That is a config and
  indexer change requiring a re-index, and per project convention the benchmark must be
  regenerated alongside it. Proposed for Phase 1.

- **2026-09-17 — Bloom's taxonomy wired into generation, and a baseline (task 34):**

  **The finding that prompted this: `bloom_level` was a dead config field.** It has sat in
  `config.yml` against every question type since the original build (recall 1, conceptual 2,
  application 3, case_study 4) and **no Python file ever read it**. The generation prompt
  said only `Question type: conceptual`; cognitive level was asserted in configuration and
  never acted on. This is what Samar was asking about.

  **Now wired.** New `utils/bloom.py` turns the configured level into a prompt instruction
  naming the level, its cognitive demand, and preferred verbs. For level 3 and above it
  also **prohibits lower-order verbs** — that clause is what bites, because a request for
  "Apply" is otherwise commonly answered with "Define X", a definition being far easier to
  produce from retrieved context than an application.

  **BASELINE (measured on the 60-question corpus generated 23 Aug, BEFORE the wiring):**

  | | |
  |---|---|
  | Classified by verb | 59/60 |
  | Exact level match | **34%** (indicative only) |
  | Coarse band match (lower vs higher order) | **58%** (the reportable figure) |
  | **Below requested level** | **30 of 59 (51%)** |
  | Above requested level | 9 |

  | question_type | asked | achieved distribution |
  |---|---|---|
  | application | L3 | **L2:12**, L3:3, L6:1 |
  | case_study | L4 | **L2:9**, L3:3, L6:2 |
  | conceptual | L2 | L1:6, L2:5, L6:2 |
  | recall | L1 | L1:12, L2:2, L3:2 |

  **The system systematically collapses toward "Explain".** Whatever level is requested,
  the dominant output is level 2 Understand — 12 of 16 application questions and 9 of 14
  case studies landed there. Only recall, which is the floor, is reliably hit. This is a
  quantified statement of the problem the wiring is meant to fix, and it is a *pre*-change
  baseline, so the comparison will be valid.

  **Measurement is rule-based, per the project standing decision** (rules beat GPT-4 on
  structural checks, 91% vs 79%). `classify_by_verb()` scans highest level first, so the
  highest cognitive demand present governs; for case studies it reads the QUESTION line and
  ignores the scenario.

  **Two classifier defects found and fixed while producing the baseline.** Bare `when` /
  `who` / `where` were matching mid-sentence, so *"When a virtual machine executes an
  interrupt, why must the VMM take control?"* classified as level 1 on the subordinate
  clause instead of level 2 on `why must`. And `what are` / `what does` were missing
  entirely, leaving 5 genuine recall questions unclassified. Interrogatives now only count
  in clause-initial position, checked at the start of each comma-separated clause.
  Classification coverage went 55/60 → 59/60.

  **Honesty caveat carried in the code and the script output:** verb classification is a
  coarse proxy. Human agreement on Bloom level is only Cohen's kappa 0.12–0.60, and GPT-4
  agreed with teacher classification at 65–69% with roughly half its "Analysing" labels
  rated "Evaluating". Report the coarse lower/higher-order band; treat exact-level match as
  indicative. Tests: `scripts/test_bloom.py`; measurement:
  `scripts/measure_bloom_alignment.py`.

  **Post-wiring comparison — now done (2026-09-24).** Regenerated the same 60 learning
  outcomes with the instruction active (`outputs/bloom_after.json`, same seed) and measured
  both corpora with the same classifier. One ruler, both sides; only the prompt differs.

  | | Before | After |
  |---|---|---|
  | **Below requested level** | **51%** (30/59) | **30%** (17/57) |
  | At or above target | 49% | **70%** |
  | Exact level match | 34% | **53%** |
  | Coarse band match | 58% | **67%** |

  | Asked | Before | After |
  |---|---|---|
  | **case_study (L4)** | L2:9, L3:3, L6:2 — **0 of 14 on target** | L2:3, L3:2, **L4:8**, L6:1 — **8 of 14 on target** |
  | **conceptual (L2)** | L1:6, L2:5, L6:2 — 38% | **L2:11**, L3:1, L6:1 — **85%**, L1 drift eliminated |
  | recall (L1) | L1:12 of 16 | L1:11 of 16 — flat, already at the floor |
  | **application (L3)** | L2:12, L3:3, L6:1 — 3 of 16 | L1:1, **L2:11**, L4:1, L6:1 — **0 of 14** |

  **The instruction works, but not uniformly.** Case study never once reached level 4 in the
  baseline and now does so in 8 of 14. Conceptual questions stopped sliding into recall
  entirely. Below-target roughly halved overall.

  **Application did not improve and arguably regressed** (3 of 16 → 0 of 14), still
  collapsing to level 2 despite the explicit prohibition on lower-order verbs.
  *Hypothesis, untested:* "Apply" requires the question to demand a procedure or
  computation, and textbook prose rarely supplies the worked setup to build one from —
  "compare X and Y" is easy to write from two co-occurring concepts, "calculate Z" is not.
  If correct, the fix is not prompt wording but conditioning application questions on chunks
  containing worked examples. Recorded as a hypothesis, not a finding.

  **Caveat:** verb classification remains a coarse proxy (human agreement on Bloom level is
  κ 0.12–0.60). The coarse band figure, 58% → 67%, is the defensible one; exact-level match
  is indicative. Classification coverage differs slightly between runs (59/60 before,
  57/60 after), so per-type n varies by one or two.

- **2026-08-25 — Independent judge live + flaw-detection finding (task 33):**

  **The self-bias is fixed and measurable.** `judge.backend: api` now points at
  `claude-haiku-4-5-20251001` through Anthropic's OpenAI-compatibility endpoint. Chosen
  over OpenRouter because it puts one fewer party in front of corpus-derived content,
  which is the binding NDA constraint. Scored all 60 corpus questions (12 min, ~$0.30).

  | Judge | question_overall distribution |
  |---|---|
  | Qwen 2.5 7B (self, `reuse_generator`) | 5.0 on essentially every item |
  | Claude Haiku 4.5 (independent) | **min 2.20, median 4.40, max 5.00** |

  A real distribution replaces a flat one. This is our own evidence for the self-bias
  problem, not a citation.

  ---

  **FINDING — rule-based and LLM flaw detection are complementary, not redundant.**

  We now have two independent ways to detect a question that refers to material the
  student cannot see: deterministic regex rules (`scripts/inspect_corpus.py`) and the
  LLM judge. They disagree **in both directions**.

  | Question | Rules | Judge |
  |---|---|---|
  | Q004 *"What does **Chapter 15 of the course material** focus on regarding query processing?"* | caught | **missed** — scored 4.6/5 |
  | Q047 *"...according to **the course material**..."* | caught | **missed** — scored 5.0/5 |
  | Q021 *"What are the two main types of testing **mentioned in the context**...?"* | **missed** | caught — scored 2.2, lowest in corpus |

  **Aggregate: flagged items averaged 4.25, clean items 4.28 (n=4 / n=50).**
  Statistically indistinguishable — the judge showed **no ability to detect
  self-containment as a category**. It marked Q021 down for vagueness, not because it
  recognised the flaw class.

  **Why the judge misses these.** The judge is shown the retrieval context, and those
  chunks genuinely do discuss the referenced chapter. From the judge's position the
  question looks faithful to its source. What it fails to adopt is the *student's*
  position, where no chapter 15 is available. The flaw is invisible to a grader who can
  see the source material.

  **Why the rules miss others.** Rules have perfect recall on patterns enumerated in
  advance and zero recall on anything else. "mentioned in the context" simply was not in
  the pattern list.

  **This independently reproduces Moore et al. (2023)**, who found rule-based detection
  caught **91%** of human-annotated item-writing flaws against GPT-4's **79%** on 200
  questions across four subjects (*Assessing the quality of multiple-choice questions
  using GPT-4 and rule-based methods*, EC-TEL 2023,
  <https://arxiv.org/abs/2307.08161>). Our result arrives at the same conclusion from
  our own corpus and adds the reverse case they do not emphasise: the LLM catches
  semantic context-dependence that rules cannot express.

  **Consequence for the pipeline:** keep BOTH stages. Deterministic rules for structural
  flaws (cheap, auditable, exhaustive on known patterns); LLM judge for semantic quality
  (generalises to the unanticipated). This confirms the design already recorded in
  `EVALUATION_ROADMAP.md` §4.2 — now on evidence rather than on citation alone.
  The feedback loop is cheap: when the judge finds a flaw class the rules missed, encode
  it as a rule. "mentioned in the context" has been added, verified not to false-positive
  on legitimate uses such as "context switch overhead".

  ---

  **Final scored distribution (all 60):** min 2.2, median 4.3, mean 4.24, max 5.0,
  SD 0.59. Buckets: 2 items below 3.0, 12 in the 3s, 27 at 4.0–4.4, 8 at 4.5–4.9,
  11 at 5.0. Answer scores: 57/60, mean 4.05.

  **Parse failures.** 6 of 60 items (10%) initially returned no scores because the judge
  wrote its reasoning and omitted the SCORES block. Added the JSON-only forcing retry to
  `judge_corpus.py` (mirroring the fix in `evaluation/evaluator.py`) plus `--retry-failed`
  to re-score only affected items. All 6 recovered.
  *Correction to the obvious inference:* the `retried` flag shows the JSON-only path was
  **never exercised** — the second attempt succeeded on its first call. So these failures
  are **stochastic**, not a systematic prompt defect; the judge occasionally runs long and
  drops the block. The retry remains as insurance but is not what recovered them.

  **Confound to note before quoting per-domain means.** multi_domain scores lowest (3.55,
  n=4), but those are precisely the 4 items whose retrieval context could not be
  reconstructed — the judge was shown *"(retrieval context unavailable)"* and scored
  faithfulness against nothing. Their score reflects missing context, not question
  quality. Excluding them the domain range is 4.07–4.40 (deep_learning 4.07,
  software_engineering 4.20, operating_systems 4.30, database 4.36,
  computer_architecture 4.40), which is tight enough that no domain stands out as weak
  on question quality.

- **2026-08-25 — Routing accuracy measured (task 32):** ran
  `scripts/test_routing_accuracy.py` over all 75 single-domain learning outcomes.
  **Top-1 accuracy 90.7% (68/75)**; 17.3% fall below the 0.05 margin and are forced to
  multi-domain; median margin 0.114.
  **This corrects an earlier over-generalisation.** The 50% domain drift seen while
  generating the corpus was mostly the `difficulty.hard.multi_domain` rule firing *by
  design*, not mass misrouting. The router is largely sound.
  **The failure is confined to one domain:** deep_learning 100%, software_engineering
  100%, database 93%, computer_architecture 93%, **operating_systems 67% correct with
  53% ambiguous**. Five of the seven misroutes are OS topics absorbed by database
  (deadlock detection, banker's algorithm, file systems, page replacement) or by
  deep_learning (memory management), several on margins under 0.02.
  **Diagnosis from the raw scores:** every other domain scores 0.63–0.78 on its own
  topics; operating_systems reaches only 0.55–0.62. Its keyword list is dominated by
  terms generic across computer science (memory, cache, buffer, disk, page, process,
  thread, socket), so its profile centroid sits in shared "systems" space beside
  database and architecture. Deep learning, with distinctive vocabulary, routes
  perfectly.
  **Two of the seven are arguably label errors rather than routing errors** ("Two phase
  locking and deadlock prevention" is filed under database but spans both; floating
  point spans architecture and deep learning), so true accuracy is nearer 93%.
  **Planned fix (post-demo, config only):** enrich the operating_systems profile with
  distinctive terms (banker's algorithm, page replacement, LRU, context switch, kernel
  mode, critical section, mutual exclusion, TLB, inode) and remove "deadlock" from the
  database keyword list. Deliberately NOT changed before 1 September — integration is
  being designed against current behaviour.
- **2026-08-23 — Independent judge wiring + missing retrieval context (task 31):**
  **Gap found before it mattered:** the corpus did not store `retrieved_chunks`, and the
  question rubric needs that context to judge faithfulness and groundedness — the very
  dimension we most want to compare against the human raters. Regenerating would have
  been the obvious fix and the wrong one, since generation samples at temperature 0.7 and
  the new questions would no longer match the rater workbook. Retrieval is deterministic,
  so `scripts/attach_context.py` reconstructs the context for the existing corpus without
  altering a single question (multi-domain items excepted — their merge path cannot be
  faithfully replayed, reported rather than hidden). `build_question_corpus.py` now stores
  the context for future runs.
  Added `scripts/judge_corpus.py`: scores the **frozen** corpus with the independent
  judge, writing to `independent_judge`. `build_eval_sheet.py` prefers those scores over
  anything the generation pipeline produced. Fail-fast guards refuse to run under
  `reuse_generator`, with an empty `base_url`, or without the API key set — verified.
  **One variable at a time:** the existing 1–5 rubrics are used unchanged. The roadmap
  prefers 1–10, but changing scale and judge model together would confound them — any
  score shift could be either cause. Swap the model, measure, then revisit the scale.
  `config.yml` now carries presets for OpenRouter, Azure OpenAI and self-hosted vLLM.
- **2026-08-23 — Blinding leak fixed + full-corpus Excel export (task 30):**
  **The rater workbook was showing the system's intended difficulty.** The Questions
  tab carried a `difficulty_target` column while raters were being asked to judge
  difficulty independently — the exact anchoring the hidden `System_Key` tab exists to
  prevent. Removed from the Questions tab; `system_difficulty` is now written into
  System_Key as a literal rather than as a formula referencing the visible sheet.
  Third defect found in this workbook by inspecting the artefact rather than trusting
  the build. Added `scripts/corpus_to_excel.py`, an **internal** Excel view of the full
  corpus (answers, intended difficulty, pair ids, routing provenance, judge scores,
  self-containment flag) with a red warning on the Summary sheet — this file must never
  be sent to raters.
- **2026-08-23 — Two workbook defects fixed before distribution (task 29):**
  **(1) Broken join.** `build_eval_sheet.py` re-shuffled the corpus and assigned fresh
  sequential ids, so workbook Q001 was a different question from corpus Q001 — **0 of
  60 ids matched**. Rater scores could not have been joined back to the corpus, and
  therefore never compared against judge scores: the calibration analysis would have
  been impossible after the labelling effort was already spent. The corpus is already
  shuffled by the generator, so the sheet now preserves corpus ids and order. Verified
  **60/60 match**. **(2) Pre-filled example row.** The template ships one worked
  example so the expected format is clear; the builder did not clear it, so Rater_1
  opened the live sheet with Q001 already scored (`medium, 4, 5, 4, 4`) — confusing and
  anchoring. Rater tabs are now cleared on build. Both found by inspecting the artefact
  rather than trusting that it built correctly.
- **2026-08-23 — Golden corpus generated (60 items) + routing finding (task 28):**
  First full corpus generated with the judge disabled. **Routing drift forced a design
  change.** With auto-routing, 30 of 60 items (50%) left their requested domain and
  the realised distribution was operating_systems **4** and computer_architecture **1**
  against a plan of 16 and 7 — the two focus domains gutted, and per-domain reporting
  impossible. Root cause: each domain profile is a *single* embedded string of a
  description plus ~30 keywords, producing a diluted centroid; all five domain scores
  cluster in 0.4–0.7 with margins of 0.02–0.05, so `routing.margin_threshold: 0.05`
  fires constantly. Overlapping keywords make it worse ("deadlock" and "b-tree" sit in
  the *database* list, pulling OS topics across). Observed misroutes include "Banker
  algorithm" → database, "Page replacement algorithms" → database, "File systems and
  directory structures" → database, "Memory management and segmentation" →
  deep_learning. Added `scripts/test_routing_accuracy.py` to quantify this against the
  labelled LO bank. **Fix applied to the corpus, not the router:** `--force-domain`
  pins single-domain items (multi-domain items still auto-route), because the corpus
  measures *question* quality and routing is now measured separately. Routing drift
  fell 50% → **7%** and the realised distribution matched the plan exactly (database 16,
  operating_systems 16, computer_architecture 7). **The router itself is deliberately
  unchanged** — Samar is designing integration against current behaviour and changing
  it days before 1 September is not worth the risk. Deferred to post-demo; the likely
  fix is scoring per keyword and taking the max rather than embedding one centroid.
  **Corpus quality:** 60 questions, 0 failures, no duplicates, length 68–1179 chars.
  **8% (5/60) are not self-contained** — two cite chapter/section numbers, three say
  "the course material". Worst case is Q004, *"What does Chapter 15 of the course
  material focus on regarding query processing?"*, which asks about the textbook rather
  than the subject. All are left in deliberately, per the range-restriction argument.
- **2026-08-23 — Corpus inspection + dangling-reference detector (task 27):**
  `scripts/inspect_corpus.py` prints each generated question with its routing
  provenance and flags **non-self-contained questions** using deterministic rules
  (figure/table/section references, "as shown above", "refer to the passage").
  Found immediately in a 4-item smoke test: one question asked the student to
  "describe how it relates to the addressing concerns shown in **Figure 2.10**" —
  unanswerable, since the student cannot see the figure. The generator had copied a
  cross-reference out of the source chunk. Rules rather than an LLM, per Moore et al.
  (EC-TEL 2023), where rule-based detection caught 91% of item-writing flaws against
  GPT-4's 79%. **The detector measures, it does not filter:** flawed items must remain
  in the calibration corpus or the judge's ability to detect flaws cannot be assessed.
  Also reports **routing drift** — the smoke test had 2 of 4 items auto-route away
  from their requested domain, so realised per-domain counts differ from the sampling
  plan. Open item: `routing.margin_threshold` (0.05) sent "Context switching and
  process states" to multi-domain on a 0.046 margin, pulling in database and computer
  architecture context for a clean OS topic. Question quality was unaffected in that
  instance; monitor across the full corpus before changing the threshold.
- **2026-08-20 — Question corpus generator (task 26):** `scripts/build_question_corpus.py`
  builds the golden-corpus question set; `scripts/build_eval_sheet.py` now accepts
  `--corpus`. Three deliberate design decisions, each with a reason:
  **(1) The judge is OFF by default.** The pipeline normally regenerates any question
  scoring below `judge.score_threshold`. For a calibration corpus that is wrong: if the
  judge filters out its own low-scoring questions before humans see them, the human
  labels cover only questions the judge already liked, so the judge's ability to detect
  *bad* questions can never be measured. That is range restriction and it inflates any
  judge-human agreement statistic. Judge scores are instead computed afterwards over the
  fixed corpus, which also means changing judge later does not invalidate a single label.
  **(2) Difficulty and question_type are assigned by the script, not read from the LO
  bank.** In `dataset/inputs/input_questions.csv` the two are perfectly confounded —
  every `easy` is `recall`, every `medium` is `conceptual`, every `hard` is
  `application` — so a rater marking difficulty could equally be responding to question
  format, making the difficulty analysis uninterpretable. The script now crosses them:
  all 12 difficulty x type cells are populated with 4–7 items each. This also fixes a
  latent crash: 15 rows carry `difficulty='case_study'`, which is not a valid difficulty
  and would raise `KeyError` in the pipeline.
  **(3) Matched pairs.** `--pairs N` regenerates N learning outcomes at the opposite
  difficulty, giving a within-item control on whether the difficulty setting changes the
  output at all — evidence a between-item comparison cannot provide. Recorded as
  `pair_id` in the corpus JSON.
  Default selection (60 items): database 16, operating_systems 16, computer_architecture 7,
  software_engineering 8, deep_learning 5, multi_domain 8; easy 24 / medium 16 / hard 20.
  Seeded (42) and shuffled so raters cannot infer grouping.
- **2026-08-20 — Metric corrections found by running the harness (task 25b):** two
  defects in the new metrics, both found and fixed before any number was reported
  externally. **(1) Saturated rank metrics.** The first relevance proxy counted a
  chunk as relevant if it contained *any* gold concept; since gold sets include
  common words ("table", "row", "memory"), nearly every chunk qualified and hit@k,
  mrr and ndcg all pinned at 1.00 — measuring nothing. Now a chunk must contain an
  anchor (multi-word) concept or ≥2 distinct concepts; p@k moved 0.97 → 0.74 and
  gained a real range (0.14–1.00). Gold items may now carry an optional
  `anchor_entities` field. **(2) Hyphen-blind text matching.** `_in_text` normalised
  whitespace but not hyphenation, so "white-box testing" in the source never matched
  "white box testing" in the gold set. This produced false misses on every hyphenated
  compound and understated software_engineering in particular: after the fix its
  cov@text went 0.92 → **1.00** and p@k 0.38 → 0.48; overall cov@text 0.92 → 0.94.
  *Consequence for reporting:* the earlier 0.81 → 0.91 pair was measured with the
  hyphen-blind matcher, so current figures are not directly comparable to it. See the
  warning above the current-state table in §1.
- **2026-08 — Rank/set retrieval metrics (task 25):** added **hit@k, p@k, mrr and
  ndcg@k** to `evaluation/retrieval_eval.py` alongside the existing cov@node /
  cov@text / mean_rel, addressing Bani's request for hit rate and NDCG. Relevance
  is a documented **proxy** (a chunk counts as relevant if it contains at least one
  expected gold concept) because we hold no per-chunk qrels, so the figures are
  directional and not comparable with published IR benchmarks. **cov@text remains
  the headline** — it is already concept-level recall into the prompt; true
  chunk-level Recall@k would require full qrels. **ndcg@k is secondary only**: our
  labels are binary and all k chunks are concatenated into a single prompt, so its
  log position discount is mis-specified (BEIR's stated rationale for nDCG is graded
  judgements, which we do not have). p@k is the decision-relevant addition, since
  near-miss distractors measurably degrade generation (Cuconasu et al., SIGIR 2024).
  Metric maths verified against hand-computed cases.
- **2026-08 — Numeric-prefixed technical concepts (task 24):** two long-standing
  filters (`^\d+[a-z]`, `^\d+[a-z\s]`) plus the 40%-digit-ratio rule and the
  `^\d+[a-z]+\d*$` artifact pattern were rejecting **every entity beginning with a
  digit**. They were written for deep-learning slide artifacts ("2d convolution",
  "1motivation") but also removed an entire class of genuine IT concepts from all
  five graphs: normalisation forms (**1nf–4nf**), bit widths ("32 bit",
  "128 bit key"), processor families ("8086 processor") and n-phase/n-tier/n-stage
  architecture terms. Added a `TECHNICAL_NUMERIC` allowlist and `_is_technical_numeric()`
  guard; also tightened the cross-reference rule so lettered references
  ("figure a.3", "section g.6", "appendix m …") are now caught, which previously
  slipped through. Bare processor numbers ("80386") remain rejected by design —
  indistinguishable from page numbers. Test: `scripts/test_technical_numeric.py`
  (45 cases, all passing); no regression on the task-22 artifact cases.
  **Effect unmeasured until re-index** — expected to help database (normalisation
  forms) and computer architecture (bit widths) most, deep_learning least.
  *Process note:* first reported as a regression in the task-22 regex. That was
  wrong — the task-22 pattern was correct and I had mis-transcribed it from a
  grep display that stripped the escaped dot. The real defect was in older,
  unrelated rules. Corrected before any change was made on that basis.
- **2026-08 — Entity quality (task 22):** hardened the indexer to reject
  section-number/heading/citation-initial junk concepts and recover soft-hyphen
  split words. Preview: **690 junk nodes** would be removed on re-index
  (database 360, CA 149, OS 94, DL 58, SE 29). Expected to lift cov@node
  (previously flat) — number pending the re-index + measurement.
- **2026-08 — Evaluation & judge:** fixed retrieval-judge empty scores (JSON-only retry);
  made the judge pluggable/config-driven and fixed judge_model provenance; added the
  label-free retrieval health check; logged the entity-quality lever.
- **2026-07 — Re-index & measurement:** built the 1.5 judge-free eval + gold set; ran the
  full re-index; measured cov@text 0.81 → 0.91 and the honest graph stats.
- **2026-07 — Data/extraction:** pdfplumber extraction, duplicate exclusion, boilerplate
  filter.
- **2026-07 — Retrieval:** query-time re-ranking, depth-2 cap, alias merging, degree
  down-weighting, confidence-margin routing, score-weighted multi-domain merge.
