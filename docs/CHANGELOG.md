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
