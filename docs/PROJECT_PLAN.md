# Project plan — flaw register and phased roadmap

**Project:** P000233DS (EdGenAI) · **Author:** Mohammed Danish Alam · **17 September 2026**
**Purpose:** consolidate every known defect and gap into one register, order the work by
dependency and evidence, and replace reactive task-switching with a sequence.

This supersedes ad-hoc prioritisation. `EVALUATION_ROADMAP.md` remains the methodology
reference; this document is the execution plan.

---

## 0. Current sprint — week of 25 September

*Added after the 24 September meeting, which generated six new requests while Phase 5
remained blocked. This section is the ordered answer to "what now". Everything below is
unchanged.*

### Today, before the long weekend — 45 minutes

| # | Task | Why it is first |
|---|---|---|
| 1 | **Verify `danish1804/graphrag-qgen` is private** | 82 client corpus PDFs are tracked and the branch is pushed to both remotes. If it is public this is an NDA breach and nothing else matters until it is fixed |
| 2 | Create `feature/distribution-metrics`, commit today's work | Told not to extend `feature/retrieval-ranking` further |
| 3 | Send the graph stats + structure doc to Bani | He was asked to review it and cannot start without it |
| 4 | **Decide the workbook `groundedness` column, reissue, attach a date** | The only item blocking Phase 5. Five weeks, 0 of 60 |

### Decision needed on item 4 — do not skip this

The `groundedness` column cannot be rated without source material. Two options:

- **Add source passages to the Questions tab.** Honest, but the reconstructed context may
  not be what the model saw, so raters would be judging against the wrong evidence.
- **Drop the column for round 1.** Loses a dimension, keeps the other four sound.

**Recommendation: drop it.** Rating against a context that may be wrong is worse than not
rating it, and the four remaining dimensions still support the alt-test.

### A related point worth stating explicitly

The 60-question corpus was generated **25 August, before the Bloom wiring**. It therefore
already describes a generation path we have changed. That does *not* invalidate it for its
actual purpose — judge calibration needs human labels on the same items the judge scored,
and the judge scored these. **Re-frame it in the instructions as a judge-calibration set,
not a snapshot of current system quality.** That removes the temptation to regenerate it and
restart the five-week wait.

### Next week — cheap, high-value, unblocked

| # | Task | Effort | Addresses |
|---|---|---|---|
| 5 | Check how many missed gold concepts land on **isolated nodes** | 1 hr | Confirms or kills the leading hypothesis for A1 (cov@node 0.55) |
| 6 | **Persist the full generation-time context** in the corpus builder | 1 hr | Groundedness is permanently unmeasurable until this exists |
| 7 | Reword `ANSWER_SYSTEM_PROMPT` to stop inviting parametric recall | 15 min | Direct contradiction with the user prompt in the same call |
| 8 | Entity rules: character doubling, author names, figure fragments | 2 hr | `rrmmiitt ccllaassssiiffiiccaattiioonn` is the top hub in deep_learning |

Items 5 and 8 both attack graph construction, which is where the evidence now points and
which none of the 24 September requests touched.

### Held until labelling resolves

**Do not re-index or switch FAISS to exact search yet.** Both change retrieval, and the
sequencing constraint in the risk section applies: either labelling completes first, or
Phase 2 waits. The FAISS finding is measured and recorded; acting on it can wait a week.

### The six new requests from 24 September — parked, with an order

Recorded so they are decisions rather than drift. None start before items 1–8.

| Request | Verdict | When |
|---|---|---|
| Calibrated domain probability | Script written, needs the `--calibrate` run | Phase 1 |
| Domain spread / vMF concentration | **Measured. All five domains are equally broad** (R 0.73–0.76) — little to exploit at domain level | Done; report only |
| Node-level breadth → question depth control | The version of Samar's idea with teeth, and it feeds the unsolved application-Bloom problem | Phase 3 |
| KL divergence | Needs two distributions. Fits generated-mix vs blueprint, or retrieved vs predicted. Prefer JSD | Phase 4 |
| Embedding model swap | **86.4% held-out domain separation says this is not the bottleneck.** A/B against the expanded gold set, never a blind swap | Phase 2, conditional |
| Diversity, 3–4 questions per query | Already designed | Phase 3 |

---

## 1. Where we actually are

**Measured and working:**

| | Value | Note |
|---|---|---|
| Retrieval content coverage (cov@text) | **0.94** | from 0.82, like-for-like on one ruler |
| Precision @1 | 0.93 | top chunk relevant in 14 of 15 queries |
| Routing accuracy | 90.7% | overall; one domain is the problem |
| Judge | independent, live | Haiku 4.5, distribution 2.2–5.0 (was flat 5.0) |
| Corpus | 60 questions, 0 failures | judge-scored, joinable, workbook built |

**The system generates usable exam questions and we can measure it.** That is the platform
everything below builds on.

---

## 2. Flaw register

Severity: **S1** blocks a deliverable or invalidates a claim · **S2** measurable quality
defect · **S3** gap with no current measurement · **S4** hygiene.

### A. Measured defects — we have numbers

| ID | Flaw | Evidence | Sev |
|---|---|---|---|
| **A1** | **Graph node coverage lags text coverage badly.** Right content reaches the prompt; only half the expected concepts exist as clean nodes | cov@node **0.55** vs cov@text 0.94 | S2 |
| **A2** | **Software engineering ranking starts wrong**, not just dilutes. One query has an irrelevant chunk at rank 1 | p@1 **0.67** (others 1.00), p@k 0.48, mrr 0.83 | S2 |
| **A3** | **Computer architecture corpus gap.** "throughput", "stage", "register" absent from retrieved text | cov@text **0.78** (lowest) | S2 |
| **A4** | **Operating systems misroutes to database.** OS profile scores 0.55–0.62 on its own topics while others reach 0.63–0.78 | routing **67%** for OS vs 90.7% overall; 5 of 7 misroutes are OS | S2 |
| **A5** | **Generation collapses toward "Explain".** Whatever level is requested, output is mostly Bloom L2 | **51% below** requested level; application L2×12 of 16 | S2 |
| **A6** | **Questions referencing unseen material.** "What does Chapter 15 of the course material focus on" | **8%** (5 of 60) not self-contained | S2 |
| **A7** | **k=7 is past the point of diminishing returns.** 7→10 adds zero coverage, costs precision | cov@text 0.94 at both k=7 and k=10; p@k 0.74→0.73 | S2 |
| **A8** | **Judge cannot detect self-containment as a flaw class** | flagged 4.25 vs clean 4.28 — indistinguishable | S2 |
| **A9** | **Gold set too small to decide anything close.** 15 queries means a 0.02 difference is one or two concepts | ARES: below 100–150 labels cannot separate systems differing by 2.5 points | **S1** |

### B. Structural gaps — no measurement exists

| ID | Gap | Why it matters | Sev |
|---|---|---|---|
| **B1** | **No human labels.** All three rater tabs empty after 3 weeks | Blocks Fleiss' kappa, alt-test, judge-vs-human — the entire calibration analysis and the strongest research contribution | **S1** |
| **B2** | **Scalability never measured.** IVF approximate search used on all five graphs; never compared against exact | We traded accuracy for scale and never checked the price | S3 |
| **B3** | **No diversity mechanism.** Same query returns same chunks, so N questions from one outcome are near-identical | Blocks multi-question-per-outcome, which the LO feature needs | S3 |
| **B4** | **No learning-outcome / exam generation.** Designed, not built | Samar's integration request; the product use case | **S1** |
| **B5** | **Dense-only retrieval.** No BM25, no hybrid, no reranker | Hybrid is standard practice and targets exactly the rare-term failures in A3 | S3 |
| **B6** | **No answerability check.** Nothing verifies a question is solvable | Samar-requested; RQUGE-style, needs a non-generator model | S3 |
| **B7** | **Bloom wiring unverified.** Instruction added, controlled comparison not run | A5 may or may not be fixed; currently unknown | S2 |

### C. Hygiene

| ID | Item | Sev |
|---|---|---|
| **C1** | Uncommitted work since 26 Aug; `LEARNING_OUTCOMES_DESIGN.md` never committed | S4 |
| **C2** | Documentation scattered; Bani asked for a consolidated source outside GitHub | S4 |
| **C3** | 4 corpus items lack retrieval context (multi-domain merge not replayable) | S4 |
| **C4** | New PDF added 27 Aug, never indexed | S4 |

---

## 3. How this is ordered

Three rules, applied in order:

1. **Unblock other people first.** Anything waiting on Bani or Samar goes first, because
   their calendar is not ours to control.
2. **Fix measurement before fixing the system.** A9 and B2 are foundational — with 15 gold
   queries, every retrieval decision is a guess dressed as a number. Improving retrieval
   before the ruler is trustworthy produces unfalsifiable claims.
3. **Then order by (evidence of impact ÷ effort).** Prefer defects with a measured size over
   suspected ones.

---

## 4. The plan

### Phase 0 — Stabilise · 1 day

| Task | Addresses | Done when |
|---|---|---|
| Commit everything, push to `industry` | C1 | `git status` clean |
| Run the controlled Bloom comparison (60 items, same seed) | B7, A5 | before/after on one ruler, in CHANGELOG |
| Re-run baseline with the improved classifier | B7 | both sides measured identically |
| Consolidate docs into one living report on Drive | C2 | link sent to Bani |
| **Escalate labelling with a concrete fallback** | **B1** | a date, or reduced scope agreed |

**On B1:** three weeks of silence means another reminder will not work. Propose 20 items
each instead of 60 — that still clears the alt-test floor of 30 total and takes maybe half
an hour. A reduced set that gets done beats a full set that does not.

### Phase 1 — Trustworthy measurement · 1 week

Nothing downstream is believable until this is done.

| Task | Addresses | Method | Done when |
|---|---|---|---|
| Expand gold set 15 → 50 queries | **A9** | 10 per domain, binary TREC-style labels, pooled across retrieval configs | 50 queries, re-pooled |
| Scalability report | **B2** | build time, query latency, memory vs graph size | numbers per domain |
| **Exact vs approximate FAISS** | **B2** | same queries through `IndexFlatIP` and `IndexIVFFlat`, measure top-k agreement | recall@k of IVF vs exact known |
| Distribution report | Samar #5 | cosine score distributions, node degree, question distributions by domain/difficulty/Bloom/judge score | single report |

The exact-versus-approximate check is the one I would not skip. If IVF recall against exact
is 0.95, the trade is defensible. If it is 0.70, `nprobe` needs raising and every retrieval
number is affected.

### Phase 2 — Retrieval quality · 1–2 weeks

Now the measurements can be trusted.

| Task | Addresses | Approach | Acceptance |
|---|---|---|---|
| Entity extraction: prefer NER over noun-chunks | **A1** | biggest single lever on cov@node | cov@node > 0.65 |
| OS domain profile enrichment | **A4** | config only — add distinctive OS terms, remove "deadlock" from database keywords | OS routing > 85% |
| SE precision investigation | **A2** | diagnose why rank 1 is wrong; likely profile or chunk quality | p@1 > 0.85 |
| Reduce k from 7 to 5 | **A7** | config; only after gold set expansion | precision +5pts, coverage loss < 0.02 |
| Hybrid sparse-dense experiment | **B5, A3** | BM25 + dense with rank fusion; targets rare-term misses | CA cov@text > 0.85 |

### Phase 3 — Generation quality · 1–2 weeks

| Task | Addresses | Approach | Acceptance |
|---|---|---|---|
| Bloom iteration | A5, B7 | if prompt instruction is insufficient, condition on graph relationships rather than asking for a level | below-target < 25% |
| Diversity, stage 1 | **B3** | over-fetch 3–5×, relevance threshold, diversify only inside the filtered pool | mean_rel unchanged, pairwise overlap down |
| Diversity, stage 2 | **B3** | explicit node coverage: select by relevance within uncovered concepts | node coverage reported per query set |
| Answerability harness | **B6** | RQUGE-style, independent answerer, measure Bloom distribution before/after filtering | solvability score per question |
| Expand flaw rules | A6, A8 | rules catch what the judge cannot; feed judge findings back as rules | dangling references < 3% |

### Phase 4 — Product · 1–2 weeks

| Task | Addresses | Scope |
|---|---|---|
| Outcome parser + data model | **B4** | messy real CLO text, embedded identifiers, compound detection, explicit `level` |
| Batch endpoint, async job + polling | **B4** | `custom_id` matching, per-item failure isolation, partial results |
| Coverage report | **B4** | the accreditation artefact — which outcomes covered, which not |
| Cross-outcome duplicate flagging | **B4** | threshold calibrated on our own distribution, not a published constant |

Depends on Phase 3 diversity: several questions per outcome is only useful if they differ.

### Phase 5 — Validation · blocked on B1

| Task | Method |
|---|---|
| Inter-rater agreement | Fleiss' kappa on difficulty, Krippendorff's alpha per quality dimension |
| Judge vs human | Scott's π, never raw percent agreement |
| Can the judge replace a rater? | alt-test, ω ≥ 0.5, ε = 0.2 for expert annotators |
| Intended vs perceived difficulty | report as *intended* difficulty, per-domain, with the R² ≈ 0.33 ceiling stated |

---

## 5. Deliberately not doing

Recorded so these are decisions, not oversights.

| Item | Why not |
|---|---|
| Full automated test assembly (MIP optimisation) | Solves selection from an item *bank*; we generate on demand |
| Community detection (1.2) | Optional given cov@text 0.94; revisit only if diversity work needs it |
| Multi-domain context reconstruction | 4 items; replay risks a context differing from the one used, which is silently wrong |
| More documents for most domains | cov@text is 0.94 — coverage is not the bottleneck. Only CA and DL have real gaps, and SE would get *worse* |
| Changing routing before integration | Samar is building against current behaviour |
| Fine-grained Bloom claims | Human agreement is κ 0.12–0.60; report the coarse band only |

---

## 6. Open questions for supervisors

| # | Question | Blocks | Recommendation |
|---|---|---|---|
| 1 | Labelling: full 60 or reduced 20 each? By when? | **All of Phase 5** | 20 each, this week |
| 2 | Learning outcomes: input-and-tagging (A), or graph nodes (B)? | Phase 4 | A now, B after |
| 3 | What is KL divergence wanted *for* — a retrieval model, or comparing distributions? | Phase 1/2 scope | Clarify before building |
| 4 | Is the gold set expansion worth a day? | Phase 1, and confidence in everything after | Yes — it is the ruler |

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Labelling never happens | **High** — 3 weeks elapsed | Loses the strongest research result | Reduce scope; agree a date; consider 2 raters |
| Phase 1 feels like overhead and gets skipped | Medium | Every later claim is unfalsifiable | It is one week and it is the foundation |
| Scope grows from supervisor requests faster than delivery | **High** — 6 new items on 17 Sept | Nothing finishes | This document; agree the order explicitly |
| Retrieval changes invalidate the labelled corpus | Medium | Wasted rater effort | Freeze retrieval until labelling returns |

**The last risk deserves emphasis.** The 60-question corpus describes the system as it was
on 25 August. Phase 2 changes retrieval. If labelling has not returned before Phase 2 lands,
the labels describe a system that no longer exists. **Either labelling completes first, or
Phase 2 waits.** That is a genuine sequencing constraint, not a preference.
