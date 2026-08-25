# Evaluation & Delivery Roadmap

**Project:** P000233DS (EdGenAI) · **Author:** Mohammed Danish Alam · **Date:** 19 August 2026
**Purpose:** a research-grounded plan for the next phase, covering retrieval evaluation,
judge calibration, question quality, and difficulty. Every methodological choice below is
tied to a primary source in [§9 References](#9-references).

> **Verification note.** Sources marked **[V]** were retrieved and read directly. Sources
> marked **[S]** are real, verified-to-exist publications whose specific figure came from a
> search summary rather than full text. Re-check **[S]** figures before they enter a thesis
> or client report. Preprints are labelled as such and should not be cited as settled.

---

## 1. Executive summary

The retrieval work is done and measured. The next phase is **evaluation credibility** — the
difference between "our system produces good questions" as an assertion and as a defensible
claim. The literature review changed six things about how we should proceed:

| # | Planned approach | What the evidence says | Revised approach |
|---|---|---|---|
| 1 | Expert difficulty labels as ground truth | Expert prediction explains only **R² ≈ 0.33** of real item difficulty; after training, r = 0.144 | Relabel as *intended* difficulty. Never present as ground truth |
| 2 | High rater agreement validates our labels | Experts agree with each other (r ≈ 0.73–0.84) far more than with reality (r ≈ 0.46–0.55) — shared bias, not signal | Report agreement **and** state explicitly that it is not validity evidence |
| 3 | Add NDCG for rigour | NDCG's position discount models a human scanning a list; we concatenate all k chunks. Utility by position is **U-shaped**, not log-decreasing | **Recall@k** as headline. NDCG secondary, for literature comparability only |
| 4 | Percent agreement to validate the judge | A judge at >90% percent agreement can still misestimate scores by **10+ points** | Use **Scott's π / Cohen's κ**, and the **alt-test** (ω ≥ 0.5) |
| 5 | 15-query gold set is a starting point | Below ~100–150 labelled items, evaluation cannot discriminate systems that differ by a few points | Expand to ≥50 now, ≥150 if we need fine discrimination |
| 6 | LLM judge checks item quality | Rule-based flaw detection beat GPT-4 (**91% vs 79%**) on item-writing flaws | Deterministic rules for structural flaws; LLM only for semantic judgement |

**The single most important consequence:** the human labelling exercise agreed in the
19 August meeting will produce *reliable* difficulty labels that are *not valid* measures of
difficulty. That is not a reason to cancel it — the quality dimensions are valuable and the
difficulty labels are still useful as "intended difficulty" — but the framing must be
corrected before we build claims on it.

---

## 2. Where we are

**Done and measured:** retrieval improvements 1.1–1.6, judge-free coverage harness
(cov@text 0.81 → 0.91), honest graph statistics, pluggable judge architecture,
entity-quality fix (committed, awaiting re-index).

**Open:** independent judge not yet switched on; no human-validated quality data; no
answerability check; difficulty mapping unvalidated; gold set at 15 queries.

**Fixed dates:** integration begins 1 September · demo 11 September · components complete
by 31 August.

---

## 3. Phase 1 — Integration critical (19 Aug → 1 Sep)

Everything here either unblocks another person or is a prerequisite for the demo.

### 3.1 Re-index, then regenerate the question pool

**Order matters and is not negotiable.** The gold set and the question corpus must be built
from the *post-fix* index, for a reason with direct empirical backing: BEIR's Hole@10
analysis found that when relevance judgements are pooled from one family of retrievers and
then used to score a different retriever, the new system is systematically **under-scored** —
dense retrievers rose from nDCG 0.332 → 0.445 (DPR) and 0.481 → 0.555 (TAS-B) once the pool
was completed by hand [V, Thakur et al., 2021]. We have changed the retriever *and* the
graph, so any benchmark built on the old index is biased against the current system.

- Run the full re-index (~1 hr, unattended).
- Re-run `evaluation/retrieval_eval.py` to record post-fix cov@text / cov@node.
- **Re-pool the gold set** across both the old and new retrieval configurations, not just the new one.

### 3.2 Expand the retrieval gold set (15 → 50 queries)

ARES gives the only effect-size-calibrated number in this literature: below **100–150**
labelled datapoints their framework "cannot meaningfully distinguish between the alternate
RAG systems," where those systems differed by **2.5 accuracy points**; Kendall's τ collapses
from 0.72 at 150 labels to 0.44 at 100 [V, Saad-Falcon et al., 2024].

**Read that number correctly:** 150 is what you need to separate systems 2.5 points apart.
Our observed jump was 10 points (0.81 → 0.91), which needs far fewer. So:

- **Now:** expand to **50 queries** (10 per domain) — sufficient to detect ~10-point changes.
- **Later:** 150+ only if we need to resolve close configurations.
- Always state the effect size alongside any sample size. A sample size without an effect
  size is meaningless.

**Labelling standard — adopt TREC's rule verbatim** [V, NIST, 2000]: *"If you were writing a
report on the subject of the topic and would use the information contained in the document
in the report, then the document is relevant,"* judged binary, and *"a document is judged
relevant if any piece of it is relevant."* The "any piece" rule is what keeps binary
judgement cheap and consistent. **Spend budget on more queries, not finer grades** — nothing
in the retrieved literature shows grade granularity buys discrimination, whereas ARES shows
query count does.

### 3.3 Retrieval metrics — what to add and what to resist

Add **Recall@k**, **Hit Rate@k**, **Precision@k**, **MRR**. Do not make NDCG the headline.

**The argument, since Bani raised NDCG directly.** BEIR standardises on nDCG@10 and states
its reason explicitly: MRR and MAP *"fail to evaluate tasks with graded relevance
judgements"* [V, Thakur et al., 2021, §3.3]. **That justification does not apply to us,
because our judgements are binary.** With binary labels, NDCG carries no information that
Recall@k plus hit positions does not already carry.

There is a deeper mismatch. NDCG's `1/log₂(i+1)` discount models a user scanning a ranked
list with decaying attention. We concatenate all k chunks into one prompt, so every chunk is
"examined" with probability 1. Worse, the true utility-by-position curve is not
monotonically decreasing: models use information at the **beginning and end** of context far
better than the middle [V, Liu et al., 2023], and high-scoring near-miss distractors —
precisely what a good ranker puts at ranks 2–5 — **actively degrade** answer quality, while
random documents can *improve* accuracy by up to 35% [V, Cuconasu et al., 2024].

So NDCG is not merely overkill here; its discount function is **mis-specified** for
concatenated-context RAG. Report it as a secondary number for comparability with published
IR work, and say in the write-up that we know its user model does not match our consumption
pattern. That is a stronger position than either adopting it uncritically or refusing it.

**Also add RAGAS context precision (reference-free)** as a dilution check, motivated directly
by the Cuconasu finding that near-miss distractors hurt [V, Ragas docs, 2025]. Note that
RAGAS **context recall always requires a reference** and cannot be run label-free, and that
RAGAS's own *context relevance* metric should be avoided — ARES measured its accuracy at
**15–36%** across six datasets, and the RAGAS library has since dropped it from the core RAG
metric list [V, Saad-Falcon et al., 2024; Ragas docs, 2025].

### 3.4 Switch on the independent judge

Non-negotiable and already plumbed. The practical guidance is unambiguous even where the
science is contested: Anthropic's own evaluation documentation carries the inline comment
*"Generally best practice to use a different model to evaluate than the model used to
generate the evaluated output"* [V, Anthropic, n.d.].

**Judge design decisions, all evidence-backed** [V, Stureborg et al., 2024 — preprint]:

| Decision | Evidence | Setting |
|---|---|---|
| Scale granularity | Kendall τ vs human: 1–5 stars .339, **1–10 .428**, 1–100 .383 | Use **1–10**, not 1–5 |
| One dimension per call | Human scores across dimensions correlate r = 0.315; GPT-4's correlate **r = 0.979** — severe anchoring in multi-attribute prompts | **One attribute per generation** |
| Chain-of-thought | Contested: helps on math [V, Zheng et al., 2023] but "do not consistently improve agreement" [V, Bavaresco et al., 2025]; OpenAI measured −3.25% on a numeric rater | Default **off**; A/B it, don't assume |
| Position/order | Verdict flips on swap: GPT-4 **46.3%**, ChatGPT **82.5%** [V, Wang et al., 2024] | **Swap and average** for any pairwise comparison |
| Reference-guided | Math judge failure 14/20 → **3/20** with a reference answer [V, Zheng et al., 2023] | Supply the reference answer to the judge |

**Honest framing of self-bias for the write-up.** Do not overclaim. Zheng et al. report
GPT-4 favouring itself by 10% and Claude by 25%, but explicitly state *"our study cannot
determine whether the models exhibit a self-enhancement bias"* [V, Zheng et al., 2023, §3.3].
The strongest causal evidence is Panickssery et al. — GPT-4 self-recognises at **73.5%**, and
self-recognition confidence correlates with self-preference at Kendall τ = 0.41 rising to
0.74 after fine-tuning [V, Panickssery et al., 2024, NeurIPS]. Even there the mechanism is
contested: self-preference may be a **familiarity/perplexity** effect rather than genuine
self-recognition, since judges rate low-perplexity text higher *regardless of who wrote it*
[Wataoka et al., 2025 — preprint]. Our own evidence (52% of 590 records rating the question
≥4.5 while rating retrieval ≤2.5) is a strong internal signal; present it as ours rather than
leaning on contested external effect sizes.

### 3.5 Ship the integration artefacts

API reference (done), question corpus to GitHub, evaluation workbook to raters, basic
learning-outcome mapping. Push to the **`industry`** remote.

---

## 4. Phase 2 — Demo readiness (1 Sep → 11 Sep)

### 4.1 Answerability harness (the "AI-solvability" check Samar asked for)

This has an established name in the literature: **round-trip consistency**. The original
method samples an answer span, generates a question from it, then has an *independently
trained* QA model re-answer; the item is kept only if the recovered span matches
[V, Alberti et al., 2019].

**Use the modern continuous version, not the binary filter.** RQUGE scores answerability
with two off-the-shelf modules — a QA module that answers the generated question from
context, and a span scorer that compares the candidate answer to the gold span — requiring
**no training**. It correlates better with human judgement than BLEU/ROUGE/BERTScore/BLEURT
and, critically, is **robust to adversarial corruptions** that fool n-gram metrics
[V, Mohammadshahi et al., 2023].

**Two design rules:**

1. **The answering model must differ from the generator**, or the check is circular — the
   same reasoning as the judge decoupling.
2. **Measure the bias this introduces.** Round-trip filtering favours easy, extractive,
   lexically-overlapping questions, because those are the ones a QA model reliably
   re-answers. It will systematically discard hard inferential questions — the opposite of
   what we want when targeting higher Bloom levels. Report the Bloom/difficulty distribution
   before and after filtering rather than assuming the filter is neutral.

### 4.2 Item-writing flaw detection — rules first, LLM second

Flawed items are not a cosmetic problem. Across four medical basic-science exams,
**36–65% of items were flawed**; flawed items ran **0–15 percentage points harder** than
sound items measuring the same construct, and pass rates differed by a median of 3.5
points (53% vs 47%) [S, Downing, 2005]. Flaws inject construct-irrelevant variance, which
would corrupt any difficulty calibration layered on top.

**Implement deterministic rules, not an LLM, for structural flaws.** On 200 questions across
four subjects, a rule-based method detected **91%** of human-annotated item-writing flaws
versus GPT-4's **79%** [S, Moore et al., 2023]. Rules are cheaper, auditable, and better.
Reserve the LLM for genuinely semantic judgements.

> **CONFIRMED ON OUR OWN DATA (25 Aug 2026).** Running both methods over the 60-item
> corpus reproduced this independently. The LLM judge scored dangling-reference items at
> **4.25** and clean items at **4.28** — no discrimination whatsoever on self-containment
> as a category. It rated *"What does Chapter 15 of the course material focus on?"* at
> 4.6/5, because the judge is shown the retrieval context and the question does look
> faithful to it; the flaw is only visible from the student's position, which the judge
> does not occupy.
>
> **The reverse also holds, which Moore et al. do not emphasise.** The judge caught
> *"...the two main types of testing mentioned in the context..."* at 2.2 (lowest in the
> corpus) while our rules missed it, because that phrasing was not in the pattern list.
> Rules have perfect recall on enumerated patterns and zero on everything else; the judge
> generalises but overlooks what is plainly in the text.
>
> **Therefore: run both stages, and treat the pairing as a feedback loop.** When the judge
> surfaces a flaw class the rules missed, encode it as a rule — thereafter it is detected
> for free. See CHANGELOG, 2026-08-25.

Checks to implement (from Haladyna's 31 validated guidelines [S, Haladyna et al., 2002]):
negation in the stem, "all/none of the above", implausible or non-functioning distractors,
heterogeneous option lengths (longest-option cue), grammatical cueing, absolute terms
("always"/"never"), vague stems, answer-key position imbalance.

**Consider adopting SAQUET rather than rebuilding.** It is a peer-reviewed, open-source
toolkit implementing a 19-criterion item-writing-flaw rubric, validated at **>94% accuracy**
across Chemistry, Statistics, Computer Science, Humanities and Healthcare
[S, Moore et al., 2024]. Reimplementing it from scratch would be hard to justify.

### 4.3 Human evaluation — run it properly

The workbook is built. Protocol corrections based on published AQG practice:

- **Adopt QGEval's seven dimensions** rather than an ad-hoc rubric — fluency, clarity,
  conciseness, relevance, consistency, answerability, answer consistency. It is EMNLP-2024
  peer-reviewed, so it gives a citable basis for the rubric [V, Fu et al., 2024]. Our current
  four quality columns map onto a subset; extend if rater budget allows.
- **Define every criterion operationally.** A review of 165 NLG papers found **204 distinct
  quality dimensions** collapsing to 71 criteria, with the single word "fluency" resolving
  into **15 different definitions** [S, Howcroft et al., 2020]. An undefined label is not
  comparable to anyone else's.
- **Run a calibration round and report both rounds.** QGEval's Krippendorff's α roughly
  doubled between rounds — **0.181–0.559 → 0.427–0.800** — after guideline refinement and
  annotator training [S, Fu et al., 2024]. Reporting only the post-calibration figure without
  saying so is misleading.
- **Expect moderate agreement and do not panic.** Published AQG work reports Fleiss
  κ = 0.48 / 0.43 / 0.32 for intelligibility / clarity / relevance [S, Ousidhoum et al.,
  2022]. α ≈ 0.4–0.6 on subjective dimensions is the norm, not a failure.
- **Blind the raters** to generation source and to the judge's scores — already enforced by
  the `System_Key` tab. State the blinding explicitly in the write-up.
- **Report** item count, rater count, items per rater, overlap size, and the named agreement
  statistic. Most published work omits at least one [S, Howcroft et al., 2020].

### 4.4 Difficulty — the honest position

**This section exists to stop us making a claim we cannot support.**

| Evidence | Finding |
|---|---|
| Getahun et al. (2024) — 200 MCQs, 815 examinees, 6–7 expert physician raters | Expert prediction explained **R² = 0.33** of actual difficulty. Inter-rater r = **0.725–0.838**; rater-vs-reality r = **0.455–0.545**. Per-domain R² ranged **0.001 to 0.80** [V] |
| Yoon et al. (2024) — 62 faculty, repeated training | Post-training predicted-vs-actual agreement **r = 0.144** [S] |
| São Paulo progress test | **54%** match between expert difficulty expectation and reality on a 3-band task (chance = 33%) [S] |
| BEA 2024 shared task — 667 retired USMLE items, 17 teams, run by NBME | Best system RMSE **0.299** against a predict-the-mean baseline of **0.311** [S] |
| LLM direct difficulty judgement | Spearman ρ ≈ **0.34–0.44**; GPT-5 scored *worse* than GPT-4.1 [preprint] |

Three consequences:

1. **Rename the column.** "Intended difficulty" or "expert-anticipated difficulty", never
   "ground-truth difficulty".
2. **Never cite inter-rater agreement on difficulty as evidence the labels are correct.**
   The Getahun data is the decisive case: raters agreed with each other at r ≈ 0.73–0.84
   while correlating with reality at r ≈ 0.50. That gap is the signature of *shared
   systematic bias*. If a reviewer sees high κ on our difficulty labels and infers validity,
   the inference is wrong and we should pre-empt it in writing.
3. **Report per-domain, not pooled.** Aggregate correlations hide domains where prediction
   is pure noise (R² = 0.001 in one domain of the same exam).

**What we can legitimately do.** If any student response data becomes available, even 30–50
responses per item gives a noisy classical p-value — a genuinely different kind of evidence —
and lets us report the expert-versus-empirical correlation *for our own items*, which is the
honest headline result whatever it turns out to be.

**If we must predict difficulty without responses, simulate rather than ask.** Prompting a
model to "rate this question's difficulty" yields ρ ≈ 0.3. Simulating a cohort of students
at varying ability, collecting their responses, and fitting an IRT model to the simulated
response matrix reaches **r = 0.75–0.82** against real NAEP difficulties [preprint], and the
peer-reviewed SMART pipeline formalises this for open-ended items [S, Scarlatos et al.,
2025]. That is the difference between a weak claim and a defensible one.

**Framing for the report:** difficulty prediction is an open problem that NBME
psychometricians have not solved with real USMLE data and 17 competing teams. A capstone not
solving it is expected. Treat it as a limitations section, not a results claim.

---

## 5. Phase 3 — Research depth (post-11 Sep)

### 5.1 Validate the judge properly — the alt-test

Once human labels exist, the question "may we replace human raters with the judge?" has a
published, rigorous answer procedure rather than a threshold rule of thumb
[V, Calderon et al., 2025, ACL 2025].

**Procedure (leave-one-annotator-out):**

1. Exclude each human rater in turn. Estimate (a) the LLM's alignment with the remaining
   raters and (b) the excluded rater's alignment with the remaining raters.
2. Hypothesis-test whether the LLM's alignment exceeds the excluded human's, minus a
   cost-benefit penalty ε.
3. Apply a Benjamini–Yekutieli FDR correction.
4. **Winning rate ω** = proportion of rejected nulls. **ω ≥ 0.5 justifies the LLM.**

Requirements we already satisfy or nearly do: **≥ 3 annotators per instance** (we have 3),
**≥ 30 instances** absolute floor (our 50–60 corpus clears it). Set **ε = 0.2** for expert
annotators [V, Calderon et al., 2025]. Reference implementation:
<https://github.com/nitaytech/AltTest>.

**Do not report percent agreement as the headline.** Judges at >90% percent agreement still
misestimated system scores by more than 10 points, whereas Scott's π above 80 kept scores
within about 5 points; the gap between percent agreement and π reached **64 points** in their
data [V, Thakur et al., 2025]. Their recommendation is to report **both**, plus qualitative
analysis. Note also that no judge in that study reached human-human parity: *"none of the
judge models we considered were aligned with humans as well as the humans were aligned
amongst themselves."*

**And do not use a good agreement rate to argue we can stop collecting human labels.** An
ICLR 2025 Oral result proves that when the judge is no more accurate than the system being
judged, the sample-efficiency gain is capped at **2× the ground-truth data**, and states
directly that *"high agreement rates are not just insufficient for avoiding judge bias —
they're also insufficient for obtaining a meaningful sample efficiency factor"*
[V, Dorner et al., 2025].

### 5.2 Bloom's taxonomy — coarse claims only

Human raters classifying Bloom levels agree at only **κ = 0.12–0.60**, so the task has a low
reliability ceiling. GPT-4-based classification reaches **64.2% accuracy, κ = 0.547** against
three trained experts [S]. Worse, roughly **50% of questions GPT-4 Turbo labelled
"Analyzing" were rated "Evaluating"** by a teacher — the upper levels, where we most need
discrimination, are where it fails [S, Scaria et al., 2024 — preprint].

Report a coarse split (remember/understand versus apply-and-above), give κ against ≥2 trained
annotators, and never treat the generator's self-assigned Bloom level as a label.

### 5.3 Deferred

Community detection (1.2); graded relevance labels; NDCG with true graded judgements;
fine-tuning round two on clean chunks.

---

## 6. Acceptance criteria

What "done" means for each workstream, so completion is not a judgement call.

| Workstream | Acceptance criterion |
|---|---|
| Re-index | All 5 domains rebuilt; cov@text/cov@node recorded pre/post; CHANGELOG updated |
| Gold set | ≥ 50 queries, ≥ 10 per domain, binary TREC-style labels, pooled across both retrieval configs |
| Retrieval metrics | Recall@k, Hit@k, P@k, MRR reported per domain; NDCG@10 secondary with stated caveat |
| Independent judge | `judge.backend: api`, non-Qwen, 1–10 scale, one dimension per call, reference-guided; provenance logged |
| Answerability | RQUGE-style harness with a non-generator answerer; Bloom distribution reported pre/post filtering |
| Flaw detection | Deterministic rules for ≥ 8 Haladyna categories, or SAQUET adopted |
| Human evaluation | ≥ 3 raters, ≥ 50 items, calibration round run, α reported per dimension for **both** rounds, raters blinded |
| Judge validation | Scott's π **and** alt-test ω reported; percent agreement reported only alongside π |
| Difficulty | Reported as *intended* difficulty; expert-vs-empirical correlation reported if any response data exists; limitations stated |

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Rater availability slips past 11 Sep | High | Blocks judge validation | Send corpus in Phase 1; a partial set of 30 items still clears the alt-test floor |
| Gold set expansion competes with integration work | Medium | Weak retrieval claims | 50 queries is the floor; expanding to 150 is explicitly deferred |
| Difficulty results look weak | High | Perceived as project failure | Pre-empt with the BEA 2024 and Getahun evidence — this is a hard open problem, not our shortfall |
| Judge endpoint decision stalls | Medium | Blocks all quality claims | OpenRouter works with existing `APIJudge` unchanged; proceed and swap later |
| Bedrock workshop collides with demo prep | Medium | Demo slips | Already proposed for post-11 Sep |
| NDA / wrong-remote push | Low | Serious | Verify `git remote -v` before every push of generated content |

---

## 8. Immediate next actions

1. Send the API documentation email (unblocks Samar's integration architecture).
2. Start the full re-index.
3. Generate the question corpus from the re-indexed system; populate the workbook via
   `scripts/build_eval_sheet.py`; distribute to raters with a calibration round first.
4. Expand the retrieval gold set to 50 queries using the TREC labelling rule.
5. Switch `judge.backend` to `api` and smoke-test with `scripts/test_api_judge.py`.
6. Add Recall@k / Hit@k / P@k / MRR to `evaluation/retrieval_eval.py`.

---

## 9. References

Alberti, C., Andor, D., Pitler, E., Devlin, J., & Collins, M. (2019). Synthetic QA corpora
generation with roundtrip consistency. In *Proceedings of the 57th Annual Meeting of the
Association for Computational Linguistics* (pp. 6168–6173). Association for Computational
Linguistics. https://aclanthology.org/P19-1620/

Anthropic. (n.d.). *Create strong empirical evaluations* [Documentation].
https://docs.claude.com/en/docs/test-and-evaluate/develop-tests

Bavaresco, A., Bernardi, R., Bertolazzi, L., Elliott, D., Fernández, R., Gatt, A., …
Testoni, A. (2025). LLMs instead of human judges? A large scale empirical study across
20 NLP evaluation tasks. In *Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Vol. 2: Short Papers)* (pp. 238–255).
https://aclanthology.org/2025.acl-short.20/

Calderon, N., Reichart, R., & Dror, R. (2025). The alternative annotator test for
LLM-as-a-judge: How to statistically justify replacing human annotators with LLMs. In
*Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Vol. 1: Long Papers)* (pp. 16051–16081). https://aclanthology.org/2025.acl-long.782/

Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Campagnano, C., Maarek, Y.,
Tonellotto, N., & Silvestri, F. (2024). The power of noise: Redefining retrieval for RAG
systems. In *Proceedings of SIGIR '24*. https://doi.org/10.1145/3626772.3657834

Dorner, F. E., Nastl, V. Y., & Hardt, M. (2025). Limits to scalable evaluation at the
frontier: LLM as judge won't beat twice the data. In *International Conference on Learning
Representations (ICLR 2025, Oral)*. https://arxiv.org/abs/2410.13341

Downing, S. M. (2005). The effects of violating standard item writing principles on tests
and students. *Advances in Health Sciences Education, 10*(2), 133–143.
https://pubmed.ncbi.nlm.nih.gov/16078098/

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of
retrieval augmented generation. In *Proceedings of EACL 2024: System Demonstrations*
(pp. 150–158). https://aclanthology.org/2024.eacl-demo.16.pdf

Fu, W., Wei, B., Hu, J., Cai, Z., & Liu, J. (2024). QGEval: Benchmarking multi-dimensional
evaluation for question generation. In *Proceedings of EMNLP 2024* (pp. 11783–11803).
https://aclanthology.org/2024.emnlp-main.658/

Getahun, D. A., et al. (2024). Experts' prediction of item difficulty of multiple-choice
questions in the Ethiopian Undergraduate Medicine Licensure Examination. *BMC Medical
Education, 24*, 1049. https://link.springer.com/article/10.1186/s12909-024-06012-x

Haladyna, T. M., Downing, S. M., & Rodriguez, M. C. (2002). A review of multiple-choice
item-writing guidelines for classroom assessment. *Applied Measurement in Education, 15*(3),
309–333. https://eric.ed.gov/?id=EJ660246

Howcroft, D. M., Belz, A., Clinciu, M.-A., Gkatzia, D., Hasan, S. A., Mahamood, S., …
Rieser, V. (2020). Twenty years of confusion in human evaluation: NLG needs evaluation sheets
and standardised definitions. In *Proceedings of INLG 2020* (pp. 169–182).
https://aclanthology.org/2020.inlg-1.23/

Kurdi, G., Leo, J., Parsia, B., Sattler, U., & Al-Emari, S. (2020). A systematic review of
automatic question generation for educational purposes. *International Journal of Artificial
Intelligence in Education, 30*(1), 121–204.
https://link.springer.com/article/10.1007/s40593-019-00186-y

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P.
(2023). *Lost in the middle: How language models use long contexts* (arXiv:2307.03172).
https://arxiv.org/abs/2307.03172

Mohammadshahi, A., Scialom, T., Yazdani, M., Yanki, P., Fan, A., Henderson, J., & Saeidi, M.
(2023). RQUGE: Reference-free metric for evaluating question generation by answering the
question. In *Findings of the ACL: ACL 2023* (pp. 6845–6867).
https://aclanthology.org/2023.findings-acl.428/

Moore, S., Nguyen, H. A., Chen, T., & Stamper, J. (2023). Assessing the quality of
multiple-choice questions using GPT-4 and rule-based methods. In *EC-TEL 2023*, LNCS 14200
(pp. 229–245). Springer. https://arxiv.org/abs/2307.08161

Moore, S., Costello, E., Nguyen, H. A., & Stamper, J. (2024). An automatic question usability
evaluation toolkit. In *Artificial Intelligence in Education (AIED 2024)*, LNCS 14829
(pp. 31–46). Springer. https://arxiv.org/abs/2405.20529

National Institute of Standards and Technology. (2000). *Text REtrieval Conference (TREC)
data — English relevance judgements*. https://trec.nist.gov/data/reljudge_eng.html

Nema, P., & Khapra, M. M. (2018). Towards a better metric for evaluating question generation
systems. In *Proceedings of EMNLP 2018* (pp. 3950–3959). https://aclanthology.org/D18-1429/

Ousidhoum, N., Yuan, Z., & Vlachos, A. (2022). Varifocal question generation for
fact-checking. In *Proceedings of EMNLP 2022*. https://arxiv.org/abs/2210.12400

Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM evaluators recognize and favor their
own generations. In *Advances in Neural Information Processing Systems 37*.
https://arxiv.org/abs/2404.13076

Ragas. (2025). *List of available metrics* [Documentation].
https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An automated evaluation
framework for retrieval-augmented generation systems. In *Proceedings of NAACL-HLT 2024
(Vol. 1: Long Papers)* (pp. 338–354). https://aclanthology.org/2024.naacl-long.20.pdf

Scaria, N., Dharani Chenna, S., & Subramani, D. (2024). *How effective is GPT-4 Turbo in
generating school-level questions from textbooks based on Bloom's revised taxonomy?*
(arXiv:2406.15211) [Preprint]. https://arxiv.org/abs/2406.15211

Scarlatos, A., Fernandez, N., Ormerod, C., Lottridge, S., & Lan, A. (2025). SMART: Simulated
students aligned with item response theory for question difficulty prediction. In
*Proceedings of EMNLP 2025*. https://aclanthology.org/2025.emnlp-main.1274/

Stureborg, R., Alikaniotis, D., & Suhara, Y. (2024). *Large language models are inconsistent
and biased evaluators* (arXiv:2405.01724) [Preprint]. https://arxiv.org/abs/2405.01724

Thakur, A. S., Choudhary, K., Ramayapally, V. S., Vaidyanathan, S., & Hupkes, D. (2025).
Judging the judges: Evaluating alignment and vulnerabilities in LLMs-as-judges. In
*Proceedings of GEM 2025*. https://aclanthology.org/2025.gem-1.33/

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A
heterogeneous benchmark for zero-shot evaluation of information retrieval models. In
*NeurIPS Datasets and Benchmarks Track*. https://arxiv.org/abs/2104.08663

Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., … Sui, Z. (2024). Large language
models are not fair evaluators. In *Proceedings of ACL 2024 (Vol. 1: Long Papers)*
(pp. 9440–9450). https://aclanthology.org/2024.acl-long.511/

Wataoka, K., Takahashi, T., & Ri, R. (2025). *Self-preference bias in LLM-as-a-judge*
(arXiv:2410.21819) [Preprint]. https://arxiv.org/abs/2410.21819

Yaneva, V., North, K., Baldwin, P., Ha, L. A., Rezayi, S., Zhou, Y., … Clauser, B. (2024).
Findings from the first shared task on automated prediction of difficulty and response time
for multiple-choice questions. In *Proceedings of BEA 2024* (pp. 470–482).
https://aclanthology.org/2024.bea-1.39/

Yoon, H. B., et al. (2024). The impact of repeated item development training on the
prediction of medical faculty members' item difficulty index. *BMC Medical Education, 24*,
592. https://pmc.ncbi.nlm.nih.gov/articles/PMC11140961/

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., … Stoica, I. (2023).
Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information
Processing Systems 36 (Datasets and Benchmarks Track)*. https://arxiv.org/abs/2306.05685
