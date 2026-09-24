# Learning Outcomes — data model and exam-generation design

**Project:** P000233DS (EdGenAI) · **Author:** Mohammed Danish Alam · **25 August 2026**
**Status:** design proposal for Samar's integration planning. Research-backed; see §7.

---

## 1. The gap

`POST /graphrag/generate` accepts **one** learning outcome as a plain string:

```python
learning_objective: str = Field(..., min_length=3)
```

An educator preparing an exam has a *list* of outcomes and needs a *set* of questions
covering them, with evidence that every outcome was assessed. Today that means N separate
calls, no linkage between them, no coverage report, and no protection against two related
outcomes producing near-identical questions.

**This is a real market gap, not just our gap.** Across the platforms surveyed
(Wayground/Quizizz, Cognii, Questionmark, ExamSoft, Respondus, LMS-integrated tools),
**no product accepts a structured list of outcomes and returns a blueprint-conforming
exam in one request.** The market splits cleanly:

- **Generation** tools take unstructured context (a topic, a pasted passage, a PDF) plus
  scalar knobs, and return loose draft items.
- **Assembly** tools take an already-tagged item bank plus a blueprint, and return a form.

A human tagging step always joins the halves. Wayground comes closest with a standards
picker, but caps at **5 standards**, is K-12 only, and offers **no per-outcome item
count** — you cannot say "3 items for outcome A, 5 for outcome B".

Closing that join is a defensible contribution.

---

## 2. What real learning outcomes actually look like

Verbatim from published Australian course guides. This matters because the data is far
messier than a clean design would assume.

**RMIT — 004111 Operating Systems Principles (COSC1114)**

> 1. Describe how computing resources (such as CPU and memory) are managed by the
>    operating system, describe the basic principles used in the design of modern
>    operating systems.
> 2. Summarise the full range of considerations in the design of file systems; describe
>    the concepts of processes, threading and concurrency, and the techniques for
>    achieving synchronisation in an operation system,
> 5. Evaluate and report appropriate design choices when solving real-world problems

**RMIT — 004083 Database Concepts (ISYS1055)**

> 1. Describe the underlying theoretical basis of the relational database model and apply
>    the theories into practice;
> 5. Apply SQL as a programming language to define database schemas, update database
>    contents and to extract data from databases for specific users' information needs;

**RMIT — 038308 Computer Architecture and Organisation (EEET2261)**

> **CLO2.** Design and critically evaluate memory hierarchies and processor datapaths to
> optimise performance of complex computing systems.

### What this tells us

| Observation | Consequence for the design |
|---|---|
| CLO2 above ends in a **comma**; says "operation system" (sic); CLO3 has no full stop | Outcome text is dirty. Normalise on ingest; never assume clean sentences |
| Most outcomes are **compound** — "Describe X, describe Y", "Summarise A; describe B" | One outcome may need several questions to cover. Item count per outcome is not 1 |
| EEET2261 embeds the identifier **inside** the text (`CLO2. Design and...`) | Parse and strip leading identifiers on ingest |
| UniMelb INFO20003 has an "outcome" that runs into a paragraph of teaching description | Length validation and a warning, not a hard reject |
| RMIT numbering is **local and positional** — every course has a CLO1 | Identity must be `(course_id, guide_version, number)`, never the bare number |
| **"CLO" means opposite things across institutions**: at RMIT it is subject-level; at Deakin a CLO is degree-level and the subject outcome is a ULO | Store an explicit `level` field. Never infer meaning from the acronym |

### The mapping matrix already exists

RMIT publishes an *Overview of Assessment* table on every course guide, with exactly
these columns:

`Assessment Number | Individual or Group | Type | Weight (%) | CLO(s)`

Real instance, 004111 Operating Systems Principles:

| # | Ind/Group | Type | Weight | CLOs |
|---|---|---|---|---|
| 1 | Group | Written work | 10 | 1, 2 |
| 4 | Individual | Exam | 40 | 1, 2, 3, 4, 5, 6 |

**This is the artefact educators already produce, and the reason they need mapping at
all** — accreditation and course review require demonstrating that every outcome is
assessed. Our output should slot into this table, not invent a parallel format.

Note the shape: a **many-to-many junction, with the weight on the assessment**, not on
the link.

---

## 3. Proposed data model

Aligned to 1EdTech **CASE** (the only real interoperability standard for outcomes) so
that importing from an LMS later is a mapping exercise rather than a rewrite.

```jsonc
{
  "outcome_id": "rmit:004111:2026:CLO2",   // globally unique, stable
  "human_code": "CLO2",                    // CASE: humanCodingScheme
  "statement":  "Summarise the full range of considerations in the design of file systems; describe the concepts of processes, threading and concurrency...",
  "level":      "course",                  // course | program | module — EXPLICIT, never inferred
  "context": {
    "institution": "RMIT",
    "course_code": "004111",
    "course_name": "Operating Systems Principles",
    "guide_version": "2026"
  },
  "bloom_level":   "understand",           // optional, derived or supplied
  "domain_hint":   "operating_systems",    // optional; skips routing when known
  "parent_outcome_id": null                // CASE: isChildOf, for PLO linkage
}
```

Three fields carry their weight for specific reasons:

- **`level`** — because the CLO/ULO acronym collision makes inference unsafe.
- **`outcome_id` scoped by course and guide version** — RMIT requires re-approval when
  CLOs change, so an outcome's identity is only meaningful with its version.
- **`domain_hint`** — our router is 90.7% accurate overall but 67% on operating systems.
  When the educator knows the domain, taking it removes that risk entirely.

---

## 4. Proposed API

### 4.1 The exam blueprint

In educational measurement this is a **table of specifications**: outcomes × cognitive
level × item count × weighting. It is the standard planning artefact and it maps directly
onto the formal Automated Test Assembly problem, where an outcome is a categorical item
attribute and "3 items for outcome A" is a two-sided bound on the count of items in that
subset (Songmuang, 2026).

```jsonc
POST /v1/exams/generate
{
  "exam_title": "COSC1114 Final Examination",
  "outcomes": [ { "outcome_id": "...", "statement": "...", ... } ],
  "blueprint": [
    { "outcome_id": "rmit:004111:2026:CLO1", "count": 3,
      "difficulty": "medium", "question_type": "conceptual" },
    { "outcome_id": "rmit:004111:2026:CLO2", "count": 2,
      "difficulty": "hard",   "question_type": "application" }
  ],
  "options": {
    "dedupe_threshold": 0.85,
    "domain": null
  }
}
```

If `blueprint` is omitted, default to an even split across outcomes at the requested
difficulty — but **surface the default in the response**, so nobody mistakes it for a
considered choice.

### 4.2 Async, because it will take minutes

Sixty questions took **12 minutes** in our own corpus run. A synchronous endpoint is not
viable. Return a job immediately:

```jsonc
202 Accepted
{ "job_id": "exam_01H...", "status": "processing",
  "counts": { "requested": 12, "succeeded": 0, "errored": 0 },
  "poll": "/v1/exams/exam_01H..." }
```

**Design borrowed from the Anthropic Message Batches API**, which solves this exact
problem well:

| Pattern | Why adopt it |
|---|---|
| `custom_id` per request, results matched by it | Results may complete out of order; never rely on position |
| Four result states: `succeeded`, `errored`, `canceled`, `expired` | One failed item must not fail the exam |
| Explicit failure isolation | "the failure of one request in a batch does not affect the processing of other requests" |
| Partial results on cancellation | An educator who cancels at item 9 of 12 should keep the 9 |
| Dry-run one request before the batch | Validation errors surfacing only at the end is a known trap |

### 4.3 Response — the coverage report is the point

```jsonc
{
  "exam_title": "...",
  "status": "completed",
  "questions": [
    { "question_id": "Q001", "outcome_id": "rmit:004111:2026:CLO1",
      "question": "...", "sample_answer": "...",
      "difficulty": "medium", "question_type": "conceptual",
      "domain_used": "operating_systems",
      "judge_scores": { "question_overall": 4.4 },
      "flags": [] }
  ],
  "coverage": [
    { "outcome_id": "...CLO1", "requested": 3, "generated": 3, "status": "covered" },
    { "outcome_id": "...CLO6", "requested": 2, "generated": 0,
      "status": "UNCOVERED", "reason": "retrieval returned no relevant context" }
  ],
  "duplicates_removed": 1,
  "blueprint_satisfied": false
}
```

**`coverage` is the accreditation artefact.** An uncovered outcome is the single most
important thing to surface — silently returning 10 questions when 12 were asked for is
the failure mode that would embarrass a user in front of an auditor.

---

## 5. Cross-outcome duplicate detection

Compound and related outcomes will retrieve overlapping context and produce near-identical
questions. In psychometrics these are **"enemy items"**.

The best operational data available (Meng & Li, 2024 — 1,954 IT certification items,
1.9M pairs, authors from AWS Psychometrics): **cosine > 0.6 on a string-based vector model
caught 61.2% of enemy pairs while flagging 0.0% of 27,433 confirmed non-enemies.**

Three cautions before copying that number:

1. **It is specific to string-based VSM on ~30-word items.** It will not transfer
   unchanged to `bge-large-en-v1.5`. Becker & Kao's field advice is to set the threshold
   from your own bank's empirical distribution, not from a published constant.
2. **Similarity alone is insufficient.** Feature correlation with enemy status: similarity
   .66, **answer-key overlap .57**, same topic .41. Compare sample answers too, not just
   question text.
3. **Out-of-the-box transformers underperformed the string baseline** (86.5% vs 86.7%)
   until domain fine-tuning.

**Recommendation:** flag rather than silently drop, at a threshold calibrated on our own
corpus, and report `duplicates_removed` explicitly.

---

## 6. Scope for 1 September

Seven days. Ship the minimum that is genuinely useful and design the rest.

| | Scope | Effort |
|---|---|---|
| **Ship now** | Outcome model with explicit `level`; batch endpoint accepting a list; per-question `outcome_id` tagging; coverage report; async job + polling | Moderate — reuses existing generation |
| **Ship if time** | Duplicate flagging at a calibrated threshold; blueprint with per-outcome counts | Small |
| **Design only** | CASE import/export; ATA optimisation via MIP; mastery thresholds; PLO rollup | — |

**Deliberately not doing** full Automated Test Assembly optimisation. That is a
mixed-integer programming problem over an item *pool*, and we generate on demand rather
than selecting from a bank. Worth revisiting only if we build a persistent item bank.

---

## 7. Does outcome-aligned generation actually work?

Worth knowing before claiming it does.

**Supporting.** Doughty et al. (2024) rated GPT-4 items at **82.9% learning-objective
alignment versus 67.3% for human-written items** (p < 10⁻⁹). Caveat that matters: rated by
the paper's own authors, and overall Fleiss κ was **0.22** — poor agreement, so the
alignment construct itself is unstable.

**Supporting, different construct.** Isley et al. (2025, preprint) ran an iterative
generate-critique-revise architecture across **91 classes and ~1,700 students**, and IRT
analysis suggested AI-generated questions performed **comparably to expert-created
standardised-exam questions**. This is psychometric evidence that items *behave* well; it
is not evidence they measure the *intended outcome*.

**Cautioning.** A 2025 preprint analysing **1,100+ questions across 19 open-source LLMs**
found alignment requires heavy scaffolding — a question taxonomy and structured prompting
lifted quality by 0.29–0.39 points and parsability to 80%.

**Honest position for the report:** alignment is plausible and partly evidenced, but the
strongest number comes from self-rated work with κ = 0.22. Our own human labelling
exercise, with three independent raters and Fleiss' kappa computed properly, will produce
better evidence than the literature currently offers.

---

## 8. Open question for Samar

In the 19 August meeting the question was whether learning outcomes should be **in the
graph**. Two readings, with different costs:

**(a) Outcomes as query input, tagged onto output** — what this document proposes.
Satisfies the mapping requirement and the coverage report. Moderate effort.

**(b) Outcomes as first-class graph nodes** — enables corpus-side analytics: which
outcomes the material actually supports, which are under-served by the source documents,
which concepts no outcome touches. Considerably more work, and it changes the indexer.

Option (a) delivers what accreditation needs. Option (b) answers a question nobody has
asked yet but a course coordinator eventually will. **Recommend (a) for 1 September**, and
revisit (b) after the demo.

---

## References

Meng, H., & Li, J. (2024). Automatic enemy identification: Are we there yet?
*Artificial Intelligence and Applications, 2*(4), 295–306.
https://doi.org/10.47852/bonviewAIA42022424

Songmuang, P. (2026). Review of automated parallel test form assembly.
*Behaviormetrika, 53*(2), 769–803. https://doi.org/10.1007/s41237-026-00293-w

Spaccapanico Proietti, G., Matteucci, M., & Mignani, S. (2020). Automated test assembly
for large-scale standardized assessments: Practical issues and possible solutions.
*Psych, 2*(4), 315–337. https://doi.org/10.3390/psych2040024

van der Linden, W. J. (2005). *Linear models for optimal test design*. Springer.
https://link.springer.com/book/10.1007/0-387-29054-0

Becker, K. A., & Kao, S.-C. (2022). Identifying enemy item pairs using natural language
processing. *Journal of Applied Testing Technology, 23*, 41–52.
https://www.jattjournal.net/index.php/atp/article/view/172634

1EdTech. (n.d.). *Competencies and Academic Standards Exchange (CASE)*.
https://www.1edtech.org/standards/case

Anthropic. (n.d.). *Batch processing*. Claude Platform Docs.
https://platform.claude.com/docs/en/build-with-claude/batch-processing

Australian Qualifications Framework Council. (n.d.). *AQF levels*.
https://www.aqf.edu.au/framework/aqf-levels

RMIT University. (2026). *Course guides* [Handbook].
https://handbook.rmit.edu.au/

Isley, C., et al. (2025). *Assessing the quality of AI-generated exams: A large-scale
field study* (arXiv:2508.08314) [Preprint]. https://arxiv.org/abs/2508.08314
