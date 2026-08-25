Codex will review your output once you are done.

# Project context — P000233DS (EdGenAI)

GraphRAG-based exam question generation across 5 IT domains (deep_learning, database,
software_engineering, operating_systems, computer_architecture). Stack: NetworkX + spaCy NER
+ FAISS + `bge-large-en-v1.5` retrieval; Qwen 2.5 7B Instruct generator with QLoRA question
and answer adapters; FastAPI service in `main_gr.py`; LLM-as-judge evaluation loop.

Maintainer: Mohammed Danish Alam. Industry supervisors: Samar Shailendra (integration,
architecture), Banibrata Ghosh (evaluation, question assessment). NDA in place.

## Read these first

| Document | What it holds |
|---|---|
| `docs/EVALUATION_ROADMAP.md` | **Primary planning document.** Phased plan, methodology decisions with citations, acceptance criteria, risk register. Consult before proposing evaluation work |
| `docs/CHANGELOG.md` | Living technical log — every change, why, which files, measured effect. Append a dated entry for each new change |
| `docs/KEY_REFERENCES.md` | Curated peer-reviewed reading list behind the methodology choices; shareable with the team |
| `docs/API_REFERENCE.md` | All 14 endpoints, request/response schemas, known deployment gaps |

`_notes/` is gitignored and holds personal drafts, meeting prep, and reports. Do not commit it.

## Standing methodological decisions

These are settled and evidence-backed; see `docs/EVALUATION_ROADMAP.md` for sources. Do not
silently reverse them.

- **Never let the generator judge its own output.** `judge.backend` must not be
  `reuse_generator` for any reported result.
- **Judge configuration:** 1–10 scale (not 1–5), one dimension per call (multi-attribute
  prompts anchor severely), chain-of-thought off by default, reference-guided grading on,
  swap order for pairwise comparisons.
- **Retrieval headline metric is Recall@k**, not NDCG. Our labels are binary and all k chunks
  are concatenated into one prompt, so NDCG's position discount is mis-specified. Report
  NDCG@10 only as a secondary number for literature comparability, with that caveat stated.
- **Difficulty labels from experts are "intended difficulty", never ground truth.** Expert
  prediction explains only ~33% of real item difficulty. Never cite inter-rater agreement on
  difficulty as evidence the labels are valid.
- **Judge validation reports Scott's π or Cohen's κ**, never raw percent agreement alone.
  Use the alt-test (ω ≥ 0.5) once human labels exist.
- **Item-writing flaw detection uses deterministic rules**, not an LLM — rules measurably
  outperform GPT-4 on structural flaws.
- **Gold-set labelling is binary, TREC-style.** Spend budget on more queries rather than
  finer relevance grades. Re-pool the gold set whenever the retriever changes.
- **Answerability checks must use a different model from the generator**, or the check is
  circular.

## Conventions

- Two git remotes: `origin` (personal) and `industry` (EdGenAI). **Verify with
  `git remote -v` before pushing.** Generated questions and corpus-derived content go to
  `industry` only.
- No hardcoded parameters in source — everything lives in `config.yml`.
- Every behavioural change gets a `scripts/test_*.py` and a CHANGELOG entry.
- Re-index before generating any question set used for evaluation or labelling; otherwise
  the benchmark describes a system that no longer exists.
- Australian English in all written deliverables. No em-dashes in drafted emails.

## Verification standard

Do not report a number without saying how it was produced. Distinguish measured results from
estimates, and peer-reviewed sources from preprints. When a prediction turns out wrong, say
so plainly in the CHANGELOG rather than quietly restating it.
