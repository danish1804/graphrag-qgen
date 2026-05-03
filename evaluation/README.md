# evaluation/ : LLM-as-Judge Evaluation System

Evaluates the quality of generated questions, answers, and retrieved chunks using an LLM judge. Runs automatically after every generation call and saves evaluation records to disk.

**Author:** Aiswarya Sudhir
**Integrated by:** Mohammed Danish Alam

---

## Overview

After the main pipeline generates a question and sample answer, the evaluation module:

1. Sends retrieved chunks to the judge for **retrieval quality** assessment
2. Sends the generated question for **question quality** assessment
3. Sends the generated answer for **answer quality** assessment
4. If question score is below threshold -> **regenerates** and re-evaluates
5. Saves the full evaluation record as JSON to `outputs/evaluations/`

The judge model is loaded **once at server startup** and reused for every evaluation : no reloading per request.

---

## Files

```
evaluation/
├── __init__.py
├── evaluator.py    # Orchestrator : runs all three rubric evaluations
├── rubrics.py      # Prompts and scoring rubrics for the judge
└── storage.py      # JSON output parser and evaluation record saver
```

---

## evaluator.py : Evaluation Orchestrator

### evaluate_generation(...)

Main public function. Takes already-generated outputs and runs all three LLM evaluations.

```python
from evaluation.evaluator import evaluate_generation

record = evaluate_generation(
    learning_objective = "Explain CNN parameter sharing",
    retrieved_chunks   = [
        "[SEED from source_62.pdf]\nFeature extraction...",
        "[NEIGHBOUR from source_50.pdf]\nApplied across..."
    ],
    question           = "Explain how parameter sharing...",
    answer             = "Parameter sharing in CNNs...",
    difficulty         = "medium",
    question_type      = "conceptual",
    judge_model        = model,       # pre-loaded judge
    judge_tokenizer    = tokenizer,   # pre-loaded tokenizer
    save               = True         # save JSON to disk
)
```

**Integration contract:**
- Does NOT generate anything
- Does NOT touch FAISS or the knowledge graph
- Accepts already-generated outputs as input
- Caller is responsible for loading judge model

**Warm start vs standalone:**
```python
# Warm start (server mode : judge pre-loaded):
evaluate_generation(..., judge_model=model, judge_tokenizer=tok)
# -> reuses provided model, does NOT unload after

# Standalone mode (no pre-loaded model):
evaluate_generation(..., judge_model=None, judge_tokenizer=None)
# -> loads judge internally, unloads after evaluation
```

### Returns

```python
{
    "eval_id":   "eval_2026-05-03_041749_411a5e",
    "timestamp": "2026-05-03T04:17:49",
    "generator_model": "Qwen/Qwen2.5-7B-Instruct",
    "judge_model":     "Qwen/Qwen2.5-3B-Instruct",
    "input": {
        "learning_objective": "Explain CNN parameter sharing",
        "difficulty":         "medium",
        "question_type":      "conceptual"
    },
    "retrieval": {
        "chunks":       ["[SEED from source_62.pdf]\n..."],
        "scores":       {"relevance": 3, "sufficiency": 3, "noise": 5},
        "overall":      3.67,
        "reasoning":    "The chunks directly address...",
        "parse_status": "ok"
    },
    "question": {
        "generated":    "Explain how parameter sharing...",
        "scores":       {"faithfulness": 5, "clarity": 5, ...},
        "overall":      5.0,
        "reasoning":    "The question is well-formed...",
        "parse_status": "ok"
    },
    "answer": {
        "generated":    "Parameter sharing in CNNs...",
        "scores":       {"correctness": 5, "completeness": 5, ...},
        "overall":      4.5,
        "reasoning":    "The answer correctly explains...",
        "parse_status": "ok"
    }
}
```

### Standalone Test

```bash
python evaluation/evaluator.py
# Runs with hardcoded CNN parameter sharing example
# Loads judge, runs all 3 rubrics, saves JSON, unloads judge
```

---

## rubrics.py : Judge Prompts and Scoring Criteria

Defines three rubrics using the **reason-then-score** pattern: judge writes reasoning BEFORE emitting JSON scores. This reduces score inflation and produces more reliable evaluations (Zheng et al., 2023).

### Rubric 1 : Retrieval Quality

**Evaluates:** Are the retrieved chunks relevant and sufficient?

| Criterion | Description | Score 1 | Score 5 |
|-----------|-------------|---------|---------|
| RELEVANCE | How directly chunks relate to learning objective | Unrelated | All directly relevant |
| SUFFICIENCY | Enough information to generate a good question | Key concepts missing | All concepts present |
| NOISE | Free from irrelevant/distracting content | Mostly noise | Almost no irrelevant content |

**Score keys:** `["relevance", "sufficiency", "noise"]`

### Rubric 2 : Question Quality

**Evaluates:** Is the generated question well-formed?

| Criterion | Description | Score 1 | Score 5 |
|-----------|-------------|---------|---------|
| FAITHFULNESS | Answerable from context alone | Requires outside knowledge | Fully context-grounded |
| CLARITY | Clear and unambiguous | Confusing | Crisp and precise |
| DIFFICULTY_ALIGNMENT | Matches requested difficulty | Severely mismatched | Perfectly calibrated |
| TYPE_ALIGNMENT | Matches requested question type | Wrong type | Clearly matches |
| STANDALONE_QUALITY | No references to "the context" | Explicitly references passage | Genuine exam question |

**Score keys:** `["faithfulness", "clarity", "difficulty_alignment", "type_alignment", "standalone_quality"]`

**Difficulty guide used by judge:**
```
easy   = recall/basic understanding
medium = explanation/conceptual
hard   = application/analysis
```

**Type guide:**
```
recall      = define/list/state
conceptual  = explain/compare
application = design/analyse/apply
```

### Rubric 3 : Answer Quality

**Evaluates:** Is the sample answer correct and well-written?

| Criterion | Description | Score 1 | Score 5 |
|-----------|-------------|---------|---------|
| CORRECTNESS | Factually correct per context | Clear errors | Fully correct |
| COMPLETENESS | Covers key points | Misses most points | Addresses all key points |
| GROUNDEDNESS | No hallucinations | Significant invented content | Entirely grounded |
| ACADEMIC_TONE | Appropriate for university | Casual/bullet points | Precise academic prose |

**Score keys:** `["correctness", "completeness", "groundedness", "academic_tone"]`

### Judge Output Format

The judge is prompted to respond in this exact format:

```
REASONING:
<step-by-step analysis of each criterion>

SCORES:
{"criterion_1": 4, "criterion_2": 5, ...}
```

### Helper Functions

```python
from evaluation.rubrics import (
    build_retrieval_prompt,
    build_question_prompt,
    build_answer_prompt,
    format_chunks_for_prompt
)

# Format chunks for prompt display:
formatted = format_chunks_for_prompt([
    "[SEED from source_62.pdf]\nFeature extraction...",
    "[NEIGHBOUR from source_50.pdf]\nApplied across..."
])
# Output:
# [Chunk 1]
# [SEED from source_62.pdf]
# Feature extraction...
#
# [Chunk 2]
# [NEIGHBOUR from source_50.pdf]
# Applied across...
```

---

## storage.py : Parser and Storage

### parse_judge_output(raw_output, expected_keys) -> dict

Robustly parses the judge's raw text response into structured scores.

**Handles:**
- Missing `REASONING:` label -> extracts text before JSON block
- JSON wrapped in markdown fences (` ```json ... ``` `)
- Missing score keys -> marks as `partial`, fills with `None`
- Trailing commas in JSON -> cleanup pass
- Invalid score values -> coerced to int, clamped to 1-5

```python
from evaluation.storage import parse_judge_output

result = parse_judge_output(
    raw_output    = "REASONING:\nChunks are relevant...\n\nSCORES:\n{\"relevance\": 4}",
    expected_keys = ["relevance", "sufficiency", "noise"]
)

# Returns:
{
    "reasoning":    "Chunks are relevant...",
    "scores":       {"relevance": 4, "sufficiency": None, "noise": None},
    "overall":      4.0,
    "parse_status": "partial",  # missing keys
    "parse_notes":  "Missing key: sufficiency. Missing key: noise.",
    "raw_output":   "REASONING:\n..."
}
```

**Parse status values:**
```
"ok"       -> all keys found and valid
"partial"  -> some keys missing or invalid
"failed"   -> no JSON object found in output
```

### save_evaluation(record, eval_id) -> str

Saves evaluation record as JSON to `outputs/evaluations/`.

```python
from evaluation.storage import save_evaluation, make_eval_id

eval_id = make_eval_id()
# -> "eval_2026-05-03_041749_411a5e"

path = save_evaluation(record, eval_id)
# -> "outputs/evaluations/eval_2026-05-03_041749_411a5e.json"
```

### load_evaluation(eval_id) -> dict

Loads a saved evaluation by ID.

```python
from evaluation.storage import load_evaluation

record = load_evaluation("eval_2026-05-03_041749_411a5e")
print(record["question"]["overall"])  # 5.0
```

### list_evaluations() -> List[str]

Returns all saved evaluation IDs, most recent first.

```python
from evaluation.storage import list_evaluations

ids = list_evaluations()
# ["eval_2026-05-03_...", "eval_2026-05-02_...", ...]
```

### Test

```bash
python evaluation/storage.py
# Runs 4 tests:
# Test 1: clean judge output -> ok
# Test 2: markdown-fenced JSON -> ok
# Test 3: missing key -> partial
# Test 4: save/load roundtrip -> ok
```

---

## Regeneration Loop

If the question score is below `score_threshold` (default 3.5), the pipeline regenerates the question and re-evaluates, up to `max_regenerations` times:

```
Generate question
    ↓
Evaluate (judge)
    ↓
question_overall >= 3.5?
    YES -> return response
    NO  -> regenerate question
          ↓
          Evaluate again (attempt 2)
          ↓
          question_overall >= 3.5?
              YES -> return response
              NO  -> regenerate (attempt 3, final)
                    ↓
                    Return regardless of score
```

**Config:**
```yaml
judge:
  score_threshold:   3.5   # below -> regenerate
  max_regenerations: 2     # max retries (attempts = max + 1)
```

**In API response:**
```json
"judge_scores": {
    "question_overall":  4.8,
    "answer_overall":    4.5,
    "retrieval_overall": 3.2,
    "attempts":          1,
    "eval_id":           "eval_2026-05-03_..."
}
```

`attempts: 1` means question passed on first try.
`attempts: 2` means one regeneration was needed.

---

## Evaluation Output Directory

All evaluation records saved to:
```
outputs/evaluations/
└── eval_2026-05-03_041749_411a5e.json
└── eval_2026-05-03_053432_6c85a5.json
└── ...
```

Directory created automatically if it does not exist. Path configurable in `config.yml`:

```yaml
judge:
  eval_output_dir: "outputs/evaluations"
```

---

## Known Limitations

| Issue | Cause | Status |
|-------|-------|--------|
| Retrieval scores consistently low (~1.67) | OCR artifacts in PDF chunks look noisy to judge | Known : PDF extraction quality issue |
| Self-preference bias | Judge is same model as generator (Qwen 7B reused) | Acceptable for demo : dedicated judge planned |
| No evaluation in multi-domain fallback | generate_with_multi_domain() now includes judge | Fixed |

---

## Related

- [../utils/judge.py](../utils/README.md) : judge model loader
- [../graphrag/README.md](../graphrag/README.md) : where evaluate_generation() is called
- [../README.md](../README.md) : Main project README
