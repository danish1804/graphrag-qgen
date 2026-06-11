# outputs/ — Generated artifacts (local)

Runtime output from the LLM-as-judge pipeline. **Do not commit** evaluation JSON files to git — they are large and environment-specific.

---

## Structure

```
outputs/
└── evaluations/     # One JSON file per generation/judge run
    └── eval_YYYY-MM-DD_HHMMSS_<id>.json
```

---

## Evaluation records

Each file is written by `evaluation/storage.py` when `evaluate_generation(..., save=True)`.

Typical contents:

| Section | Description |
|---------|-------------|
| `input` | learning_objective, difficulty, question_type |
| `retrieval` | relevance, sufficiency, noise scores |
| `question` | faithfulness, clarity, type_alignment, difficulty_alignment |
| `answer` | correctness, completeness, groundedness |
| `prompt_version` / `rubric_version` | Traceability (when enabled) |

Path configured in `config.yml` → `judge.eval_output_dir` (default `outputs/evaluations`).

---

## Related

- [../evaluation/README.md](../evaluation/README.md) — Judge system and rubrics
- [../graphrag/README.md](../graphrag/README.md) — When judge runs (GraphRAG path only)
