"""
Build GraphRAG-aligned question fine-tuning JSONL from judge eval records.

Each row matches production inference: build_question_messages() -> generated question.

Usage (from repo root):
  python finetune_data/build_graphrag_question_dataset.py
  python finetune_data/build_graphrag_question_dataset.py --refresh-context
  python finetune_data/build_graphrag_question_dataset.py --min-score 4.0 --eval-dir outputs/evaluations
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.prompts import build_question_messages  # noqa: E402

DEFAULT_EVAL_DIR = _ROOT / "outputs" / "evaluations"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent
MCQ_PATTERN = re.compile(r"\b[A-D]\)\s", re.IGNORECASE)
SOURCE_PATTERN = re.compile(r"from\s+(source_\d+\.pdf)", re.IGNORECASE)


def _extract_sources(chunks: list[str]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        match = SOURCE_PATTERN.search(chunk)
        if match:
            sources.append(match.group(1))
    return sources or ["unknown"]


def _chunks_to_context(chunks: list[str]) -> str:
    """Approximate retrieval context from judge chunk strings."""
    body_lines: list[str] = []
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        if "\n" in text:
            _, rest = text.split("\n", 1)
            body_lines.append(rest.strip())
        else:
            body_lines.append(text)
    return "\n\n".join(body_lines).strip()


def _is_mcq(question: str) -> bool:
    return bool(MCQ_PATTERN.search(question))


def _load_eval_rows(eval_dir: Path, min_score: float) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(eval_dir.glob("eval_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        question_block = record.get("question") or {}
        score = question_block.get("overall")
        generated = (question_block.get("generated") or "").strip()
        if score is None or score < min_score or not generated:
            continue
        if _is_mcq(generated):
            continue

        inp = record.get("input") or {}
        chunks = (record.get("retrieval") or {}).get("chunks") or []
        if not chunks:
            continue

        rows.append(
            {
                "eval_id": record.get("eval_id", path.stem),
                "learning_objective": inp.get("learning_objective", "").strip(),
                "difficulty": inp.get("difficulty", "medium"),
                "question_type": inp.get("question_type", "conceptual"),
                "context": _chunks_to_context(chunks),
                "output": generated,
                "question_score": float(score),
                "sources": _extract_sources(chunks),
            }
        )
    return rows


def _refresh_context(row: dict) -> str | None:
    """Re-run GraphRAG retrieval for full context (slower, closer to production)."""
    from graphrag.graphrag_domain_indexer import route_by_embedding
    from graphrag.graphrag_retriever import retrieve_graph_context
    from utils.config import cfg

    objective = row["learning_objective"]
    difficulty = row["difficulty"]
    diff_cfg = cfg.get("difficulty", {}).get(difficulty, cfg["difficulty"]["medium"])
    max_chunks = int(diff_cfg.get("max_chunks", 5))
    top_k_nodes = int(diff_cfg.get("top_k_nodes", 3))
    neighbour_depth = int(diff_cfg.get("neighbour_depth", 1))
    seed_ratio = float(diff_cfg.get("seed_ratio", 0.6))

    route = route_by_embedding(objective, return_all_scores=False)
    domain = route.get("domain")
    if not domain:
        return None

    retrieval = retrieve_graph_context(
        query=objective,
        domain=domain,
        top_k_nodes=top_k_nodes,
        neighbour_depth=neighbour_depth,
        max_context_chunks=max_chunks,
        seed_ratio=seed_ratio,
    )
    context = (retrieval.get("context") or "").strip()
    return context or None


def _to_training_record(row: dict) -> dict:
    messages = build_question_messages(
        context=row["context"],
        learning_objective=row["learning_objective"],
        difficulty=row["difficulty"],
        question_type=row["question_type"],
    )
    return {
        "messages": messages + [{"role": "assistant", "content": row["output"]}],
        "metadata": {
            "eval_id": row["eval_id"],
            "domain": row.get("domain"),
            "difficulty": row["difficulty"],
            "question_type": row["question_type"],
            "question_score": row["question_score"],
            "sources": row["sources"],
        },
    }


def _split_by_source(rows: list[dict], eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    import random

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row["sources"][0]
        by_source[key].append(row)

    rng = random.Random(seed)
    train: list[dict] = []
    holdout: list[dict] = []
    for source_rows in by_source.values():
        rng.shuffle(source_rows)
        n_eval = max(1, int(len(source_rows) * eval_ratio)) if len(source_rows) > 2 else 1
        holdout.extend(source_rows[:n_eval])
        train.extend(source_rows[n_eval:])
    return train, holdout


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GraphRAG question fine-tune JSONL")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--refresh-context",
        action="store_true",
        help="Re-run GraphRAG retrieval for full context (slow; needs indexes loaded)",
    )
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        print(f"Eval dir not found: {args.eval_dir}")
        return 1

    rows = _load_eval_rows(args.eval_dir, args.min_score)
    print(f"Loaded {len(rows)} high-quality eval rows (min question score {args.min_score})")

    if not rows:
        print("No rows — generate more questions with the judge enabled first.")
        return 1

    if args.refresh_context:
        print("Refreshing context via GraphRAG retrieval (this may take a while)...")
        refreshed = 0
        for row in rows:
            context = _refresh_context(row)
            if context:
                row["context"] = context
                refreshed += 1
        print(f"Refreshed context for {refreshed}/{len(rows)} rows")

    train_rows, eval_rows = _split_by_source(rows, args.eval_ratio, args.seed)
    train_records = [_to_training_record(r) for r in train_rows]
    eval_records = [_to_training_record(r) for r in eval_rows]

    train_path = args.out_dir / "graphrag_question_train.jsonl"
    eval_path = args.out_dir / "graphrag_question_eval.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(eval_path, eval_records)

    print(f"Wrote train: {train_path} ({len(train_records)} rows)")
    print(f"Wrote eval:  {eval_path} ({len(eval_records)} rows)")
    print("Next: python finetune_data/train_question_lora.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
