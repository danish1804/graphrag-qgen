"""
generate_teacher_data.py
========================
Uses Claude Haiku as a teacher model to generate high-quality question
training samples from existing GraphRAG eval contexts.

Strategy:
  - Reads all eval_*.json files from outputs/evaluations/
  - For each high-quality record (question score >= min_score):
      * Sends the SAME GraphRAG context + LO to Claude Haiku
      * Generates `n_variants` different questions per context
  - Saves in identical JSONL format as graphrag_question_train.jsonl
  - Merges with or replaces existing training data (see --mode flag)

Why this works:
  - Claude Haiku is a stronger teacher than Qwen 2.5 7B
  - Same prompt format = fine-tuned model learns the right input->output mapping
  - n_variants > 1 multiplies your dataset size without new API calls to your pipeline

Usage:
  # Get API key: https://console.anthropic.com/ → top up $5 minimum
  export ANTHROPIC_API_KEY="sk-ant-..."

  # Dry run — see what would be generated
  python finetune_data/generate_teacher_data.py --dry-run

  # Generate 2 variants per context, merge with existing data (recommended first run)
  python finetune_data/generate_teacher_data.py --n-variants 2 --mode merge

  # Generate 3 variants, replace existing data
  python finetune_data/generate_teacher_data.py --n-variants 3 --mode replace

  # Only generate for a specific difficulty
  python finetune_data/generate_teacher_data.py --difficulty hard --n-variants 2

Repro:
  Python   : 3.10+
  anthropic: 0.25.0
  Seed     : 42
  Cost     : ~$0.001 per question; 268 evals × 2 variants ≈ $0.54 USD
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import anthropic
except ImportError:
    print("anthropic not installed. Run: pip install anthropic==0.25.0")
    sys.exit(1)

from utils.prompts import QUESTION_SYSTEM_PROMPT, build_question_messages  # noqa: E402

# ─── CONFIG ──────────────────────────────────────────────────────────────────
EVAL_DIR        = _ROOT / "outputs" / "evaluations"
DATA_DIR        = Path(__file__).resolve().parent
TRAIN_OUT       = DATA_DIR / "graphrag_question_train.jsonl"
EVAL_OUT        = DATA_DIR / "graphrag_question_eval.jsonl"
TEACHER_CACHE   = DATA_DIR / "teacher_cache.jsonl"   # resume on crash

DEFAULT_MIN_SCORE  = 4.0
DEFAULT_N_VARIANTS = 2
DEFAULT_EVAL_RATIO = 0.05
SLEEP_BETWEEN      = 0.3    # seconds between API calls
SEED               = 42
MCQ_PATTERN        = re.compile(r"\b[A-D]\)\s", re.IGNORECASE)
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_eval_rows(min_score: float, difficulty_filter: str | None) -> list[dict]:
    rows = []
    for path in sorted(EVAL_DIR.glob("eval_*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        q_block = rec.get("question") or {}
        score = q_block.get("overall")
        if score is None or score < min_score:
            continue

        inp = rec.get("input") or {}
        difficulty = inp.get("difficulty", "medium")
        if difficulty_filter and difficulty != difficulty_filter:
            continue

        chunks = (rec.get("retrieval") or {}).get("chunks") or []
        if not chunks:
            continue

        # Build context string (same logic as build_graphrag_question_dataset.py)
        context_parts = []
        for chunk in chunks:
            text = chunk.strip()
            if "\n" in text:
                _, rest = text.split("\n", 1)
                context_parts.append(rest.strip())
            else:
                context_parts.append(text)
        context = "\n\n".join(context_parts).strip()

        rows.append({
            "eval_id"           : rec.get("eval_id", path.stem),
            "learning_objective": inp.get("learning_objective", "").strip(),
            "difficulty"        : difficulty,
            "question_type"     : inp.get("question_type", "conceptual"),
            "context"           : context,
            "original_question" : (q_block.get("generated") or "").strip(),
            "question_score"    : float(score),
        })
    return rows


def load_teacher_cache() -> dict[str, list[str]]:
    """Load previously generated questions (keyed by eval_id+variant_idx)."""
    cache = {}
    if TEACHER_CACHE.exists():
        for line in TEACHER_CACHE.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                cache[entry["cache_key"]] = entry["question"]
            except Exception:
                pass
    return cache


def save_to_cache(cache_key: str, question: str):
    with open(TEACHER_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cache_key": cache_key, "question": question}) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# Claude Haiku call
# ════════════════════════════════════════════════════════════════════════════

TEMPERATURE_BY_VARIANT = [0.3, 0.7, 1.0]   # variant 0 = precise, 1/2 = creative


def generate_question_with_claude(
    client: anthropic.Anthropic,
    context: str,
    learning_objective: str,
    difficulty: str,
    question_type: str,
    variant_idx: int = 0,
) -> str | None:
    """
    Call Claude Haiku with the EXACT same prompt format as the production pipeline.
    Temperature varies by variant to encourage diversity.
    """
    messages = build_question_messages(
        context=context,
        learning_objective=learning_objective,
        difficulty=difficulty,
        question_type=question_type,
    )
    # messages[0] is system, messages[1] is user
    temperature = TEMPERATURE_BY_VARIANT[min(variant_idx, len(TEMPERATURE_BY_VARIANT) - 1)]

    try:
        # Case study: SCENARIO (3-6 sentences) + QUESTION line needs more tokens
        max_tokens = 380 if question_type == "case_study" else 200
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=messages[0]["content"],
            messages=[{"role": "user", "content": messages[1]["content"]}],
            temperature=temperature,
        )
        question = response.content[0].text.strip()

        # Reject MCQs
        if MCQ_PATTERN.search(question):
            return None
        # Reject empty or too short
        if len(question) < 20:
            return None
        # Case study must have both SCENARIO: and QUESTION: blocks
        if question_type == "case_study":
            if "SCENARIO" not in question.upper() or "QUESTION" not in question.upper():
                return None

        return question

    except anthropic.APIError as e:
        print(f"    [API error] {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# JSONL output
# ════════════════════════════════════════════════════════════════════════════

def to_training_record(
    row: dict,
    question: str,
    source: str = "claude-haiku",
) -> dict:
    """Build a training record in the same format as build_graphrag_question_dataset.py."""
    messages = build_question_messages(
        context=row["context"],
        learning_objective=row["learning_objective"],
        difficulty=row["difficulty"],
        question_type=row["question_type"],
    )
    return {
        "messages": messages + [{"role": "assistant", "content": question}],
        "metadata": {
            "eval_id"        : row["eval_id"],
            "difficulty"     : row["difficulty"],
            "question_type"  : row["question_type"],
            "question_score" : row["question_score"],
            "source"         : source,
        },
    }


def split_by_source(
    records: list[dict],
    eval_ratio: float,
) -> tuple[list[dict], list[dict]]:
    """80/20 split grouped by eval_id source to avoid leakage."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_id[rec["metadata"]["eval_id"]].append(rec)

    rng = random.Random(SEED)
    train, holdout = [], []
    for group in by_id.values():
        rng.shuffle(group)
        n_eval = max(1, int(len(group) * eval_ratio)) if len(group) > 1 else 0
        holdout.extend(group[:n_eval])
        train.extend(group[n_eval:])
    return train, holdout


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_existing_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate teacher training data using Claude Haiku"
    )
    parser.add_argument("--min-score",   type=float, default=DEFAULT_MIN_SCORE,
                        help="Minimum judge score to include source eval (default: 4.0)")
    parser.add_argument("--n-variants",  type=int,   default=DEFAULT_N_VARIANTS,
                        help="Questions to generate per context (default: 2)")
    parser.add_argument("--difficulty",  type=str,   default=None,
                        help="Filter to one difficulty: easy | medium | hard")
    parser.add_argument("--mode",        type=str,   default="merge",
                        choices=["merge", "replace"],
                        help="merge = add to existing JSONL; replace = overwrite (default: merge)")
    parser.add_argument("--eval-ratio",  type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print what would be generated without calling Claude")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print(
            "ANTHROPIC_API_KEY not set.\n"
            "Get it at: https://console.anthropic.com/ → API Keys\n"
            "Set it:    export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "Estimated cost: ~$0.001/question; 268 × 2 variants ≈ $0.54 USD"
        )
        return 1

    rows = load_eval_rows(args.min_score, args.difficulty)
    print(f"Source eval rows (score >= {args.min_score}): {len(rows)}")
    print(f"Variants per context                        : {args.n_variants}")
    print(f"Expected new questions                      : {len(rows) * args.n_variants}")
    print(f"Estimated cost                              : ~${len(rows) * args.n_variants * 0.001:.2f} USD")
    print(f"Output mode                                 : {args.mode}")

    if args.dry_run:
        print("\n[DRY RUN] First 5 rows:")
        for row in rows[:5]:
            print(f"  [{row['difficulty']:6}] [{row['question_type']:12}] {row['learning_objective'][:60]}")
        return 0

    client = anthropic.Anthropic(api_key=api_key)
    cache = load_teacher_cache()
    print(f"Resuming from cache: {len(cache)} questions already generated\n")

    all_records: list[dict] = []
    failed = 0

    for i, row in enumerate(rows, 1):
        for v in range(args.n_variants):
            cache_key = f"{row['eval_id']}__v{v}"

            if cache_key in cache:
                question = cache[cache_key]
                rec = to_training_record(row, question, source=f"claude-haiku-v{v}")
                all_records.append(rec)
                continue

            question = generate_question_with_claude(
                client,
                context=row["context"],
                learning_objective=row["learning_objective"],
                difficulty=row["difficulty"],
                question_type=row["question_type"],
                variant_idx=v,
            )

            if question:
                save_to_cache(cache_key, question)
                rec = to_training_record(row, question, source=f"claude-haiku-v{v}")
                all_records.append(rec)
            else:
                failed += 1

            time.sleep(SLEEP_BETWEEN)

        if i % 20 == 0:
            print(f"  [{i:>3}/{len(rows)}] generated so far: {len(all_records)}  failed: {failed}")

    print(f"\nGenerated {len(all_records)} records ({failed} failed)")

    # Merge with existing if requested
    if args.mode == "merge":
        existing_train = load_existing_jsonl(TRAIN_OUT)
        existing_eval  = load_existing_jsonl(EVAL_OUT)
        print(f"Merging with existing: {len(existing_train)} train + {len(existing_eval)} eval")
        all_records = existing_train + existing_eval + all_records
        # Deduplicate by (eval_id + assistant content)
        seen = set()
        deduped = []
        for rec in all_records:
            key = (
                rec.get("metadata", {}).get("eval_id", ""),
                rec["messages"][-1]["content"][:100],
            )
            if key not in seen:
                seen.add(key)
                deduped.append(rec)
        all_records = deduped
        print(f"After dedup: {len(all_records)} total records")

    # Split and write
    train_records, eval_records = split_by_source(all_records, args.eval_ratio)
    write_jsonl(TRAIN_OUT, train_records)
    write_jsonl(EVAL_OUT,  eval_records)

    print(f"\n{'='*55}")
    print(f"Done.")
    print(f"  Train : {len(train_records)} rows → {TRAIN_OUT}")
    print(f"  Eval  : {len(eval_records)} rows  → {EVAL_OUT}")
    print(f"  Cache : {TEACHER_CACHE} (safe to delete after training)")
    print(f"{'='*55}")
    print("\nNext: python finetune_data/train_question_lora.py --max-seq-len 1024 --epochs 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
