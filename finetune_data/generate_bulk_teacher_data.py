"""
generate_bulk_teacher_data.py
==============================
Generates ~10,000 training samples for Kaggle fine-tuning.

Strategy (three passes, combined to hit target):

  Pass 1 — Combo expansion (fast, ~3,200 samples)
    For each of the 268 high-quality eval contexts, generate questions
    for ALL type × difficulty combos that are NOT the original combo.
    This balances the dataset (original data is 75% conceptual/medium).

  Pass 2 — Synthetic context mixing (~4,000 samples)
    Collect all unique chunks from eval JSONs. Group by domain.
    Sample random subsets of 4–6 same-domain chunks to create new
    contexts. For each synthetic context Claude generates (LO, question).
    This diversifies beyond the 268 original contexts.

  Pass 3 — New learning objectives (~3,000 samples, optional)
    Claude generates 300 new domain-specific LOs. Pair each with
    existing high-scoring contexts from the same domain. Generates
    one question per (new LO, existing context) pair.

Total: ~10,000 samples with balanced type/difficulty distribution.

Usage:
  # Dry run — show plan
  python finetune_data/generate_bulk_teacher_data.py --dry-run

  # Full run (recommended: run overnight, resumes on crash)
  python finetune_data/generate_bulk_teacher_data.py --target 10000

  # Run only Pass 1 + 2 (no new LOs)
  python finetune_data/generate_bulk_teacher_data.py --passes 1,2 --target 7000

  # Specify output files (default: overwrites train/eval JSONL)
  python finetune_data/generate_bulk_teacher_data.py --out-prefix bulk

Repro:
  Python   : 3.10+
  anthropic: 0.25.0
  Cost     : ~$0.001/question → 10K ≈ $10 USD
  Seed     : 42
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
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import anthropic
except ImportError:
    print("Run: pip install anthropic==0.25.0")
    sys.exit(1)

from utils.prompts import QUESTION_SYSTEM_PROMPT, build_question_messages  # noqa: E402

# ─── CONFIG ──────────────────────────────────────────────────────────────────
EVAL_DIR    = _ROOT / "outputs" / "evaluations"
DATA_DIR    = Path(__file__).resolve().parent
CACHE_FILE  = DATA_DIR / "bulk_cache.jsonl"

QUESTION_TYPES  = ["recall", "conceptual", "application", "case_study"]
DIFFICULTIES    = ["easy", "medium", "hard"]
DOMAINS         = ["deep_learning", "database", "software_engineering",
                   "operating_systems", "computer_architecture"]

MCQ_PATTERN = re.compile(r"\b[A-D]\)\s", re.IGNORECASE)
SEED        = 42
SLEEP       = 0.25   # seconds between API calls
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _chunks_to_context(chunks: list[str]) -> str:
    parts = []
    for chunk in chunks:
        text = chunk.strip()
        if "\n" in text:
            _, rest = text.split("\n", 1)
            parts.append(rest.strip())
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


CASE_STUDY_SCENARIO = re.compile(r"SCENARIO\s*:", re.IGNORECASE)
CASE_STUDY_QUESTION = re.compile(r"QUESTION\s*:", re.IGNORECASE)


def _is_valid(question: str, qtype: str = "") -> bool:
    if not question or len(question) < 20:
        return False
    if MCQ_PATTERN.search(question):
        return False
    # Case study must have both SCENARIO: and QUESTION: blocks
    if qtype == "case_study":
        if not CASE_STUDY_SCENARIO.search(question):
            return False
        if not CASE_STUDY_QUESTION.search(question):
            return False
    return True
    return True


def load_eval_rows(min_score: float = 4.0) -> list[dict]:
    rows = []
    for path in sorted(EVAL_DIR.glob("eval_*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        q_block = rec.get("question") or {}
        score   = q_block.get("overall")
        if score is None or score < min_score:
            continue
        inp    = rec.get("input") or {}
        chunks = (rec.get("retrieval") or {}).get("chunks") or []
        if not chunks:
            continue
        rows.append({
            "eval_id"           : rec.get("eval_id", path.stem),
            "learning_objective": inp.get("learning_objective", "").strip(),
            "difficulty"        : inp.get("difficulty", "medium"),
            "question_type"     : inp.get("question_type", "conceptual"),
            "context"           : _chunks_to_context(chunks),
            "raw_chunks"        : chunks,
            "question_score"    : float(score),
        })
    return rows


def load_cache() -> dict[str, str]:
    cache = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
                cache[e["key"]] = e["question"]
            except Exception:
                pass
    return cache


def append_cache(key: str, question: str):
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "question": question}) + "\n")


def to_record(context: str, lo: str, difficulty: str, qtype: str,
              question: str, meta: dict) -> dict:
    msgs = build_question_messages(
        context=context,
        learning_objective=lo,
        difficulty=difficulty,
        question_type=qtype,
    )
    return {
        "messages": msgs + [{"role": "assistant", "content": question}],
        "metadata": {**meta, "difficulty": difficulty, "question_type": qtype},
    }


def call_claude(client, context: str, lo: str, difficulty: str,
                qtype: str, temperature: float = 0.5) -> str | None:
    msgs = build_question_messages(
        context=context, learning_objective=lo,
        difficulty=difficulty, question_type=qtype,
    )
    # Case study needs more tokens: SCENARIO (3-6 sentences) + QUESTION line
    max_tokens = 380 if qtype == "case_study" else 220
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=msgs[0]["content"],
            messages=[{"role": "user", "content": msgs[1]["content"]}],
            temperature=temperature,
        )
        q = resp.content[0].text.strip()
        return q if _is_valid(q, qtype) else None
    except Exception as e:
        print(f"    [API] {e}")
        return None


def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_eval(records: list[dict], eval_ratio: float = 0.05,
                     seed: int = SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_id: dict[str, list] = defaultdict(list)
    for r in records:
        by_id[r["metadata"].get("eval_id", "unknown")].append(r)
    train, val = [], []
    for group in by_id.values():
        rng.shuffle(group)
        n = max(1, int(len(group) * eval_ratio)) if len(group) > 2 else 0
        val.extend(group[:n])
        train.extend(group[n:])
    return train, val


# ════════════════════════════════════════════════════════════════════════════
# Pass 1 — Combo expansion
# ════════════════════════════════════════════════════════════════════════════

def pass1_combo_expansion(
    client, rows: list[dict], cache: dict, target: int
) -> list[dict]:
    """
    For each eval context, generate questions for type×difficulty combos
    the original data doesn't have. Prioritises underrepresented combos.
    """
    print(f"\n[Pass 1] Combo expansion — target: {target} records")

    # Count existing distribution to prioritise gaps
    combo_counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        combo_counts[(r["question_type"], r["difficulty"])] += 1

    # Build priority list: rarest combos first
    all_combos = list(product(QUESTION_TYPES, DIFFICULTIES))
    all_combos.sort(key=lambda c: combo_counts.get(c, 0))
    print(f"  Combo distribution (existing): {dict(combo_counts)}")

    records: list[dict] = []
    rng = random.Random(SEED)

    # Cycle through contexts + combos until target
    contexts_cycle = rows * 20   # allow revisiting contexts
    rng.shuffle(contexts_cycle)

    i = 0
    for row in contexts_cycle:
        if len(records) >= target:
            break
        orig_combo = (row["question_type"], row["difficulty"])
        # Pick a combo different from the original (prioritise rare ones)
        for combo in all_combos:
            if combo == orig_combo:
                continue
            qtype, diff = combo
            cache_key = f"p1__{row['eval_id']}__{qtype}__{diff}"
            if cache_key in cache:
                q = cache[cache_key]
            else:
                temp = 0.4 if diff == "easy" else 0.6 if diff == "medium" else 0.8
                q = call_claude(client, row["context"], row["learning_objective"],
                                diff, qtype, temperature=temp)
                time.sleep(SLEEP)
                if q:
                    append_cache(cache_key, q)
            if q:
                rec = to_record(
                    row["context"], row["learning_objective"], diff, qtype, q,
                    {"eval_id": row["eval_id"], "pass": 1, "source": "claude-haiku"}
                )
                records.append(rec)
                combo_counts[combo] += 1
            if len(records) >= target:
                break
        i += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(contexts_cycle)}] Pass 1 records so far: {len(records)}")

    print(f"  [Pass 1] Generated {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# Pass 2 — Synthetic context mixing
# ════════════════════════════════════════════════════════════════════════════

def _infer_domain(chunks: list[str]) -> str | None:
    """Infer domain from chunk text using keyword matching."""
    combined = " ".join(chunks).lower()
    scores = {
        "deep_learning"       : sum(combined.count(k) for k in ["neural", "cnn", "backprop", "gradient", "epoch", "layer", "weight", "convolution"]),
        "database"            : sum(combined.count(k) for k in ["sql", "join", "table", "query", "transaction", "index", "foreign key", "schema"]),
        "software_engineering": sum(combined.count(k) for k in ["agile", "scrum", "uml", "design pattern", "testing", "sprint", "requirements", "software"]),
        "operating_systems"   : sum(combined.count(k) for k in ["process", "thread", "scheduler", "memory", "deadlock", "kernel", "semaphore", "cpu"]),
        "computer_architecture": sum(combined.count(k) for k in ["cache", "pipeline", "register", "instruction", "cpu", "memory hierarchy", "alu", "clock"]),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def pass2_synthetic_contexts(
    client, rows: list[dict], cache: dict, target: int
) -> list[dict]:
    """
    Collect all unique chunks. Group by inferred domain.
    Sample random subsets of 4–6 same-domain chunks.
    Claude generates (LO, question) for each synthetic context.
    """
    print(f"\n[Pass 2] Synthetic context mixing — target: {target} records")

    # Collect all chunks grouped by domain
    domain_chunks: dict[str, list[str]] = defaultdict(list)
    seen_chunks: set[str] = set()
    for row in rows:
        domain = _infer_domain(row["raw_chunks"])
        if not domain:
            continue
        for chunk in row["raw_chunks"]:
            key = chunk[:80]
            if key not in seen_chunks:
                seen_chunks.add(key)
                domain_chunks[domain].append(chunk)

    print(f"  Unique chunks per domain: { {d: len(v) for d, v in domain_chunks.items()} }")

    records: list[dict] = []
    rng = random.Random(SEED + 1)
    sample_idx = 0

    # How many synthetic contexts to create
    n_contexts = min(target, 2000)

    LO_GEN_PROMPT = (
        "You are an IT education expert. Given these course material excerpts, "
        "write ONE concise learning objective (max 12 words) that could be assessed "
        "using this material. Return ONLY the learning objective, nothing else."
    )

    for _ in range(n_contexts * 3):   # 3x attempts to account for failures
        if len(records) >= target:
            break

        # Pick a domain with enough chunks
        valid_domains = [d for d, v in domain_chunks.items() if len(v) >= 4]
        if not valid_domains:
            break
        domain = rng.choice(valid_domains)

        # Sample 4–6 chunks
        n_chunks = rng.randint(4, min(6, len(domain_chunks[domain])))
        sampled = rng.sample(domain_chunks[domain], n_chunks)

        # First 2–3 as SEED, rest as NEIGHBOUR (mimic real GraphRAG format)
        n_seed = rng.randint(2, 3)
        formatted_chunks = (
            [f"[SEED from {domain}]\n{c}" for c in sampled[:n_seed]] +
            [f"[NEIGHBOUR from {domain}]\n{c}" for c in sampled[n_seed:]]
        )
        context = _chunks_to_context(formatted_chunks)
        context_key = context[:60]

        # Generate LO with Claude (cached by context fingerprint)
        lo_key = f"p2_lo__{hash(context_key) % 999999}"
        if lo_key in cache:
            lo = cache[lo_key]
        else:
            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=40,
                    system=LO_GEN_PROMPT,
                    messages=[{"role": "user", "content": f"Excerpts:\n{context[:800]}"}],
                    temperature=0.6,
                )
                lo = resp.content[0].text.strip()
                time.sleep(SLEEP)
                if lo:
                    append_cache(lo_key, lo)
            except Exception:
                lo = None

        if not lo:
            continue

        # Generate 2 questions for this context (different type/difficulty)
        for qtype, diff in rng.sample(list(product(QUESTION_TYPES, DIFFICULTIES)), 2):
            q_key = f"p2_q__{lo_key}__{qtype}__{diff}"
            if q_key in cache:
                q = cache[q_key]
            else:
                q = call_claude(client, context, lo, diff, qtype, temperature=0.6)
                time.sleep(SLEEP)
                if q:
                    append_cache(q_key, q)

            if q:
                rec = to_record(
                    context, lo, diff, qtype, q,
                    {"eval_id": f"synthetic_{sample_idx:04d}", "pass": 2,
                     "domain": domain, "source": "claude-haiku-synthetic"}
                )
                records.append(rec)
                if len(records) >= target:
                    break

        sample_idx += 1
        if sample_idx % 50 == 0:
            print(f"  [{sample_idx} contexts] Pass 2 records so far: {len(records)}")

    print(f"  [Pass 2] Generated {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# Pass 3 — New learning objectives
# ════════════════════════════════════════════════════════════════════════════

def pass3_new_learning_objectives(
    client, rows: list[dict], cache: dict, target: int
) -> list[dict]:
    """
    Claude generates new LOs per domain. Pair with high-scoring
    existing contexts from that domain. One question per pair.
    """
    print(f"\n[Pass 3] New learning objectives — target: {target} records")

    LO_BATCH_PROMPT = """You are an IT university curriculum designer.
Generate {n} diverse learning objectives for the subject: {subject}

Rules:
- Each must be assessable in a written exam (not lab/practical)
- Mix Bloom's levels: recall, understand, apply, analyse
- Each max 15 words
- Return ONLY a numbered list, one per line: 1. ... 2. ... etc
- No duplicates, no sub-bullets"""

    DOMAIN_SUBJECTS = {
        "deep_learning"        : "Deep Learning (CNNs, backpropagation, training, architectures)",
        "database"             : "Database Systems (SQL, transactions, normalisation, indexing)",
        "software_engineering" : "Software Engineering (SDLC, design patterns, testing, agile)",
        "operating_systems"    : "Operating Systems (processes, memory management, scheduling)",
        "computer_architecture": "Computer Architecture (CPU, cache, pipelines, memory hierarchy)",
    }

    # Group existing high-scoring rows by domain
    domain_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        domain = _infer_domain(row["raw_chunks"])
        if domain:
            domain_rows[domain].append(row)

    records: list[dict] = []
    rng = random.Random(SEED + 2)
    los_per_domain = max(10, target // (len(DOMAIN_SUBJECTS) * 2))

    for domain, subject in DOMAIN_SUBJECTS.items():
        if len(records) >= target:
            break
        if domain not in domain_rows or not domain_rows[domain]:
            continue

        # Generate new LOs
        lo_cache_key = f"p3_los__{domain}__{los_per_domain}"
        if lo_cache_key in cache:
            new_los = cache[lo_cache_key].split("|||")
        else:
            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=600,
                    system="You are a university curriculum designer. Return ONLY a numbered list.",
                    messages=[{"role": "user", "content": LO_BATCH_PROMPT.format(
                        n=los_per_domain, subject=subject)}],
                    temperature=0.7,
                )
                raw_los = resp.content[0].text.strip()
                time.sleep(SLEEP)
                new_los = []
                for line in raw_los.splitlines():
                    line = re.sub(r"^\d+\.\s*", "", line).strip()
                    if len(line) > 10:
                        new_los.append(line)
                if new_los:
                    append_cache(lo_cache_key, "|||".join(new_los))
            except Exception:
                new_los = []

        print(f"  [{domain}] {len(new_los)} new LOs")

        for lo in new_los:
            if len(records) >= target:
                break
            # Pair with a random high-scoring context from same domain
            ctx_row = rng.choice(domain_rows[domain])
            qtype = rng.choice(QUESTION_TYPES)
            diff  = rng.choice(DIFFICULTIES)
            q_key = f"p3_q__{hash(lo[:30]) % 999999}__{qtype}__{diff}"

            if q_key in cache:
                q = cache[q_key]
            else:
                q = call_claude(client, ctx_row["context"], lo, diff, qtype, temperature=0.6)
                time.sleep(SLEEP)
                if q:
                    append_cache(q_key, q)

            if q:
                rec = to_record(
                    ctx_row["context"], lo, diff, qtype, q,
                    {"eval_id": ctx_row["eval_id"], "pass": 3,
                     "domain": domain, "source": "claude-haiku-newlo"}
                )
                records.append(rec)

    print(f"  [Pass 3] Generated {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ~10K teacher training samples for Kaggle fine-tuning"
    )
    parser.add_argument("--target",     type=int, default=10000,
                        help="Total training samples to generate (default: 10000)")
    parser.add_argument("--min-score",  type=float, default=4.0)
    parser.add_argument("--passes",     type=str, default="1,2,3",
                        help="Comma-separated passes to run: 1,2,3 (default: all)")
    parser.add_argument("--out-prefix", type=str, default="graphrag_question",
                        help="Output file prefix (default: graphrag_question → overwrites existing)")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print(
            "ANTHROPIC_API_KEY not set.\n"
            "Get it at: https://console.anthropic.com/\n"
            f"Estimated cost for {args.target} questions: ~${args.target * 0.001:.0f} USD"
        )
        return 1

    passes = [int(p) for p in args.passes.split(",")]
    rows   = load_eval_rows(args.min_score)
    print(f"Source eval rows: {len(rows)}")

    # Budget per pass
    n_passes = len(passes)
    per_pass  = args.target // n_passes
    budgets   = {p: per_pass for p in passes}
    budgets[passes[-1]] += args.target - per_pass * n_passes  # remainder to last pass

    print(f"Target: {args.target} | Passes: {passes} | Per pass: {list(budgets.values())}")
    est_cost = args.target * 0.001
    print(f"Estimated cost: ~${est_cost:.2f} USD")
    print(f"Estimated time: ~{args.target * 0.3 / 60:.0f} min (excluding cache hits)\n")

    if args.dry_run:
        print("[DRY RUN] Would run passes:", passes)
        print(f"  Pass 1 combos: {len(rows)} contexts × up to 11 combos = {len(rows)*11} possible")
        domain_chunks = defaultdict(int)
        for r in rows:
            d = _infer_domain(r["raw_chunks"])
            if d: domain_chunks[d] += len(r["raw_chunks"])
        print(f"  Pass 2 chunks available: {dict(domain_chunks)}")
        print(f"  Pass 3 new LOs per domain: ~{budgets.get(3, 0)//5}")
        return 0

    client = anthropic.Anthropic(api_key=api_key)
    cache  = load_cache()
    print(f"Cache: {len(cache)} entries loaded (will skip these API calls)\n")

    all_records: list[dict] = []

    if 1 in passes:
        recs = pass1_combo_expansion(client, rows, cache, budgets[1])
        all_records.extend(recs)

    if 2 in passes:
        recs = pass2_synthetic_contexts(client, rows, cache, budgets[2])
        all_records.extend(recs)

    if 3 in passes:
        recs = pass3_new_learning_objectives(client, rows, cache, budgets[3])
        all_records.extend(recs)

    # Merge with existing original data
    existing_train = []
    existing_eval  = []
    train_path = DATA_DIR / f"{args.out_prefix}_train.jsonl"
    eval_path  = DATA_DIR / f"{args.out_prefix}_eval.jsonl"
    if train_path.exists():
        for line in train_path.read_text(encoding="utf-8").splitlines():
            try: existing_train.append(json.loads(line))
            except: pass
    if eval_path.exists():
        for line in eval_path.read_text(encoding="utf-8").splitlines():
            try: existing_eval.append(json.loads(line))
            except: pass

    combined = existing_train + existing_eval + all_records

    # Deduplicate by assistant content fingerprint
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in combined:
        fp = rec["messages"][-1]["content"][:60]
        if fp not in seen:
            seen.add(fp)
            deduped.append(rec)

    random.shuffle(deduped)
    train_recs, eval_recs = split_train_eval(deduped, args.eval_ratio)

    write_jsonl(train_path, train_recs)
    write_jsonl(eval_path,  eval_recs)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  New records generated : {len(all_records)}")
    print(f"  Total after dedup     : {len(deduped)}")
    print(f"  Train                 : {len(train_recs)} → {train_path.name}")
    print(f"  Eval                  : {len(eval_recs)}  → {eval_path.name}")
    print(f"  Cache saved at        : {CACHE_FILE}")
    print(f"{'='*60}")
    print(f"\nNext: python finetune_data/train_question_lora.py --max-seq-len 1024 --epochs 3")
    print(f"Or upload to Kaggle and run with KAGGLE_MODE=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
