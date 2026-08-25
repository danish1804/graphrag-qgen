# ============================================================
# scripts/build_question_corpus.py
# Generate the question corpus for human labelling (golden corpus).
#
# WHY THE JUDGE IS OFF BY DEFAULT
# -------------------------------
# The generation pipeline normally runs an LLM judge and regenerates
# any question scoring below judge.score_threshold, up to 3 attempts.
# That is right for production and WRONG for a calibration corpus.
#
# To calibrate a judge against human raters you need items spanning
# the real quality range. If the judge filters out its own low-scoring
# questions before humans ever see them, the human labels cover only
# questions the judge already liked. You then cannot measure whether
# the judge detects bad questions, because no bad questions survived.
# That is classic range restriction, and it silently inflates any
# judge-human agreement statistic.
#
# So this script generates the NATURAL output distribution with the
# regeneration loop disabled. Judge scores are added afterwards, by
# running the independent judge over this fixed corpus (see
# --emit-judge-input), which also means switching judges later does
# not invalidate the human labels.
#
# Usage:
#   python scripts/build_question_corpus.py --limit 60
#   python scripts/build_question_corpus.py --limit 60 \
#       --focus database,operating_systems --focus-share 0.6
#   python scripts/build_question_corpus.py --limit 12 --dry-run
#
# Output:
#   outputs/question_corpus.json   (feeds scripts/build_eval_sheet.py)
# ============================================================
import os
import sys
import csv
import json
import random
import argparse
from collections import defaultdict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT, "dataset", "inputs", "input_questions.csv")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "question_corpus.json")
SEED = 42


def _norm_domain(d):
    """The LO bank contains cross-domain rows ('database+operating_systems')
    and 'auto' routing probes. Collapse them into one 'multi_domain'
    bucket so they do not fragment the sampling budget into 16 groups of
    one or two - while still keeping them in the corpus, since
    multi-domain retrieval is a real feature of the system."""
    d = (d or "").strip()
    if not d or d.lower() == "auto" or "+" in d:
        return "multi_domain"
    return d


VALID_DIFFICULTY = ("easy", "medium", "hard")
VALID_QTYPE = ("recall", "conceptual", "application", "case_study")


def load_learning_outcomes(path):
    """Read the learning-outcome bank, keeping ONLY the outcome text and
    domain.

    The bank's own difficulty and question_type columns are deliberately
    discarded, for two reasons:

    1. They are perfectly confounded - every 'easy' row is 'recall',
       every 'medium' is 'conceptual', every 'hard' is 'application'.
       If difficulty never varies independently of question format, a
       rater marking difficulty may be responding to the format instead,
       and the difficulty analysis cannot be interpreted.
    2. 15 rows carry difficulty='case_study', which is not a valid
       difficulty (config.yml defines easy/medium/hard only) and would
       raise KeyError inside the pipeline.

    This script assigns difficulty and question_type itself, crossed, so
    the two vary independently.
    """
    seen, rows = set(), []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            lo = (r.get("learning_objective") or "").strip()
            if not lo:
                continue
            raw = (r.get("domain") or "").strip()
            key = (lo.lower(), _norm_domain(raw))
            if key in seen:            # same LO repeated per difficulty
                continue
            seen.add(key)
            rows.append({"lo": lo, "domain": _norm_domain(raw),
                         "domain_raw": raw})
    return rows


def select(rows, limit, focus, focus_share, n_pairs=0):
    """Balanced selection across domain x difficulty x question_type.

    `focus` domains receive `focus_share` of the budget so the two
    priority domains are labelled deeply enough to report per-domain,
    while the others still get coverage. Per-domain reporting matters:
    expert-judgement accuracy varies enormously between domains of the
    same exam, so a pooled number hides the variation
    (see docs/EVALUATION_ROADMAP.md section 4.4).
    """
    random.seed(SEED)
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    for v in by_domain.values():
        random.shuffle(v)

    focus = [d for d in focus if d in by_domain]
    others = [d for d in sorted(by_domain) if d not in focus]

    n_focus = int(round(limit * focus_share)) if focus else 0
    n_other = limit - n_focus
    budget = {}
    if focus:
        per = n_focus // len(focus)
        for i, d in enumerate(focus):
            budget[d] = per + (1 if i < n_focus - per * len(focus) else 0)
    if others:
        per = n_other // len(others)
        for i, d in enumerate(others):
            budget[d] = per + (1 if i < n_other - per * len(others) else 0)

    # Assign difficulty x question_type in a crossed rotation, so the two
    # vary independently of each other and of domain.
    cells = [(d, q) for d in VALID_DIFFICULTY for q in VALID_QTYPE]

    chosen, leftovers, cell_i = [], [], 0
    for d, n in budget.items():
        pool = list(by_domain[d])
        take, rest = pool[:n], pool[n:]
        for r in take:
            diff, qt = cells[cell_i % len(cells)]
            cell_i += 1
            chosen.append({**r, "difficulty": diff, "qtype": qt})
        leftovers.extend(rest)

    # Integer division across domains leaves the budget short; top up
    # from the focus domains first so the requested limit is met.
    #
    # Only domains that received a budget are eligible. Without this,
    # setting --focus-share 1.0 still pulled dozens of items from
    # non-focus domains via the top-up, silently defeating the focus
    # restriction (multi_domain reached 23 of 60 in testing).
    eligible = {d for d, n in budget.items() if n > 0}
    leftovers = [r for r in leftovers if r["domain"] in eligible]
    if len(chosen) < limit and leftovers:
        random.shuffle(leftovers)
        leftovers.sort(key=lambda r: 0 if r["domain"] in focus else 1)
        for r in leftovers[:limit - len(chosen)]:
            diff, qt = cells[cell_i % len(cells)]
            cell_i += 1
            chosen.append({**r, "difficulty": diff, "qtype": qt})

    chosen = chosen[:limit]

    # Matched pairs: regenerate a sample of the SAME learning outcomes at
    # the opposite end of the difficulty scale. This is a within-item
    # control - it tests directly whether asking for 'hard' actually
    # produces a harder question than asking for 'easy' from identical
    # source material, which a between-item comparison cannot show.
    if n_pairs > 0:
        candidates = [c for c in chosen if c["difficulty"] in ("easy", "hard")]
        random.shuffle(candidates)
        pairs = []
        for j, base in enumerate(candidates[:n_pairs]):
            flipped = "hard" if base["difficulty"] == "easy" else "easy"
            base["pair_id"] = f"P{j + 1:02d}"
            pairs.append({**base, "difficulty": flipped,
                          "pair_id": f"P{j + 1:02d}"})
        chosen.extend(pairs)

    random.shuffle(chosen)          # raters must not infer grouping
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=60,
                    help="TOTAL items produced, including matched pairs. "
                         "The labelling workbook holds 60.")
    ap.add_argument("--focus", default="database,operating_systems",
                    help="comma-separated priority domains")
    ap.add_argument("--focus-share", type=float, default=0.6,
                    help="fraction of the budget for focus domains")
    ap.add_argument("--pairs", type=int, default=8,
                    help="matched pairs: N learning outcomes regenerated at "
                         "the opposite difficulty, as a within-item control "
                         "on whether the difficulty setting does anything. "
                         "Adds N extra items. 0 to disable.")
    ap.add_argument("--force-domain", action="store_true",
                    help="pass the requested domain explicitly instead of "
                         "auto-routing, for single-domain items. Guarantees "
                         "the per-domain sampling plan. Recommended for a "
                         "calibration corpus: it measures QUESTION quality, "
                         "and routing is measured separately by "
                         "scripts/test_routing_accuracy.py. Multi-domain "
                         "items still auto-route.")
    ap.add_argument("--with-judge", action="store_true",
                    help="ENABLE the judge + regeneration loop. Not "
                         "recommended for a calibration corpus - see the "
                         "range-restriction note at the top of this file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the selection without loading any model")
    args = ap.parse_args()

    focus = [d.strip() for d in args.focus.split(",") if d.strip()]

    if not os.path.isfile(args.input):
        sys.exit(f"[!] learning-outcome bank not found: {args.input}")
    rows = load_learning_outcomes(args.input)
    print(f"Loaded {len(rows)} learning outcomes from {args.input}")

    n_pairs = max(0, min(args.pairs, args.limit // 4))
    if n_pairs != args.pairs:
        print(f"[!] --pairs reduced {args.pairs} -> {n_pairs} "
              f"(capped at a quarter of --limit)")
    base = args.limit - n_pairs          # pairs add one item each
    picked = select(rows, base, focus, args.focus_share, n_pairs)
    if len(picked) > 60:
        print(f"[!] {len(picked)} items exceeds the 60-row workbook "
              f"capacity; the sheet builder will truncate.")

    def dist(key):
        c = defaultdict(int)
        for r in picked:
            c[r[key]] += 1
        return c

    print(f"\nSelected {len(picked)} items "
          f"({base} base + {n_pairs} matched-pair duplicates)")
    print("\n  by domain")
    for d, n in sorted(dist("domain").items()):
        print(f"   {d:26}{n:>4}{'  (focus)' if d in focus else ''}")
    print("\n  by difficulty")
    for d, n in sorted(dist("difficulty").items()):
        print(f"   {d:26}{n:>4}")
    print("\n  by question_type")
    for d, n in sorted(dist("qtype").items()):
        print(f"   {d:26}{n:>4}")

    cross = defaultdict(int)
    for r in picked:
        cross[(r["difficulty"], r["qtype"])] += 1
    print("\n  difficulty x question_type (should be spread, not 1:1)")
    for k in sorted(cross):
        print(f"   {k[0]:12} x {k[1]:14}{cross[k]:>4}")
    n_paired = sum(1 for r in picked if r.get("pair_id"))
    if n_paired:
        print(f"\n  matched pairs: {n_paired} items "
              f"({n_paired // 2} learning outcomes at two difficulties)")

    if args.dry_run:
        print("\n--dry-run: stopping before model load.")
        for r in picked[:8]:
            pid = f" pair={r['pair_id']}" if r.get("pair_id") else ""
            print(f"   [{r['domain']}/{r['difficulty']}/{r['qtype']}{pid}] "
                  f"{r['lo']}")
        return

    print("\nLoading Qwen 2.5 7B (+ LoRA). First load takes a minute...")
    from graphrag.graphrag_pipeline import GraphRAGPipeline
    p = GraphRAGPipeline()
    p.load_models()

    if args.with_judge:
        print("[!] Judge ENABLED - regeneration loop active. This corpus "
              "will be quality-filtered and is NOT suitable for judge "
              "calibration (range restriction).")
        p.load_judge_model()
    else:
        print("Judge DISABLED - generating the natural quality "
              "distribution for human calibration.")

    corpus, failures = [], 0
    for i, r in enumerate(picked, start=1):
        print(f"  [{i:>3}/{len(picked)}] {r['domain']:22} {r['lo'][:46]}")
        try:
            # Multi-domain items always auto-route; single-domain items
            # can be pinned so the sampling plan survives (see
            # --force-domain).
            want = (r["domain"] if (args.force_domain
                                    and r["domain"] != "multi_domain")
                    else None)
            res = p.generate_with_routing(
                learning_objective=r["lo"],
                difficulty=r["difficulty"],
                question_type=r["qtype"],
                domain=want,
            )
            q = (res.get("question") or "").strip()
            if not q or res.get("error"):
                failures += 1
                print(f"        skipped: {res.get('error', 'empty question')}")
                continue
            md = res.get("metadata", {}) or {}
            corpus.append({
                "question_id": f"Q{len(corpus) + 1:03d}",
                "question": q,
                "sample_answer": md.get("sample_answer", ""),
                "learning_outcome": r["lo"],
                "domain": md.get("domain") or md.get("domain_routed_to")
                          or r["domain"],
                "requested_domain": r["domain"],
                "difficulty": r["difficulty"],
                "question_type": r["qtype"],
                "pair_id": r.get("pair_id"),
                # The retrieval context MUST be stored. Judging a question
                # for faithfulness or groundedness without the material it
                # was generated from is impossible, and groundedness is
                # exactly the dimension we want to compare against the
                # human raters. Without this the corpus cannot support
                # post-hoc judging at all.
                "retrieved_chunks": md.get("retrieved_chunks"),
                "context": md.get("context"),
                "routing_confidence": md.get("routing_confidence"),
                "sources": md.get("sources"),
                "key_concepts": md.get("key_concepts"),
                "judge_scores": md.get("judge_scores"),   # None when off
                "judge_enabled": bool(args.with_judge),
            })
        except Exception as exc:               # keep going on one failure
            failures += 1
            print(f"        FAILED: {exc}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "judge_enabled": bool(args.with_judge),
        "force_domain": bool(args.force_domain),
        "focus_domains": focus,
        "requested": len(picked),
        "produced": len(corpus),
        "failures": failures,
        "note": ("Judge regeneration disabled: this is the natural output "
                 "distribution, required for unbiased judge calibration."
                 if not args.with_judge else
                 "Judge ENABLED - quality-filtered, not suitable for "
                 "judge calibration."),
        "questions": corpus,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {args.out}")
    print(f"  produced {len(corpus)} questions ({failures} failures)")
    dd = defaultdict(int)
    for c in corpus:
        dd[c["domain"]] += 1
    for d in sorted(dd):
        print(f"   {d:26}{dd[d]:>4}")
    print("\nNext:")
    print("  1. python scripts/build_eval_sheet.py --corpus outputs/question_corpus.json")
    print("  2. Run a calibration round: all raters mark the same 5-6 items,")
    print("     compare, then proceed to the full set.")
    print("  3. Judge scores are added later over this FIXED corpus, so")
    print("     changing judge does not invalidate the human labels.")


if __name__ == "__main__":
    main()
