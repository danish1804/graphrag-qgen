# ============================================================
# scripts/test_routing_accuracy.py
# Measure domain-routing accuracy against the labelled learning-outcome
# bank. Judge-free and fast: embeds each learning outcome and compares
# the router's choice with the domain the outcome is filed under.
#
# Motivation: generating the 60-item corpus produced only 4 operating
# systems items and 1 computer architecture item out of a planned 16
# and 7, because ~48% of items auto-routed to multi-domain. Several
# routing decisions were clearly wrong ("Banker algorithm" -> database,
# "Page replacement algorithms" -> database). This quantifies it.
#
# Usage:
#   python scripts/test_routing_accuracy.py
#   python scripts/test_routing_accuracy.py --show-errors
# ============================================================
import os
import sys
import csv
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "dataset", "inputs", "input_questions.csv")


def load_single_domain_los():
    """Learning outcomes filed under exactly one domain."""
    seen, rows = set(), []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            lo = (r.get("learning_objective") or "").strip()
            dom = (r.get("domain") or "").strip()
            if not lo or not dom or "+" in dom or dom.lower() == "auto":
                continue
            key = (lo.lower(), dom)
            if key in seen:
                continue
            seen.add(key)
            rows.append((lo, dom))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--margin", type=float, default=None,
                    help="override routing.margin_threshold for a what-if")
    args = ap.parse_args()

    from utils.config import cfg
    from utils.embedder import Embedder
    from graphrag.graphrag_domain_indexer import (
        build_domain_profiles, route_by_embedding)

    margin = (args.margin if args.margin is not None
              else cfg.get("routing", {}).get("margin_threshold", 0.05))

    rows = load_single_domain_los()
    print(f"Evaluating {len(rows)} single-domain learning outcomes")
    print(f"margin_threshold = {margin}\n")

    embedder = Embedder()
    build_domain_profiles(embedder)

    correct = ambiguous = 0
    wrong = []
    per_domain = defaultdict(lambda: {"n": 0, "ok": 0, "amb": 0})
    margins = []

    for lo, expected in rows:
        scores = route_by_embedding(lo, embedder, return_all_scores=True)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = best_s - second_s
        margins.append(gap)

        d = per_domain[expected]
        d["n"] += 1
        if best == expected:
            correct += 1
            d["ok"] += 1
        else:
            wrong.append((lo, expected, best, best_s, ranked[1][0], second_s))
        if gap < margin:
            ambiguous += 1
            d["amb"] += 1

    n = len(rows)
    print("=" * 78)
    print("ROUTING ACCURACY")
    print("=" * 78)
    print(f"  top-1 correct        : {correct}/{n} ({correct / n * 100:.1f}%)")
    print(f"  below margin ({margin})  : {ambiguous}/{n} "
          f"({ambiguous / n * 100:.1f}%) -> forced to multi-domain")
    print(f"  median margin        : "
          f"{sorted(margins)[len(margins) // 2]:.3f}")
    print(f"  min / max margin     : {min(margins):.3f} / {max(margins):.3f}")

    print("\n  per requested domain")
    print(f"   {'domain':26}{'n':>4}{'correct':>9}{'ambiguous':>11}")
    for dom in sorted(per_domain):
        s = per_domain[dom]
        print(f"   {dom:26}{s['n']:>4}{s['ok'] / s['n'] * 100:>8.0f}%"
              f"{s['amb'] / s['n'] * 100:>10.0f}%")

    if wrong:
        print(f"\n  MISROUTED ({len(wrong)}):")
        for lo, exp, got, gs, snd, ss in (wrong if args.show_errors
                                          else wrong[:12]):
            print(f"   {lo[:44]:46} {exp:22} -> {got} ({gs:.3f}) "
                  f"[2nd {snd} {ss:.3f}]")
        if not args.show_errors and len(wrong) > 12:
            print(f"   ... {len(wrong) - 12} more (--show-errors)")

    print("\n" + "-" * 78)
    print("  Interpreting this: low top-1 accuracy means the domain")
    print("  PROFILES are not discriminating - they are built from the")
    print("  description + keywords in config.yml, so thin keyword lists")
    print("  leave whole topic areas unrepresented. A high 'ambiguous'")
    print("  rate means margin_threshold is sending clear winners to")
    print("  multi-domain retrieval unnecessarily.")


if __name__ == "__main__":
    main()
