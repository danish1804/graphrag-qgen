# ============================================================
# scripts/measure_bloom_alignment.py
# Does a generated question actually reach the cognitive level it
# was asked for?
#
# Wiring the Bloom level into the prompt makes the system REQUEST a
# level. Whether it HITS that level is a separate empirical question,
# and this measures the gap.
#
# Run it on a corpus generated BEFORE the wiring to get a baseline,
# then on one generated after, to see whether the instruction moved
# anything. Without the baseline, any post-wiring number is unanchored.
#
# Usage:
#   python scripts/measure_bloom_alignment.py
#   python scripts/measure_bloom_alignment.py --corpus outputs/question_corpus.json
#   python scripts/measure_bloom_alignment.py --show-misses
#
# HONESTY NOTE: verb-based classification is a coarse proxy. Human
# raters agree on Bloom level at only Cohen's kappa 0.12-0.60, and
# GPT-4 agreed with teacher classification 65-69% with roughly half its
# "Analysing" labels rated "Evaluating". Report the coarse
# lower/higher-order split as the headline; treat exact-level match as
# indicative only.
# ============================================================
import os
import sys
import json
import argparse
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bloom import (bloom_level_for, classify_by_verb, coarse_band,
                         bloom_name, BLOOM_LEVELS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "outputs", "question_corpus.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT)
    ap.add_argument("--show-misses", action="store_true",
                    help="print every question whose achieved level differs")
    args = ap.parse_args()

    if not os.path.isfile(args.corpus):
        sys.exit(f"[!] corpus not found: {args.corpus}")
    payload = json.load(open(args.corpus, encoding="utf-8"))
    qs = payload.get("questions", [])
    if not qs:
        sys.exit("[!] corpus has no questions")

    print(f"{args.corpus}")
    print(f"generated_at={payload.get('generated_at')}   n={len(qs)}")
    print("=" * 78)

    rows, unclassified = [], []
    for q in qs:
        qt = q.get("question_type", "")
        want = bloom_level_for(qt)
        got, verb = classify_by_verb(q.get("question", ""))
        if got is None:
            unclassified.append(q)
        rows.append({
            "id": q.get("question_id"), "qtype": qt, "want": want,
            "got": got, "verb": verb, "q": q.get("question", ""),
        })

    scored = [r for r in rows if r["want"] is not None and r["got"] is not None]

    # ── exact and coarse agreement ──────────────────────────
    exact = sum(1 for r in scored if r["got"] == r["want"])
    coarse = sum(1 for r in scored
                 if coarse_band(r["got"]) == coarse_band(r["want"]))
    at_or_above = sum(1 for r in scored if r["got"] >= r["want"])

    print("\nALIGNMENT")
    print("-" * 78)
    n = len(scored)
    print(f"  classified          : {n}/{len(rows)}"
          f"   ({len(unclassified)} had no recognisable level verb)")
    if n:
        print(f"  exact level match   : {exact}/{n} ({exact/n*100:.0f}%)"
              f"   <- indicative only")
        print(f"  coarse band match   : {coarse}/{n} ({coarse/n*100:.0f}%)"
              f"   <- the reportable figure")
        print(f"  at or above target  : {at_or_above}/{n} "
              f"({at_or_above/n*100:.0f}%)")

    # ── where does it land per requested type? ──────────────
    print("\nREQUESTED vs ACHIEVED")
    print("-" * 78)
    print(f"  {'question_type':16}{'asked':>7}{'n':>5}   achieved level distribution")
    by_type = defaultdict(list)
    for r in scored:
        by_type[r["qtype"]].append(r)
    for qt in sorted(by_type):
        rs = by_type[qt]
        want = rs[0]["want"]
        dist = Counter(r["got"] for r in rs)
        bar = "  ".join(f"L{lv}:{dist[lv]}" for lv in sorted(dist))
        print(f"  {qt:16}{want:>7}{len(rs):>5}   {bar}")

    # ── the direction of the error matters ──────────────────
    below = [r for r in scored if r["got"] < r["want"]]
    above = [r for r in scored if r["got"] > r["want"]]
    print("\nDIRECTION OF MISS")
    print("-" * 78)
    print(f"  below target : {len(below)}   <- the failure that matters:")
    print(f"                     asked for higher-order, produced recall")
    print(f"  above target : {len(above)}   (usually acceptable)")

    if below:
        print("\n  worst cases (asked high, delivered low):")
        for r in sorted(below, key=lambda r: r["got"] - r["want"])[:6]:
            print(f"    {r['id']} {r['qtype']:12} "
                  f"asked L{r['want']} -> got L{r['got']} "
                  f"({r['verb']!r})")
            print(f"       {r['q'][:96]}")

    if args.show_misses:
        print("\nALL MISSES")
        print("-" * 78)
        for r in scored:
            if r["got"] != r["want"]:
                print(f"  {r['id']} {r['qtype']:12} L{r['want']}->L{r['got']} "
                      f"({r['verb']}) {r['q'][:70]}")

    if unclassified:
        print(f"\nUNCLASSIFIED ({len(unclassified)})")
        print("-" * 78)
        print("  No recognisable Bloom verb. Often a 'What is the purpose of'")
        print("  phrasing, which is a real gap in the verb list rather than a")
        print("  defect in the question.")
        for q in unclassified[:5]:
            print(f"    {q.get('question_id')}: {q.get('question','')[:82]}")

    print("\n" + "-" * 78)
    print("  Verb classification is a coarse proxy. Human agreement on Bloom")
    print("  level is only kappa 0.12-0.60, and the analyse/evaluate/create")
    print("  boundary is where both human and automated labelling break down.")
    print("  Quote the coarse band figure; treat exact match as indicative.")


if __name__ == "__main__":
    main()
