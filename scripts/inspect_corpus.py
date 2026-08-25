# ============================================================
# scripts/inspect_corpus.py
# Eyeball a generated question corpus before sending it to raters.
#
# Prints each question with its routing provenance, then a summary
# including how often auto-routing landed somewhere other than the
# requested domain (routing drift), which changes the effective
# per-domain sample sizes.
#
# Usage:
#   python scripts/inspect_corpus.py
#   python scripts/inspect_corpus.py outputs/_smoke.json
#   python scripts/inspect_corpus.py outputs/question_corpus.json --full
# ============================================================
import os
import re
import sys
import json
import argparse
from collections import Counter, defaultdict

# ── Dangling-reference detection ────────────────────────────
# A generated question must be self-contained. When the generator
# copies a cross-reference out of the source chunk ("...shown in
# Figure 2.10"), the student cannot see the referent and the item is
# unanswerable as written - an item-writing flaw in Haladyna's sense.
#
# Deliberately deterministic rules, not an LLM: on 200 questions across
# four subjects, rule-based detection caught 91% of human-annotated
# item-writing flaws against GPT-4's 79% (Moore et al., EC-TEL 2023).
# Rules are also cheaper and auditable.
#
# NOTE: this MEASURES the rate; it does not filter. A calibration
# corpus needs flawed items in it, or the judge's ability to detect
# flaws cannot be assessed.
DANGLING_PATTERNS = [
    (r'\b(figure|fig\.?|table|chart|diagram|graph|exhibit)\s*\d',
     "figure/table reference"),
    (r'\b(section|chapter|appendix|equation|eq\.?)\s*\d', "section reference"),
    (r'\b(shown|illustrated|depicted|described|listed|given)\s+'
     r'(in|above|below)\b', "shown in/above/below"),
    (r'\b(the\s+)?(above|following|preceding)\s+'
     r'(figure|table|diagram|passage|text|code|example)\b', "the above/following X"),
    (r'\b(in|refer\s+to|according\s+to|based\s+on|from)\s+the\s+'
     r'(passage|excerpt|text|reading|article|scenario\s+above)\b',
     "refers to an unseen passage"),
    (r'\bas\s+(shown|seen|illustrated|discussed)\b', "as shown/discussed"),
    (r'\bthis\s+(figure|table|diagram|chart)\b', "this figure/table"),
    # Found by reading the first 60-item corpus: the generator refers to
    # source artefacts the student cannot see.
    (r'\b(provided|given|following|above)\s+'
     r'(code\s*snippet|snippet|code|program|listing|pseudocode)\b',
     "refers to unseen code"),
    (r'\b(the\s+)?course\s+material\b', "refers to 'the course material'"),
    # Found by the independent judge, not by these rules: it scored
    # "...the two main types of testing mentioned in the context..." at
    # 2.2, the lowest in the corpus, while the rules passed it. The two
    # methods catch different flaw classes and are worth running together.
    (r'\b(mentioned|described|discussed|listed|shown|stated|referred\s+to)\s+'
     r'in\s+the\s+(context|material|source|content)\b',
     "refers to unseen context"),
    (r'\bin\s+the\s+(given|provided|above)\s+context\b', "unseen context"),
    (r'\baccording\s+to\s+the\s+(course|material|textbook|book|notes|lecture)\b',
     "according to the textbook/notes"),
    (r'\bwhat\s+does\s+(chapter|section|unit|module)\s', "question about the book"),
]
_DANGLING = [(re.compile(p, re.IGNORECASE), lbl)
             for p, lbl in DANGLING_PATTERNS]


def dangling_refs(text):
    """Return the list of dangling-reference labels found in a question."""
    hits = []
    for rx, label in _DANGLING:
        m = rx.search(text or "")
        if m:
            hits.append(f"{label} ({m.group(0).strip()!r})")
    return hits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "outputs", "question_corpus.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--full", action="store_true",
                    help="print sample answers too")
    ap.add_argument("--max", type=int, default=100)
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        sys.exit(f"[!] not found: {args.path}")
    payload = json.load(open(args.path, encoding="utf-8"))
    qs = payload.get("questions", [])
    print(f"{args.path}")
    print(f"generated_at={payload.get('generated_at')}  "
          f"judge_enabled={payload.get('judge_enabled')}  "
          f"produced={payload.get('produced')}  "
          f"failures={payload.get('failures')}")
    print("=" * 96)

    drift = 0
    lengths = []
    for q in qs[:args.max]:
        req = q.get("requested_domain")
        got = q.get("domain")
        moved = req and got and req != got
        drift += 1 if moved else 0
        arrow = f"{req} -> {got}" if moved else str(got)
        pid = f"  pair={q['pair_id']}" if q.get("pair_id") else ""
        text = (q.get("question") or "").strip()
        lengths.append(len(text))
        print(f"\n[{q.get('question_id')}] {arrow}"
              f"  | {q.get('difficulty')}/{q.get('question_type')}{pid}")
        print(f"  LO : {q.get('learning_outcome')}")
        print(f"  Q  : {text if text else '(EMPTY)'}")
        dang = dangling_refs(text)
        if dang:
            print(f"  !! NOT SELF-CONTAINED: {'; '.join(dang)}")
        if args.full:
            ans = (q.get("sample_answer") or "").strip()
            print(f"  A  : {ans[:400]}{'...' if len(ans) > 400 else ''}")

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    empty = sum(1 for q in qs if not (q.get("question") or "").strip())
    print(f"  questions          : {len(qs)}")
    print(f"  empty              : {empty}"
          f"{'   <-- investigate' if empty else ''}")
    if lengths:
        print(f"  length (chars)     : min {min(lengths)}  "
              f"median {sorted(lengths)[len(lengths)//2]}  max {max(lengths)}")
    print(f"  routing drift      : {drift}/{len(qs)} "
          f"({drift / len(qs) * 100:.0f}%) landed outside the requested domain"
          if qs else "")

    flagged = [(q.get("question_id"), dangling_refs(q.get("question")))
               for q in qs]
    flagged = [(i, d) for i, d in flagged if d]
    if qs:
        print(f"  dangling refs      : {len(flagged)}/{len(qs)} "
              f"({len(flagged) / len(qs) * 100:.0f}%) not self-contained")
        for qid, d in flagged[:10]:
            print(f"      {qid}: {d[0]}")
        if flagged:
            print("      (these are LEFT IN the corpus on purpose - a")
            print("       calibration set needs flawed items, or the judge's")
            print("       ability to detect flaws cannot be measured)")

    def show(label, key):
        c = Counter(q.get(key) for q in qs)
        print(f"\n  by {label}")
        for k, v in sorted(c.items(), key=lambda kv: str(kv[0])):
            print(f"   {str(k):26}{v:>4}")

    show("routed domain", "domain")
    show("requested domain", "requested_domain")
    show("difficulty", "difficulty")
    show("question_type", "question_type")

    # duplicate detection - identical questions waste rater time
    seen = defaultdict(list)
    for q in qs:
        seen[" ".join((q.get("question") or "").lower().split())].append(
            q.get("question_id"))
    dupes = {k: v for k, v in seen.items() if len(v) > 1 and k}
    if dupes:
        print(f"\n  [!] {len(dupes)} duplicate question texts:")
        for k, v in list(dupes.items())[:5]:
            print(f"      {v} : {k[:70]}...")
    else:
        print("\n  no duplicate question texts.")


if __name__ == "__main__":
    main()
