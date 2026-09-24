# ============================================================
# scripts/test_bloom.py
# Bloom's taxonomy wiring and verb-based classification.
#
# Background: config.yml has carried `bloom_level` against every
# question type since the original build, but no code read it. The
# cognitive level was asserted in config and never acted on. This
# verifies that it now reaches the prompt, and that the measurement
# side behaves.
#
# Run:
#   python scripts/test_bloom.py
# ============================================================
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bloom import (BLOOM_LEVELS, bloom_level_for, bloom_block,
                         classify_by_verb, coarse_band, bloom_name)
from utils.prompts import build_question_messages


def main():
    failures = []

    # ── 1. config wiring ────────────────────────────────────
    print("=" * 70)
    print("1. CONFIG -> LEVEL MAPPING")
    print("=" * 70)
    expected = {"recall": 1, "conceptual": 2,
                "application": 3, "case_study": 4}
    for qt, want in expected.items():
        got = bloom_level_for(qt)
        ok = got == want
        failures += [] if ok else [f"{qt}: expected level {want}, got {got}"]
        print(f"  {qt:14} -> level {got} ({bloom_name(got)})"
              f"{'' if ok else f'   <-- expected {want}'}")
    got = bloom_level_for("nonexistent_type")
    print(f"  {'unconfigured':14} -> {got}   (None expected, not a default)")
    if got is not None:
        failures.append("unconfigured type should return None")

    # ── 2. the block reaches the prompt ─────────────────────
    print()
    print("=" * 70)
    print("2. BLOOM BLOCK REACHES THE GENERATION PROMPT")
    print("=" * 70)
    msgs = build_question_messages(
        context="Paging maps virtual addresses to physical frames.",
        learning_objective="Explain virtual memory",
        difficulty="hard",
        question_type="application",
    )
    user = msgs[1]["content"]
    for needle in ("Cognitive level: 3 - Apply", "Do NOT write a question"):
        present = needle in user
        failures += [] if present else [f"prompt missing: {needle!r}"]
        print(f"  {'found' if present else 'MISSING':>8}  {needle!r}")

    print("\n  --- application prompt, Bloom section ---")
    for line in bloom_block("application").splitlines():
        print(f"    {line}")

    # recall must NOT carry the lower-verb prohibition (it IS level 1)
    recall_block = bloom_block("recall")
    if "Do NOT write a question" in recall_block:
        failures.append("recall should not prohibit lower-order verbs")
    print(f"\n  recall block prohibits lower verbs: "
          f"{'Do NOT write' in recall_block}   (False expected)")

    # ── 3. verb classification ──────────────────────────────
    print()
    print("=" * 70)
    print("3. VERB-BASED CLASSIFICATION")
    print("=" * 70)
    cases = [
        ("Define the term deadlock.", 1),
        ("List the four ACID properties.", 1),
        ("Explain how paging translates a virtual address.", 2),
        ("Describe the role of a page table.", 2),
        ("Calculate the effective access time given a 2% miss rate.", 3),
        ("Apply the banker's algorithm to determine whether the state is safe.", 3),
        ("Compare pre-emptive and non-pre-emptive scheduling.", 4),
        ("Justify the choice of a write-back cache for this workload.", 5),
        ("Design a locking strategy that prevents deadlock.", 6),
    ]
    for text, want in cases:
        lvl, verb = classify_by_verb(text)
        ok = lvl == want
        failures += [] if ok else [f"{text[:40]!r}: expected {want}, got {lvl}"]
        print(f"  {str(lvl):>4} {str(bloom_name(lvl)):12} "
              f"({str(verb):12}) {'ok ' if ok else 'BAD'} {text[:44]}")

    # highest level present should govern
    mixed = "Describe the two schemes and evaluate which suits a real-time system."
    lvl, verb = classify_by_verb(mixed)
    ok = lvl == 5
    failures += [] if ok else [f"mixed-verb case: expected 5, got {lvl}"]
    print(f"\n  mixed verbs ('describe' + 'evaluate') -> level {lvl} "
          f"({verb}){'' if ok else '   <-- should be 5, highest governs'}")

    # case_study: classify the QUESTION line, not the scenario
    cs = ("SCENARIO: A company lists its five servers and defines each role.\n"
          "QUESTION: Evaluate which server should host the primary replica.")
    lvl, verb = classify_by_verb(cs)
    ok = lvl == 5
    failures += [] if ok else [f"case_study: expected 5 from QUESTION line, got {lvl}"]
    print(f"  case_study (scenario says 'lists'/'defines') -> level {lvl} "
          f"({verb}){'' if ok else '   <-- should read the QUESTION line only'}")

    # ── 4. coarse band ──────────────────────────────────────
    print()
    print("=" * 70)
    print("4. COARSE BAND (the split that can be reported with confidence)")
    print("=" * 70)
    for lvl in (1, 2, 3, 4, 5, 6, None):
        print(f"  level {str(lvl):4} -> {coarse_band(lvl)}")
    if coarse_band(2) != "lower-order" or coarse_band(3) != "higher-order":
        failures.append("coarse band boundary wrong")

    # ── result ──────────────────────────────────────────────
    print()
    print("-" * 70)
    if failures:
        print(f"[FAIL] {len(failures)} problem(s):")
        for f in failures:
            print(f"   {f}")
        sys.exit(1)
    print("[OK] Bloom level reaches the prompt and classification behaves.")
    print()
    print("NOTE: this proves the level is REQUESTED, not that it is HIT.")
    print("Use scripts/measure_bloom_alignment.py on a generated corpus to")
    print("measure the gap between requested and achieved level.")


if __name__ == "__main__":
    main()
