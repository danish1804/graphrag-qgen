# ============================================================
# scripts/test_technical_numeric.py
# Task 24 — numeric-prefixed technical concepts must survive
# entity filtering, while slide/figure artifacts must not.
#
# Background: two long-standing rules in is_valid_entity
#   re.match(r'^\d+[a-z]', text)      # "22d", "1motivation"
#   re.match(r'^\d+[a-z\s]', text)    # slide outline numbering
# rejected EVERY entity beginning with a digit. They were written
# for deep-learning slide artifacts, but they also removed
# normalisation forms (2nf/3nf), bit widths (32 bit), processor
# families (8086) and n-tier/n-phase architecture terms from all
# five domain graphs.
#
# Run:
#   python scripts/test_technical_numeric.py
# ============================================================
import os
import re
import ast
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEXER = os.path.join(ROOT, "graphrag", "graphrag_indexer.py")


def load_validator():
    """Load is_valid_entity (and its helpers) without importing the
    whole module, which would pull in faiss/spacy."""
    src = open(INDEXER, encoding="utf-8").read()
    tree = ast.parse(src)
    ns = {"re": re, "MIN_ENTITY_LEN": 3, "MAX_ENTITY_LEN": 60}

    wanted_funcs = {"_is_technical_numeric", "is_valid_entity"}
    wanted_names = {"TECHNICAL_NUMERIC", "_TECHNICAL_NUMERIC_RE"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            exec(ast.get_source_segment(src, node), ns)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & wanted_names:
                exec(ast.get_source_segment(src, node), ns)
    if "is_valid_entity" not in ns:
        sys.exit("[!] could not load is_valid_entity")
    return ns["is_valid_entity"]


# Concepts that MUST be kept — real IT terminology
MUST_KEEP = [
    # normalisation forms (database) — the 2NF/3NF demo question
    "2nf", "3nf", "1nf", "4nf",
    # bit widths (architecture, security)
    "32 bit", "64 bit architecture", "128 bit key", "64-bit register",
    # processor families (qualified form only — see MUST_REJECT note)
    "8086 processor", "80386 cpu", "68000 microprocessor",
    # n-phase / n-tier / n-stage names
    "2 phase commit", "2 phase locking", "3 tier architecture",
    "5 stage pipeline", "3 way handshake",
    # ordinary concepts that merely start with a reference word
    "page fault rate", "page table", "page replacement",
    "table schema", "virtual memory", "cache coherence",
    "b tree", "hash join", "deadlock detection",
]

# Strings that MUST be rejected — extraction artifacts
MUST_REJECT = [
    # slide-outline numbering (the original target of the rules)
    "1motivation", "3basic", "22d",
    # section numbers
    "8.3.1 motivation", "14.1.3", "2.4 introduction",
    # numbered / lettered cross references
    "section 14.1", "table 5", "figure a.3", "section g.6",
    "figure k.49", "appendix m historical perspective",
    "chapter two memory hierarchy design",
    # citation initials
    "j. h", "a b",
    # publishers / generic words
    "springer", "data", "value",
    # Documented limitation: a BARE processor number is indistinguishable
    # from a page or line number, so it is rejected by design. The
    # qualified forms above ("8086 processor") are kept.
    "80386", "68000",
]


def main():
    is_valid_entity = load_validator()
    failures = []

    print("=" * 66)
    print("MUST KEEP — genuine technical concepts")
    print("=" * 66)
    for term in MUST_KEEP:
        ok = is_valid_entity(term)
        flag = "" if ok else "   <-- FAIL (wrongly rejected)"
        if not ok:
            failures.append(("kept", term))
        print(f"  {term:44}{str(ok):>6}{flag}")

    print()
    print("=" * 66)
    print("MUST REJECT — extraction artifacts")
    print("=" * 66)
    for term in MUST_REJECT:
        ok = is_valid_entity(term)
        flag = "" if not ok else "   <-- FAIL (still passing)"
        if ok:
            failures.append(("rejected", term))
        print(f"  {term:44}{str(ok):>6}{flag}")

    print()
    print("-" * 66)
    total = len(MUST_KEEP) + len(MUST_REJECT)
    if failures:
        print(f"[FAIL] {len(failures)} of {total} cases wrong:")
        for kind, term in failures:
            print(f"   should have been {kind}: {term!r}")
        sys.exit(1)
    print(f"[OK] all {total} cases correct "
          f"({len(MUST_KEEP)} kept, {len(MUST_REJECT)} rejected).")


if __name__ == "__main__":
    main()
