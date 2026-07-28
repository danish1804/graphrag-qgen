# ============================================================
# scripts/test_alias_merge.py
# GraphRAG 1.1 test: synonym/alias merging in normalize_entity.
#
# Part 1 (fast, no graph): assert that abbreviation and expansion
#   forms normalise to the SAME canonical string, so the two-pass
#   builder will union them into one node on re-index.
# Part 2 (preview): apply the NEW normalize_entity to the CURRENT
#   graph's node names and report how many nodes would collapse,
#   plus the biggest merges. This previews the re-index effect
#   WITHOUT rebuilding.
#
# Run:
#   python scripts/test_alias_merge.py
# ============================================================
import os
import sys
import pickle
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.graphrag_indexer import normalize_entity

BASE_DIR  = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
GRAPH_PKL = os.path.join(BASE_DIR, "indexes", "graphrag_database_graph.pkl")

# (abbreviation form, expansion form) -> must map to the same canonical
PAIRS = [
    ("SQL", "Structured Query Language"),
    ("CNNs", "Convolutional Neural Networks"),
    ("DBMS", "database management systems"),
    ("DBA", "Database Administrator"),
    ("LSTM", "long short-term memory"),
    ("OOP", "object-oriented programming"),
]


def part1_unit():
    print("Part 1 — normalisation unit checks")
    ok = True
    for abbr, expansion in PAIRS:
        a, b = normalize_entity(abbr), normalize_entity(expansion)
        status = "OK " if a == b else "FAIL"
        if a != b:
            ok = False
        print(f"  [{status}] {abbr!r:8} -> {a!r:32} | {expansion!r:38} -> {b!r}")
    assert ok, "some abbreviation/expansion pairs did NOT converge"
    print("  All pairs converge to one canonical.\n")


def part2_preview():
    print("Part 2 — merge preview on the current database graph")
    if not os.path.exists(GRAPH_PKL):
        print(f"  (skipped — graph not found at {GRAPH_PKL})")
        return
    G = pickle.load(open(GRAPH_PKL, "rb"))
    groups = defaultdict(list)
    for n in G.nodes():
        groups[normalize_entity(n)].append(n)
    merges = {k: v for k, v in groups.items() if len(v) > 1}
    collapsed = G.number_of_nodes() - len(groups)
    print(f"  nodes {G.number_of_nodes()} -> {len(groups)} "
          f"({collapsed} collapse, {len(merges)} merge groups)")
    print("  biggest merges (canonical <= originals with degree):")
    for k, v in sorted(merges.items(),
                       key=lambda kv: -sum(G.degree(x) for x in kv[1]))[:12]:
        degs = {x: G.degree(x) for x in v}
        print(f"    {k!r}: {degs}")
    print("\n  Note: at re-index the merge also consolidates EDGE weight")
    print("  (co-occurrences of every alias accrue to the one node),")
    print("  so the real benefit is larger than the node-count drop.")


if __name__ == "__main__":
    part1_unit()
    part2_preview()
    print("\nDone.")
