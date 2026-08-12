# ============================================================
# scripts/test_entity_quality.py
# GraphRAG task-22 test: entity-extraction quality hardening.
#
# Part 1 (unit): the new rules reject junk concepts (section numbers,
#   numbered headings, author initials) and recover soft-hyphen splits,
#   while KEEPING real concepts (guards against over-rejection).
# Part 2 (preview): how many CURRENT graph nodes the new is_valid_entity
#   would reject — i.e. the junk that gets cleaned out on re-index.
#
# Run:
#   python scripts/test_entity_quality.py
# ============================================================
import os
import sys
import pickle
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.graphrag_indexer import is_valid_entity, normalize_entity

BASE_DIR = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
DOMAINS = ["graphrag_database", "graphrag_deep_learning",
           "graphrag_software_engineering", "graphrag_operating_systems",
           "graphrag_computer_architecture"]

SHOULD_REJECT = ["8.3.1 motivation", "section 14.1.3", "chapter 12 active",
                 "j. h", "a b", "figure 2.4"]
SHOULD_KEEP   = ["inner join", "primary key", "convolutional neural network",
                 "virtual memory", "backpropagation", "cache memory"]


def part1_unit():
    print("Part 1 — unit checks")
    ok = True
    for e in SHOULD_REJECT:
        good = (is_valid_entity(e) is False)
        ok &= good
        print(f"  reject {e!r:26} -> {'OK' if good else 'FAIL (kept)'}")
    for e in SHOULD_KEEP:
        good = (is_valid_entity(e) is True)
        ok &= good
        print(f"  keep   {e!r:26} -> {'OK' if good else 'FAIL (rejected)'}")
    # soft-hyphen recovery
    rec = normalize_entity("exam­ ple")
    print(f"  soft-hyphen 'exam<shy> ple' -> {rec!r} "
          f"{'OK' if rec == 'example' else 'FAIL'}")
    ok &= (rec == "example")
    assert ok, "some entity-quality checks failed"
    print("  all checks passed.\n")


def part2_preview():
    print("Part 2 — junk currently in the graphs (removed on re-index)")
    total = 0
    for dom in DOMAINS:
        path = os.path.join(INDEX_DIR, f"{dom}_graph.pkl")
        if not os.path.exists(path):
            print(f"  {dom}: (no graph)")
            continue
        G = pickle.load(open(path, "rb"))
        rejects = [n for n in G.nodes() if not is_valid_entity(n)]
        total += len(rejects)
        print(f"  {dom:32} {len(rejects):>4} of {G.number_of_nodes()} "
              f"nodes now rejected")
    print(f"\n  total junk nodes across domains: {total}")
    print("  Re-index to apply, then re-run retrieval_eval.py / "
          "retrieval_health.py to measure the cov@node lift.")


if __name__ == "__main__":
    part1_unit()
    part2_preview()
