# ============================================================
# scripts/test_depth2_cap.py
# Cycle-2 validation: does depth2_cap=50 generalise ACROSS domains?
#
# Tuning a cap on one domain isn't a generalisation claim. Graph
# density varies a lot (deep_learning ~6k nodes vs database ~27k),
# so both the uncapped pool size AND whether cap=50 recovers the
# top-relevance chunk can differ per domain.
#
# For each domain this compares cap=50 vs uncapped (rerank on) and
# reports whether cap=50 keeps the same top neighbour relevance.
#
#   MATCH = YES  -> cap=50 loses nothing on that domain
#   MATCH = NO   -> cap=50 is too tight there; edge-weight capping
#                   is the wrong mechanism -> use relevance-ranked
#                   depth-2 traversal (Option C) instead.
#
# Run:
#   python scripts/test_depth2_cap.py
# ============================================================
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.graphrag_indexer import load_graphrag_index
from graphrag.graphrag_retriever import retrieve_graph_context
from utils.embedder import Embedder

BASE_DIR  = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

HARD = dict(top_k_nodes=5, neighbour_depth=2,
            max_context_chunks=7, seed_ratio=0.4)
BIG  = 10 ** 9   # uncapped baseline
CAP  = 50        # the value we're validating

# One representative HARD query per domain. Adjust the query text
# (or the index name) if your files differ — Windows is case-
# insensitive, so graphrag_deep_learning matches your file.
DOMAINS = {
    "graphrag_deep_learning":
        "How does parameter sharing reduce the number of parameters in CNNs?",
    "graphrag_database":
        "Explain how an inner join combines rows from two tables in SQL",
    "graphrag_software_engineering":
        "Explain the difference between black-box and white-box testing",
    "graphrag_operating_systems":
        "How does paging translate a virtual address to a physical address?",
    "graphrag_computer_architecture":
        "How does pipelining improve CPU instruction throughput?",
}


def top_nb(res):
    rels = [c.get("relevance", 0.0)
            for c in res.get("chunk_pool", []) if c["type"] == "neighbour"]
    return round(max(rels), 3) if rels else None


def main():
    print("Loading embedder...")
    embedder = Embedder()

    header = (f"{'domain':>32} | {'nodes':>7} | {'uncap_pool':>10} | "
              f"{'cap50_pool':>10} | {'uncap_top':>9} | {'cap50_top':>9} | match")
    print("\n" + header)
    print("-" * len(header))

    for domain, query in DOMAINS.items():
        try:
            index, graph, nodes = load_graphrag_index(INDEX_DIR, domain)
        except Exception as e:
            print(f"{domain:>32} |  SKIPPED (load failed: {e})")
            continue

        uncap = retrieve_graph_context(
            query, index, graph, nodes, embedder,
            rerank=True, depth2_cap=BIG, **HARD)
        cap = retrieve_graph_context(
            query, index, graph, nodes, embedder,
            rerank=True, depth2_cap=CAP, **HARD)

        u_top, c_top = top_nb(uncap), top_nb(cap)
        match = "YES" if u_top == c_top else "NO"
        print(f"{domain:>32} | {len(nodes):>7} | "
              f"{len(uncap['neighbour_nodes']):>10} | "
              f"{len(cap['neighbour_nodes']):>10} | "
              f"{str(u_top):>9} | {str(c_top):>9} | {match}")

    print("-" * len(header))
    print("\nIf every row is MATCH=YES  -> cap=50 generalises; commit it.")
    print("If any row is  MATCH=NO   -> cap=50 is too tight there;")
    print("  we switch to relevance-ranked depth-2 (Option C) instead.")


if __name__ == "__main__":
    main()
