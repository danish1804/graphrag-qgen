# ============================================================
# scripts/test_reranking.py
# Cycle-1 test: query-time re-ranking in
# graphrag_retriever.retrieve_graph_context
#
# Compares rerank=False (old, graph-insertion order)
# vs      rerank=True  (new, query-relevance order)
# on the same query, using your already-built domain index.
#
# Run:
#   python scripts/test_reranking.py
#
# What to look for:
#   In the AFTER list the most on-topic chunks should sit at the
#   top of each group, and off-topic neighbours (e.g. a DELETE/
#   cascade or LongString chunk for an inner-join query) should be
#   pushed down or out of the budget. The assertions confirm the
#   ordering is by relevance.
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

# If index file is named differently, change this to match
# (see the files in indexes/, e.g. graphrag_deep_Learning.*).
DOMAIN  = "graphrag_database"

# Hard params reproduce the 7-chunk, depth-2 scenario from the review.
PARAMS = dict(
    top_k_nodes=5,
    neighbour_depth=2,
    max_context_chunks=7,
    seed_ratio=0.4,
)

QUERIES = ["Explain how an inner join combines rows from two tables in SQL"]

def show(tag, res):
    print(f"\n--- {tag} ---")
    pool = res.get("chunk_pool", [])
    if not pool:
        print("  (no chunks returned)")
        return
    for i, c in enumerate(pool, 1):
        rel = c.get("relevance")
        rel_s = f"{rel:.3f}" if isinstance(rel, float) else "  -  "
        preview = c["text"][:90].strip().replace("\n", " ")
        print(f"{i}. [{c['type']:9}] rel={rel_s} "
              f"src={c.get('source','?')}  {preview!r}")


def main():
    print("Loading index + embedder (first run downloads the "
          "embedding model)...")
    index, graph, nodes = load_graphrag_index(INDEX_DIR, DOMAIN)
    embedder = Embedder()

    for q in QUERIES:
        print("\n" + "=" * 80)
        print("QUERY:", q)
        print("=" * 80)

        base = retrieve_graph_context(
            q, index, graph, nodes, embedder, rerank=False, **PARAMS)
        new = retrieve_graph_context(
            q, index, graph, nodes, embedder, rerank=True, **PARAMS)

        show("BEFORE (rerank=False, graph-insertion order)", base)
        show("AFTER  (rerank=True, query-relevance order)", new)

        # Sanity checks on the reranked result
        pool = new.get("chunk_pool", [])
        nb = [c.get("relevance", 0.0) for c in pool
              if c["type"] == "neighbour"]
        sd = [c.get("relevance", 0.0) for c in pool
              if c["type"] == "seed"]
        assert nb == sorted(nb, reverse=True), \
            "FAIL: neighbour chunks not ordered by relevance"
        assert sd == sorted(sd, reverse=True), \
            "FAIL: seed chunks not ordered by relevance"
        print("\n[OK] reranked chunks are ordered by query relevance "
              "within each group.")

    print("\nAll checks passed. Compare BEFORE vs AFTER above: the "
          "AFTER ordering should be visibly more on-topic.")


if __name__ == "__main__":
    main()
