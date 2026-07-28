# ============================================================
# scripts/test_degree_penalty.py
# GraphRAG 1.2 test: degree down-weighting of neighbour selection.
#
# Neighbour selection used raw edge weight, so generic hub terms
# (very high degree, e.g. "attribute" deg 3699) dominated. Now the
# ranking discounts edge weight by the neighbour's degree, behind a
# tunable degree_penalty (0 = old behaviour).
#
# This sweeps the penalty and reports, for the final neighbour
# chunks: their mean node degree (lower = less hub-dominated) and
# the top query relevance (rerank on, so quality should hold).
#
# Run:
#   python scripts/test_degree_penalty.py
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
DOMAIN = "graphrag_database"

HARD = dict(top_k_nodes=5, neighbour_depth=2,
            max_context_chunks=7, seed_ratio=0.4)
QUERY = "Explain how an inner join combines rows from two tables in SQL"
PENALTIES = [0.0, 0.5, 1.0]


def summarise(res, graph):
    pool = res.get("chunk_pool", [])
    nb = [c for c in pool if c["type"] == "neighbour"]
    info = [(c["entity"],
             graph.degree(c["entity"]) if c["entity"] in graph else 0,
             round(c.get("relevance", 0.0), 3))
            for c in nb]
    degs = [d for _, d, _ in info]
    mean_deg = sum(degs) / len(degs) if degs else 0
    top_rel = max((r for _, _, r in info), default=None)
    return mean_deg, top_rel, info


def main():
    print("Loading index + embedder...")
    index, graph, nodes = load_graphrag_index(INDEX_DIR, DOMAIN)
    embedder = Embedder()

    print(f"\nQUERY: {QUERY}")
    for p in PENALTIES:
        res = retrieve_graph_context(
            QUERY, index, graph, nodes, embedder,
            rerank=True, degree_penalty=p, **HARD)
        mean_deg, top_rel, info = summarise(res, graph)
        tag = "  (raw weight = old behaviour)" if p == 0.0 else ""
        print(f"\n--- degree_penalty={p}{tag} ---")
        print(f"  mean neighbour degree: {mean_deg:.0f}   top_rel: {top_rel}")
        for ent, deg, rel in info:
            print(f"    {ent!r:42} deg={deg:<6} rel={rel}")

    print("\nExpect: as penalty rises, mean neighbour degree DROPS")
    print("(hub terms demoted) while top_rel stays about the same or")
    print("improves. Pick the smallest penalty that clearly cuts hub")
    print("degree without hurting relevance.")


if __name__ == "__main__":
    main()
