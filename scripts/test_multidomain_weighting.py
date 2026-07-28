# ============================================================
# scripts/test_multidomain_weighting.py
# GraphRAG 1.6 test: weight the multi-domain chunk budget by
# routing confidence.
#
# Before: every qualifying domain (score >= threshold) got the SAME
# chunk budget, so a domain at 0.51 diluted one at 0.85 equally.
# Now the total budget is split in proportion to each domain's score.
#
# For a couple of boundary-spanning queries this prints the
# qualifying domains, the OLD equal split vs the NEW weighted split,
# and the chunks each domain actually returns. Needs the embedder +
# domain graphs (no generator).
#
# Run:
#   python scripts/test_multidomain_weighting.py
# ============================================================
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.graphrag_domain_indexer import (
    route_by_embedding, load_domain_graph)
from graphrag.graphrag_retriever import retrieve_graph_context
from utils.embedder import Embedder
from utils.config import cfg

hard      = cfg["difficulty"]["hard"]
CPD       = hard.get("chunks_per_domain", hard.get("max_chunks", 7))
THRESHOLD = hard.get("domain_score_threshold", 0.5)
TOPK      = hard.get("top_k_nodes", 5)
DEPTH     = hard.get("neighbour_depth", 2)
SR        = hard.get("seed_ratio", 0.4)

QUERIES = [
    "Performance implications of database indexing",   # DB + CA
    "Explain virtual memory and demand paging",         # OS + CA (+ DB)
]


def weighted_budgets(qualifying):
    total = CPD * len(qualifying)
    ssum = sum(qualifying.values()) or 1.0
    return {d: max(1, round(total * s / ssum)) for d, s in qualifying.items()}


def main():
    embedder = Embedder()
    for q in QUERIES:
        print("\n" + "=" * 72)
        print("QUERY:", q)
        scores = route_by_embedding(q, embedder, return_all_scores=True)
        qualifying = {d: s for d, s in scores.items() if s >= THRESHOLD}

        if len(qualifying) <= 1:
            print(f"  only {len(qualifying)} domain >= {THRESHOLD} "
                  f"-> single-domain path, 1.6 not exercised")
            continue

        budgets = weighted_budgets(qualifying)
        print(f"  qualifying (>= {THRESHOLD}): {qualifying}")
        print(f"  OLD  equal   : {CPD} each  (total {CPD * len(qualifying)})")
        print(f"  NEW  weighted: {budgets}  (total {sum(budgets.values())})")
        for d in qualifying:
            index, graph, nodes = load_domain_graph(d)
            res = retrieve_graph_context(
                q, index, graph, nodes, embedder,
                top_k_nodes=TOPK, neighbour_depth=DEPTH,
                max_context_chunks=budgets[d], seed_ratio=SR)
            print(f"    {d:24} budget={budgets[d]:<3} -> "
                  f"{len(res.get('chunk_pool', []))} chunks returned")

    print("\nThe dominant domain should get a bigger share than a")
    print("barely-qualifying one (vs the old equal split).")


if __name__ == "__main__":
    main()
