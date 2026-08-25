# ============================================================
# scripts/retrieval_health.py
# Label-free retrieval health check: run ANY set of queries and
# flag the ones where retrieval is weak, without a gold set and
# without the (biased) LLM judge.
#
# Two intrinsic signals per query:
#   top_seed  : best query->concept-node similarity. Low means the
#               query didn't land on a real concept (generic/weak
#               seeds like "proper use", "additional consideration").
#   mean_rel  : mean cosine of the retrieved chunks to the query.
#               Low means the retrieved text isn't on-topic.
# A query is flagged WEAK if either signal is below its threshold.
#
# Usage:
#   python scripts/retrieval_health.py
#   python scripts/retrieval_health.py "your query here"
# ============================================================
import os
import sys
from statistics import mean

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.graphrag_domain_indexer import route_by_embedding, load_domain_graph
from graphrag.graphrag_retriever import retrieve_graph_context
from utils.embedder import Embedder

PARAMS = dict(top_k_nodes=5, neighbour_depth=2, max_context_chunks=7,
              seed_ratio=0.4, rerank=True, degree_penalty=0.5, depth2_cap=50)

# thresholds — tune against the printed numbers
SEED_MIN = 0.55
REL_MIN  = 0.65

# A deliberately mixed set: strong topics, the known-weak normalisation
# query, the CA pipelining weak spot, and a couple of off-corpus probes.
DEFAULT_QUERIES = [
    "Explain how an inner join combines rows from two tables in SQL",
    "What are the ACID properties of a database transaction?",
    "Explain how normalisation reduces data redundancy in relational databases",
    "What is backpropagation in a neural network?",
    "Explain how a semaphore prevents race conditions",
    "How does pipelining improve CPU instruction throughput?",
    "Describe the difference between TCP and UDP protocols",
    "Explain the CAP theorem in distributed systems",
]


def check(embedder, query):
    domain = route_by_embedding(query, embedder)     # auto-route
    index, graph, nodes = load_domain_graph(domain)
    res = retrieve_graph_context(query, index, graph, nodes, embedder, **PARAMS)
    seeds = res.get("seed_nodes", [])
    top_seed = max((s.get("score", 0.0) for s in seeds), default=0.0)
    rels = [c.get("relevance", 0.0) for c in res.get("chunk_pool", [])
            if isinstance(c.get("relevance"), float)]
    mrel = mean(rels) if rels else 0.0
    weak = (top_seed < SEED_MIN) or (mrel < REL_MIN)
    return domain, top_seed, mrel, [s["entity"] for s in seeds], weak


def main():
    queries = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_QUERIES
    print("Loading embedder...")
    embedder = Embedder()
    print(f"\nthresholds: top_seed >= {SEED_MIN}, mean_rel >= {REL_MIN}\n")
    print(f"{'flag':6} {'top_seed':8} {'mean_rel':8}  {'domain':22} query")
    print("-" * 92)
    for q in queries:
        domain, top_seed, mrel, seed_names, weak = check(embedder, q)
        flag = "WEAK" if weak else "ok"
        print(f"{flag:6} {top_seed:<8.2f} {mrel:<8.2f}  {domain:22} {q[:40]}")
        print(f"       seeds: {seed_names}")
    print("\nWEAK rows are where retrieval landed on generic/weak concepts")
    print("or off-topic chunks — inspect those with scripts/test_reranking.py.")


if __name__ == "__main__":
    main()
