# ============================================================
# scripts/test_routing_margin.py
# GraphRAG 1.3 test: domain-routing confidence margin.
#
# For each query, prints the top-2 domain scores and the margin,
# and the decision the pipeline would now make:
#   margin >= threshold -> single domain (confident)
#   margin <  threshold -> multi-domain  (boundary-spanning)
#
# Clear queries should show a large margin (single domain);
# boundary-spanning queries should show a small margin and flip
# to multi-domain. This validates both the margin logic and the
# threshold choice. Needs only the embedder (no graphs/generator).
#
# Run:
#   python scripts/test_routing_margin.py
# ============================================================
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.graphrag_domain_indexer import route_by_embedding
from utils.embedder import Embedder

try:
    from utils.config import cfg
    MT = cfg.get("routing", {}).get("margin_threshold", 0.05)
except Exception:
    MT = 0.05

CLEAR = [
    "Explain how CNNs use parameter sharing",
    "Describe the ACID properties of database transactions",
    "How does pipelining improve CPU instruction throughput?",
    "What is the difference between Agile and Waterfall methodologies?",
    "Explain virtual memory and demand paging",
]

BOUNDARY = [
    "Performance implications of database indexing",          # DB vs CA
    "How does the CPU cache affect query execution speed?",    # DB vs CA
    "Memory management and caching across OS and hardware",    # OS vs CA
]


def decide(scores):
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (d1, s1), (d2, s2) = ranked[0], ranked[1]
    margin = round(s1 - s2, 3)
    return d1, s1, d2, s2, margin, (margin < MT)


def run(embedder, label, queries):
    print(f"\n===== {label} =====")
    for q in queries:
        scores = route_by_embedding(q, embedder, return_all_scores=True)
        d1, s1, d2, s2, margin, amb = decide(scores)
        decision = "MULTI-DOMAIN (ambiguous)" if amb else f"single -> {d1}"
        print(f"Q: {q}")
        print(f"   top: {d1}={s1}  runner-up: {d2}={s2}  "
              f"margin={margin}  => {decision}\n")


def main():
    print(f"margin_threshold = {MT}")
    embedder = Embedder()
    run(embedder, "CLEAR queries (expect large margin -> single)", CLEAR)
    run(embedder, "BOUNDARY queries (expect small margin -> multi)", BOUNDARY)
    print("Check: clear queries stay single-domain; boundary queries")
    print("flip to multi-domain. If the split looks wrong, we tune")
    print("margin_threshold in config.yml (routing.margin_threshold).")


if __name__ == "__main__":
    main()
