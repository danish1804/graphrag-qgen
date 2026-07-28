# ============================================================
# evaluation/retrieval_eval.py
# GraphRAG 1.5: judge-independent retrieval evaluation.
#
# Measures retrieval quality WITHOUT the LLM judge, so a retrieval
# change (alias merging, degree penalty, re-index, ...) can be
# scored directly and cheaply, decoupled from generation quality.
#
# Metrics per query:
#   cov@node : fraction of expected entities present among the
#              retrieved graph nodes (seed + neighbour), matched
#              via normalize_entity (survives the alias re-index).
#   cov@text : fraction of expected concepts that appear in the
#              retrieved CHUNK TEXT that reaches the prompt. This is
#              the primary metric — it credits coverage of a concept
#              even if no node is named exactly like it.
#   mean_rel : mean cosine of the final chunks to the query.
#
# Run (uses the current index + retrieval defaults):
#   python evaluation/retrieval_eval.py
# Baseline before the index-time batch, then re-run after the
# re-index and compare the aggregate numbers.
# ============================================================
import os
import sys
import json
import re
from statistics import mean

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graphrag.graphrag_indexer import normalize_entity
from graphrag.graphrag_domain_indexer import load_domain_graph
from graphrag.graphrag_retriever import retrieve_graph_context
from utils.embedder import Embedder

GOLD_PATH = os.path.join(_ROOT, "evaluation", "retrieval_gold.json")

PARAMS = dict(top_k_nodes=5, neighbour_depth=2,
              max_context_chunks=7, seed_ratio=0.4,
              rerank=True, degree_penalty=0.5, depth2_cap=50)


def _norm_set(entities):
    return {normalize_entity(e) for e in entities if e}


def _in_text(term, text):
    # word-ish substring match, tolerant of plural/spacing
    t = re.sub(r"\s+", " ", term.lower().strip())
    return t in text or (t.endswith("s") is False and (t + "s") in text)


def evaluate(gold_path=GOLD_PATH):
    gold = json.load(open(gold_path, encoding="utf-8"))
    print("Loading embedder...")
    embedder = Embedder()

    per_domain = {}
    for domain, items in gold.items():
        if domain.startswith("_"):
            continue
        index, graph, nodes = load_domain_graph(domain)
        rows = []
        print(f"\n===== {domain} =====")
        for it in items:
            q = it["query"]
            expected_raw = it["expected_entities"]
            expected = _norm_set(expected_raw)
            res = retrieve_graph_context(q, index, graph, nodes, embedder,
                                         **PARAMS)
            retrieved = _norm_set(
                [s["entity"] for s in res.get("seed_nodes", [])] +
                [n["entity"] for n in res.get("neighbour_nodes", [])])
            text = " ".join(c["text"] for c in
                            res.get("chunk_pool", [])).lower()
            rels = [c.get("relevance", 0.0)
                    for c in res.get("chunk_pool", [])
                    if isinstance(c.get("relevance"), float)]

            cov_node = len(expected & retrieved) / len(expected) if expected else 0
            in_text = [e for e in expected_raw if _in_text(e, text)]
            cov_text = len(in_text) / len(expected_raw) if expected_raw else 0
            mrel = round(mean(rels), 3) if rels else None
            missed = [e for e in expected_raw if e not in in_text]
            rows.append((cov_node, cov_text, mrel))
            print(f"  {q[:56]:58} cov@node={cov_node:.2f} "
                  f"cov@text={cov_text:.2f} mean_rel={mrel}")
            if missed:
                print(f"      not in retrieved text: {missed}")
        per_domain[domain] = rows

    print("\n" + "=" * 62)
    print("SUMMARY (mean per domain)")
    print("=" * 62)
    all_n, all_t, all_m = [], [], []
    for domain, rows in per_domain.items():
        cn = mean(r[0] for r in rows)
        ct = mean(r[1] for r in rows)
        ms = [r[2] for r in rows if r[2] is not None]
        mr = mean(ms) if ms else 0
        all_n += [r[0] for r in rows]; all_t += [r[1] for r in rows]; all_m += ms
        print(f"  {domain:24} cov@node={cn:.2f}  cov@text={ct:.2f}  "
              f"mean_rel={mr:.3f}")
    print("-" * 62)
    print(f"  {'OVERALL':24} cov@node={mean(all_n):.2f}  "
          f"cov@text={mean(all_t):.2f}  "
          f"mean_rel={(mean(all_m) if all_m else 0):.3f}")
    return per_domain


if __name__ == "__main__":
    evaluate()
