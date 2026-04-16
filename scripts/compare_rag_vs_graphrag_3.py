# ============================================================
# compare_domain_graphrag.py
# Three-way comparison:
#   A) Standard RAG — all 22 docs combined
#   B) Standard RAG — domain docs only
#   C) Domain GraphRAG — graph retrieval on domain docs
#
# This isolates TWO separate improvements:
#   A->B: benefit of domain filtering alone
#   B->C: benefit of graph structure on top of filtering
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(__file__))

from standard_rag.pipeline              import Pipeline
from standard_rag.multi_pipeline        import MultiPipeline
from graphrag.graphrag_pipeline     import GraphRAGPipeline
from graphrag.graphrag_domain_indexer import (
    DOMAIN_MAP, load_domain_graph, route_query_to_domain)
from utils.embedder              import build_index, load_index
from utils.document_loader       import load_and_chunk

BASE_DIR   = os.getenv("GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")

# Test cases — one per domain
TEST_CASES = [
    {
        "domain":    "deep_learning",
        "objective": "Explain how parameter sharing reduces "
                     "learnable weights in CNNs",
        "difficulty":    "medium",
        "question_type": "conceptual",
        "domain_files":  ["source_1.pdf"]
    },
    {
        "domain":    "database",
        "objective": "How does a hash table handle "
                     "collisions in database indexing?",
        "difficulty":    "hard",
        "question_type": "application",
        "domain_files":  [
            "source_7.pdf", "source_8.pdf", "source_9.pdf",
            "source_10.pdf", "source_13.pdf",
            "source_14.pdf", "source_22.pdf"
        ]
    },
    {
        "domain":    "software_engineering",
        "objective": "What is the difference between "
                     "Agile and Waterfall methodologies?",
        "difficulty":    "medium",
        "question_type": "conceptual",
        "domain_files":  [
            "source_2.pdf", "source_3.pdf", "source_4.pdf",
            "source_5.pdf", "source_11.pdf", "source_12.pdf",
            "source_15.pdf", "source_16.pdf", "source_18.pdf",
            "source_19.pdf", "source_20.pdf", "source_21.pdf"
        ]
    },
    {
        "domain":    "operating_systems",
        "objective": "Explain how virtual memory "
                     "management works",
        "difficulty":    "hard",
        "question_type": "application",
        "domain_files":  ["source_6.pdf", "source_17.pdf"]
    },
]


def build_domain_flat_index(
        domain_files: list,
        domain_name:  str
) -> tuple:
    """
    Build a flat FAISS index (standard RAG style)
    from domain-specific documents only.

    This is option B — domain filtering without graph structure.
    """
    index_name = f"flat_{domain_name}"
    index_path = os.path.join(
        INDEX_DIR, f"{index_name}.faiss")

    # Load and chunk all domain docs
    all_chunks  = []
    all_sources = []

    for fname in domain_files:
        fpath = os.path.join(TESTED_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found")
            continue
        result  = load_and_chunk(fpath)
        all_chunks.extend(result["chunks"])
        all_sources.extend([fname] * len(result["chunks"]))

    if not all_chunks:
        raise ValueError(
            f"No chunks loaded for domain {domain_name}")

    print(f"  Domain flat index: {len(all_chunks)} chunks "
          f"from {len(domain_files)} docs")

    # Build flat FAISS index
    summary = build_index(
        all_chunks, INDEX_DIR, index_name)

    return index_name, all_chunks


def run_three_way_comparison():

    print("=" * 70)
    print("THREE-WAY COMPARISON")
    print("A) Standard RAG  — all 22 docs")
    print("B) Standard RAG  — domain docs only")
    print("C) Domain GraphRAG — graph on domain docs")
    print("=" * 70)
    print("\nIsolates:")
    print("  A->B : benefit of domain filtering alone")
    print("  B->C : benefit of graph structure on top of filtering")
    print("=" * 70)

    # Load models once
    print("\nLoading models...")
    base_pipeline = Pipeline()
    base_pipeline.load_models()
    embedder   = base_pipeline.embedder
    model      = base_pipeline.model
    tokenizer  = base_pipeline.tokenizer
    print("Models ready.\n")

    # Pipeline A: Standard RAG all docs
    print("Loading Pipeline A: Multi-document standard RAG...")
    multi_pipe = MultiPipeline()
    multi_pipe.embedder     = embedder
    multi_pipe.model        = model
    multi_pipe.tokenizer    = tokenizer
    multi_pipe._models_loaded = True
    multi_pipe.load_index()

    # Pipeline C: Domain GraphRAG
    graph_pipe = GraphRAGPipeline()
    graph_pipe.embedder     = embedder
    graph_pipe.model        = model
    graph_pipe.tokenizer    = tokenizer
    graph_pipe._models_loaded = True

    # Run comparison for each test case
    all_results = []

    for tc in TEST_CASES:
        domain = tc["domain"]
        obj    = tc["objective"]
        diff   = tc["difficulty"]
        qtype  = tc["question_type"]

        print(f"\n{'='*70}")
        print(f"DOMAIN    : {domain.upper()}")
        print(f"OBJECTIVE : {obj}")
        print(f"DIFFICULTY: {diff} | TYPE: {qtype}")
        print("=" * 70)

        # A: Standard RAG all docs
        print("\n[A] Standard RAG — all 22 docs...")
        result_a = multi_pipe.generate(
            learning_objective = obj,
            difficulty         = diff,
            question_type      = qtype,
            top_k              = 3
        )

        # B: Standard RAG domain only
        print(f"[B] Standard RAG — {domain} docs only...")
        try:
            index_name, domain_chunks = \
                build_domain_flat_index(
                    tc["domain_files"], domain)

            # Load the flat domain index into base pipeline
            domain_index, domain_chunk_list = load_index(
                INDEX_DIR, index_name)
            base_pipeline.index      = domain_index
            base_pipeline.chunks     = domain_chunk_list
            base_pipeline.index_name = index_name

            result_b = base_pipeline.generate(
                learning_objective = obj,
                difficulty         = diff,
                question_type      = qtype,
                top_k              = 3
            )
        except Exception as e:
            print(f"  ERROR in B: {e}")
            result_b = {
                "question": f"Error: {e}",
                "avg_retrieval_score": 0
            }

        # C: Domain GraphRAG
        print(f"[C] Domain GraphRAG — {domain} graph...")
        try:
            g_index, g_graph, g_nodes = \
                load_domain_graph(domain)
            graph_pipe.index  = g_index
            graph_pipe.graph  = g_graph
            graph_pipe.nodes  = g_nodes

            result_c = graph_pipe.generate(
                learning_objective = obj,
                difficulty         = diff,
                question_type      = qtype,
                top_k_nodes        = 3,
                neighbour_depth    = 1
            )
        except Exception as e:
            print(f"  ERROR in C: {e}")
            result_c = {
                "question": f"Error: {e}",
                "seed_nodes": [],
                "neighbour_nodes": [],
                "relationships": []
            }

        # Display results
        print(f"\n{'─'*70}")
        print(f"[A] STANDARD RAG — ALL DOCS:")
        print(f"    Sources : {result_a.get('sources_used', [])[:3]}")
        print(f"    Score   : {result_a.get('avg_retrieval_score', 'N/A')}")
        print(f"    Question: {result_a['question']}")

        print(f"\n[B] STANDARD RAG — {domain.upper()} ONLY:")
        print(f"    Score   : {result_b.get('avg_retrieval_score', 'N/A')}")
        print(f"    Question: {result_b['question']}")

        print(f"\n[C] DOMAIN GRAPHRAG — {domain.upper()}:")
        print(f"    Seeds   : {result_c.get('seed_nodes', [])}")
        print(f"    Neighbours: "
              f"{result_c.get('neighbour_nodes', [])[:3]}")
        print(f"    Relations : "
              f"{result_c.get('relationships', [])[:2]}")
        print(f"    Question: {result_c['question']}")
        print(f"{'─'*70}")

        all_results.append({
            "domain":     domain,
            "objective":  obj,
            "question_a": result_a["question"],
            "question_b": result_b["question"],
            "question_c": result_c["question"],
            "score_a":    result_a.get(
                "avg_retrieval_score", 0),
            "score_b":    result_b.get(
                "avg_retrieval_score", 0),
            "graph_nodes_c": result_c.get("seed_nodes", []),
            "relations_c":   result_c.get("relationships", [])
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — SCORE COMPARISON (retrieval score)")
    print("Higher = more relevant context retrieved")
    print("=" * 70)
    print(f"{'Domain':<22} {'A: All docs':<14} "
          f"{'B: Domain only':<16} {'Improvement A->B'}")
    print("-" * 70)
    for r in all_results:
        sa   = r["score_a"]
        sb   = r["score_b"]
        diff = sb - sa if isinstance(sb, float) else 0
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "->")
        print(f"  {r['domain']:<20} "
              f"{str(round(sa,4)):<14} "
              f"{str(round(sb,4)):<16} "
              f"{arrow} {abs(round(diff,4))}")

    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print("=" * 70)
    print("A->B: Does domain filtering improve retrieval score?")
    print("B->C: Does graph structure produce richer questions?")
    print("     (Cannot measure with score — requires human eval)")
    print("     Compare question complexity in output above.")

    return all_results


if __name__ == "__main__":
    results = run_three_way_comparison()
    print("\nThree-way comparison complete.")
    print("Results ready for capstone report evaluation section.")