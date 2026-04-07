# ============================================================
# compare_domain_graphrag.py
# Tests domain-routed GraphRAG against standard RAG
# across all four domains
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(__file__))

from src.pipeline             import Pipeline
from src.graphrag_pipeline    import GraphRAGPipeline
from src.graphrag_domain_indexer import load_domain_graph

BASE_DIR = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"

# One test per domain
TEST_CASES = [
    {
        "domain":    "deep_learning",
        "std_index": "source_1",
        "learning_objective": "Explain how parameter sharing "
                              "reduces learnable weights in CNNs",
        "difficulty":    "medium",
        "question_type": "conceptual"
    },
    {
        "domain":    "database",
        "std_index": None,  # use multi index
        "learning_objective": "How does a hash table handle "
                              "collisions in database indexing?",
        "difficulty":    "hard",
        "question_type": "application"
    },
    {
        "domain":    "software_engineering",
        "std_index": None,
        "learning_objective": "What is the difference between "
                              "Agile and Waterfall methodologies?",
        "difficulty":    "medium",
        "question_type": "conceptual"
    },
    {
        "domain":    "operating_systems",
        "std_index": None,
        "learning_objective": "Explain how virtual memory "
                              "management works",
        "difficulty":    "hard",
        "question_type": "application"
    },
]


def run_domain_comparison():
    print("=" * 70)
    print("DOMAIN-ROUTED GRAPHRAG vs STANDARD RAG")
    print("=" * 70)

    # Load standard pipeline
    print("\nLoading models (shared between both pipelines)...")
    std_pipeline = Pipeline()
    std_pipeline.load_models()

    # Share models with GraphRAG pipeline
    graph_pipeline = GraphRAGPipeline()
    graph_pipeline.embedder     = std_pipeline.embedder
    graph_pipeline.model        = std_pipeline.model
    graph_pipeline.tokenizer    = std_pipeline.tokenizer
    graph_pipeline._models_loaded = True

    print("\nModels ready. Running comparison...\n")

    for tc in TEST_CASES:
        domain = tc["domain"]
        obj    = tc["learning_objective"]

        print(f"\n{'='*70}")
        print(f"DOMAIN   : {domain.upper()}")
        print(f"OBJECTIVE: {obj}")
        print(f"DIFFICULTY: {tc['difficulty']} | "
              f"TYPE: {tc['question_type']}")
        print("=" * 70)

        # ── Standard RAG ──────────────────────────────────
        # For domains without a single-doc index
        # use the multi-document index
        from src.multi_pipeline import MultiPipeline
        multi_pipe = MultiPipeline()
        multi_pipe.embedder     = std_pipeline.embedder
        multi_pipe.model        = std_pipeline.model
        multi_pipe.tokenizer    = std_pipeline.tokenizer
        multi_pipe._models_loaded = True
        multi_pipe.load_index()

        print("\nGenerating with Standard RAG (multi-doc)...")
        std_result = multi_pipe.generate(
            learning_objective = obj,
            difficulty         = tc["difficulty"],
            question_type      = tc["question_type"],
            top_k              = 3
        )

        # ── Domain GraphRAG ────────────────────────────────
        print(f"Generating with GraphRAG ({domain} domain)...")
        index, graph, nodes = load_domain_graph(domain)
        graph_pipeline.index  = index
        graph_pipeline.graph  = graph
        graph_pipeline.nodes  = nodes

        graph_result = graph_pipeline.generate(
            learning_objective = obj,
            difficulty         = tc["difficulty"],
            question_type      = tc["question_type"],
            top_k_nodes        = 3,
            neighbour_depth    = 1
        )

        # ── Display ────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"STANDARD RAG:")
        print(f"  Sources  : {std_result.get('sources_used', [])}")
        print(f"  Score    : {std_result.get('avg_retrieval_score', 'N/A')}")
        print(f"  Question : {std_result['question']}")

        print(f"\nDOMAIN GRAPHRAG ({domain}):")
        print(f"  Seeds    : {graph_result.get('seed_nodes', [])}")
        print(f"  Neighbours: {graph_result.get('neighbour_nodes', [])[:3]}")
        print(f"  Relations : {graph_result.get('relationships', [])[:2]}")
        print(f"  Question : {graph_result['question']}")
        print(f"{'─'*70}")

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_domain_comparison()