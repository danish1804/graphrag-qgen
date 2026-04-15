# ============================================================
# src/graphrag_pipeline.py
# GraphRAG Pipeline — connects graph retrieval to Qwen 2.5
#
# Difference from pipeline.py:
#   pipeline.py         -> flat chunk retrieval (standard RAG)
#   graphrag_pipeline.py -> graph-aware retrieval (GraphRAG)
#
# Scope is set by what index you load:
#   load_existing_index() -> single document graph
#   load_domain_index()   -> domain-specific graph (multi-doc)
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.document_loader  import load_and_chunk
from graphrag.graphrag_indexer import (build_graphrag_index,
                                   load_graphrag_index)
from graphrag.graphrag_retriever import retrieve_graph_context
from utils.embedder          import Embedder
from utils.generator         import load_model, generate_question
from utils.config import cfg


BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
DATA_DIR   = os.path.join(BASE_DIR, "data", "tested")


class GraphRAGPipeline:

    def __init__(self):
        self.index        = None
        self.graph        = None
        self.nodes        = None
        self.embedder     = None
        self.model        = None
        self.tokenizer    = None
        self.graph_name   = None
        self._models_loaded = False

    # ----------------------------------------------------------
    # STAGE 1 — Load models (call once at startup)
    # ----------------------------------------------------------
    def load_models(self):
        """Load embedding model and Qwen 2.5 into memory."""
        if self._models_loaded:
            print("Models already loaded.")
            return

        print("\n[GraphRAGPipeline] Loading embedding model...")
        self.embedder = Embedder()

        print("\n[GraphRAGPipeline] Loading Qwen 2.5 7B...")
        self.model, self.tokenizer = load_model()

        self._models_loaded = True
        print("\n[GraphRAGPipeline] All models ready.")

    # ----------------------------------------------------------
    # STAGE 2 — Ingest a document
    # ----------------------------------------------------------
    def ingest(self, file_path: str) -> dict:
        """
        Load a PDF/PPTX, build knowledge graph, embed nodes,
        save GraphRAG index.

        Steps:
        1. Load and chunk document
        2. Extract entities + build graph (two-pass)
        3. Embed graph nodes
        4. Save FAISS index + graph + metadata
        """
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before ingest()")

        file_name = os.path.basename(file_path)
        self.graph_name = os.path.splitext(file_name)[0] \
                          + "_graphrag"

        print(f"\n[GraphRAGPipeline] Ingesting: {file_name}")

        # Load and chunk
        doc_result = load_and_chunk(file_path)
        chunks     = doc_result["chunks"]
        sources    = [file_name] * len(chunks)

        print(f"[GraphRAGPipeline] {len(chunks)} chunks loaded")

        # Build GraphRAG index
        summary = build_graphrag_index(
            chunks, sources,
            index_dir  = INDEX_DIR,
            graph_name = self.graph_name
        )

        # Load into memory
        self.index, self.graph, self.nodes = \
            load_graphrag_index(INDEX_DIR, self.graph_name)

        print(f"[GraphRAGPipeline] Graph ready — "
              f"{summary['graph_nodes']} nodes, "
              f"{summary['graph_edges']} edges")

        return {
            "file_name":   file_name,
            "graph_name":  self.graph_name,
            "graph_nodes": summary["graph_nodes"],
            "graph_edges": summary["graph_edges"],
            "chunks":      len(chunks)
        }
    
# ----------------------------------------------------------
# STAGE 3 — Load existing index
# ----------------------------------------------------------
    def load_existing_index(self, graph_name: str):
        """Load a previously built GraphRAG index."""
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before loading index.")

        self.graph_name = graph_name
        self.index, self.graph, self.nodes = \
            load_graphrag_index(INDEX_DIR, graph_name)

        print(f"[GraphRAGPipeline] Loaded: {graph_name}")

# ----------------------------------------------------------
# STAGE 3b — Load domain-specific index (updated method with routing)
# ----------------------------------------------------------
    def load_domain_index(self, domain_name: str):
        """
        Load a domain-specific GraphRAG index.
        Uses the domain indexer routing to select the right graph.
        """
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before loading index.")

        from graphrag.graphrag_domain_indexer import (
            DOMAIN_MAP, load_domain_graph)

        if domain_name not in DOMAIN_MAP:
            raise ValueError(
                f"Unknown domain: {domain_name}. "
                f"Choose from: {list(DOMAIN_MAP.keys())}")

        self.index, self.graph, self.nodes = \
            load_domain_graph(domain_name)
        self.graph_name = f"graphrag_{domain_name}"

        print(f"[GraphRAGPipeline] Domain loaded: {domain_name} "
            f"({len(self.nodes)} nodes)")

# ----------------------------------------------------------
# STAGE 3c — Generate with automatic domain routing
# ----------------------------------------------------------      
    # def generate_with_routing(self,
    #                       learning_objective: str,
    #                       difficulty:   str = "medium",
    #                       question_type: str = "conceptual",
    #                       top_k_nodes:  int = 3,
    #                       neighbour_depth: int = 1) -> dict:
    #     """
    #     Generate question with automatic domain routing.
    #     Selects the correct domain graph based on the
    #     learning objective keywords.
    #     """
    #     from src.graphrag_domain_indexer import (
    #         route_query_to_domain, load_domain_graph)

    #     # Route to correct domain
    #     domain = route_query_to_domain(learning_objective)
    #     print(f"[GraphRAGPipeline] Routing to domain: {domain}")

    #     # Load domain graph
    #     self.index, self.graph, self.nodes = \
    #         load_domain_graph(domain)
    #     self.graph_name = f"graphrag_{domain}"

    #     # Generate
    #     result = self.generate(
    #         learning_objective = learning_objective,
    #         difficulty         = difficulty,
    #         question_type      = question_type,
    #         top_k_nodes        = top_k_nodes,
    #         neighbour_depth    = neighbour_depth
    #     )

    #     result["domain_routed_to"] = domain
    #     return result

    def generate_with_routing(self,
                           learning_objective: str,
                           difficulty:    str = "medium",
                           question_type: str = "conceptual",
                           top_k_nodes:   int = 3,
                           neighbour_depth: int = 1) -> dict:
        """
        Route learning objective to correct domain graph
        then generate question automatically.

        Uses embedding-based routing if domain profiles are
        available, falls back to keyword routing otherwise.
        """
        if not self._models_loaded:
            raise RuntimeError("Call load_models() first.")

        from graphrag.graphrag_domain_indexer import (
            load_domain_graph, route_by_embedding)

        # Use embedding-based routing (smarter than keywords)
        domain = route_by_embedding(
            learning_objective, self.embedder)

        print(f"[GraphRAGPipeline] Routed to: {domain}")

        self.index, self.graph, self.nodes = \
            load_domain_graph(domain)
        self.graph_name = f"graphrag_{domain}"

        result = self.generate(
            learning_objective = learning_objective,
            difficulty         = difficulty,
            question_type      = question_type,
            top_k_nodes        = top_k_nodes,
            neighbour_depth    = neighbour_depth
        )

        # Move domain into metadata
        result["metadata"]["domain"] = domain
        result["metadata"]["domain_routed_to"] = domain
        return result

    # ----------------------------------------------------------
    # STAGE 4 — Generate a question
    # ----------------------------------------------------------
    def generate(self,
                 learning_objective: str,
                 difficulty:   str = "medium",
                 question_type: str = "conceptual",
                 top_k_nodes:  int = None,
                 neighbour_depth: int = None) -> dict:
        """
        Full GraphRAG pipeline:
        objective -> graph retrieval -> enriched context -> question

        Returns dict with question + graph metadata.
        """
        if self.index is None:
            raise RuntimeError(
                "No index loaded. Call ingest() or "
                "load_existing_index(), or " 
                "load_domain_index() first.")

        if not self._models_loaded:
            raise RuntimeError("Call load_models() first.")

        # Get difficulty-specific parameters from config
        diff_cfg      = cfg["difficulty"][difficulty]
        max_chunks    = diff_cfg["max_chunks"]
        top_k_nodes   = diff_cfg["top_k_nodes"]
        neighbour_depth = diff_cfg["neighbour_depth"]
        # Graph-aware retrieval
        retrieval = retrieve_graph_context(
            query           = learning_objective,
            index           = self.index,
            graph           = self.graph,
            nodes           = self.nodes,
            embedder        = self.embedder,
            top_k_nodes     = top_k_nodes,
            neighbour_depth = neighbour_depth,
            max_context_chunks = max_chunks
        )

        if not retrieval["context"]:
            return {
                "question": "Could not retrieve relevant context.",
                "error":    "No matching entities found in graph."
            }

        # Generate question using enriched context
        question = generate_question(
            context            = retrieval["context"],
            learning_objective = learning_objective,
            model              = self.model,
            tokenizer          = self.tokenizer,
            difficulty         = difficulty,
            question_type      = question_type
        )

        # Format relationships as readable strings
        rel_strings = [
            f"{r[0]} -> {r[1]}"
            for r in retrieval["relationships"][:5]
        ]

        # Clean key concepts — seed + neighbour nodes combined
        all_concepts = (
            [s["entity"] for s in retrieval["seed_nodes"]] +
            [n["entity"] for n in retrieval["neighbour_nodes"][:3]]
        )
        # Remove obvious noise from display
        clean_concepts = [
            c for c in all_concepts
            if len(c) > 3
            and not any(char in c for char in ['©', '\n', '\r'])
        ]

        # Format chunks for display — first 200 chars each
        retrieved_chunks = [
            {
                "text":   item["text"][:200].strip(),
                "source": retrieval["sources"][0]
                        if retrieval["sources"] else "unknown",
                "type":   item.get("type", "seed")
            }
            for item in retrieval.get("chunk_pool", [])
        ]

        return {
            "question":  question,
            "metadata": {
                "learning_objective":   learning_objective,
                "difficulty":           difficulty,
                "question_type":        question_type,
                "domain":               getattr(
                    self, "graph_name", "unknown"
                ).replace("graphrag_", ""),
                "sources":              retrieval["sources"],
                "retrieval_method":     "GraphRAG",
                "key_concepts":         clean_concepts,
                "concept_relationships": rel_strings,
                "retrieved_chunks":      retrieved_chunks,
            }
        }

# ============================================================
# TEST — python graphrag/graphrag_pipeline.py
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("GRAPHRAG PIPELINE TEST")
    print("=" * 60)

    pipeline = GraphRAGPipeline()
    pipeline.load_models()

    # Load the test index we already built
    print("\nLoading existing GraphRAG test index...")
    pipeline.load_existing_index("graphrag_test")

    # Test questions
    test_cases = [
        {
            "learning_objective": "Explain how parameter sharing "
                                  "reduces learnable weights in CNNs",
            "difficulty":    "medium",
            "question_type": "conceptual"
        },
        {
            "learning_objective": "How does sparse connectivity "
                                  "differ from fully connected layers?",
            "difficulty":    "hard",
            "question_type": "application"
        },
        {
            "learning_objective": "What is the purpose of "
                                  "pooling in CNNs?",
            "difficulty":    "easy",
            "question_type": "recall"
        },
    ]

    print("\n" + "=" * 60)
    print("GENERATED QUESTIONS — GraphRAG vs Standard RAG")
    print("=" * 60)

    for i, tc in enumerate(test_cases, 1):
        print(f"\nQuestion {i}:")
        print(f"  Objective  : {tc['learning_objective']}")
        print(f"  Difficulty : {tc['difficulty']}")
        print(f"  Type       : {tc['question_type']}")

        result = pipeline.generate(**tc)

        print(f"\n  Question   : {result['question']}")
        print(f"  Domain     : {result['metadata']['domain']}")
        print(f"  Key concepts: {result['metadata']['key_concepts']}")
        print(f"  Relations  : {result['metadata']['concept_relationships']}")
        print(f"  Sources    : {result['metadata']['sources']}")
        print("-" * 60)

    print("\nGraphRAG pipeline test complete.")
    print("\nNext step: add GraphRAG endpoints to main.py")


