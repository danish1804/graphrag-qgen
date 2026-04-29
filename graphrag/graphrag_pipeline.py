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
from utils.generator         import load_model, generate_question, generate_answer
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
        """Load embedding model and LLM... into memory."""
        if self._models_loaded:
            print("Models already loaded.")
            return

        print(f"\n[GraphRAGPipeline] Loading embedding model {cfg['model']['embedding_model']}...")
        self.embedder = Embedder()

        print(f"\n[GraphRAGPipeline] Loading {cfg['model']['llm_name']}...")
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


    def generate_with_multi_domain(self,
                               learning_objective: str,
                               difficulty:    str = "hard",
                               question_type: str = "application"
                               ) -> dict:
        """
        Multi-domain retrieval for hard questions.
        Samar: "For hard questions, pick domains above threshold
                and generate question from combined context."

        Flow:
            1. Get embedding scores for ALL domains
            2. Filter domains above threshold (from config)
            3. Retrieve chunks from each qualifying domain
            4. Combine contexts
            5. Single LLM call with combined context
        """
        if not self._models_loaded:
            raise RuntimeError("Call load_models() first.")

        from graphrag.graphrag_domain_indexer import (
            load_domain_graph, route_by_embedding)

        diff_cfg   = cfg["difficulty"][difficulty]
        threshold  = diff_cfg.get("domain_score_threshold", 0.5)
        chunks_per_domain = diff_cfg.get("chunks_per_domain", 3)
        seed_ratio = diff_cfg.get("seed_ratio", 0.4)

        # Step 1 — Get all domain scores
        all_scores = route_by_embedding(
            learning_objective,
            self.embedder,
            return_all_scores=True
        )

        # Step 2 — Filter qualifying domains above threshold
        qualifying = {
            domain: score
            for domain, score in all_scores.items()
            if score >= threshold
        }

        print(f"[MultiDomain] Qualifying domains "
            f"(threshold={threshold}): {qualifying}")

        # Step 3 — If only one domain qualifies, fall back
        # to standard single-domain routing
        if len(qualifying) <= 1:
            print(f"[MultiDomain] Only one domain qualifies "
                f"— falling back to single domain routing")
            return self.generate_with_routing(
                learning_objective = learning_objective,
                difficulty         = difficulty,
                question_type      = question_type
            )

        # Step 4 — Retrieve chunks from each qualifying domain
        combined_chunks   = []
        combined_sources  = []
        combined_concepts = []
        combined_relations = []
        domains_used      = []

        for domain in qualifying:
            print(f"[MultiDomain] Retrieving from: {domain}")

            # Load domain graph
            index, graph, nodes = load_domain_graph(domain)

            # Retrieve context
            retrieval = retrieve_graph_context(
                query              = learning_objective,
                index              = index,
                graph              = graph,
                nodes              = nodes,
                embedder           = self.embedder,
                top_k_nodes        = diff_cfg["top_k_nodes"],
                neighbour_depth    = diff_cfg["neighbour_depth"],
                max_context_chunks = chunks_per_domain,
                seed_ratio         = seed_ratio
            )

            if retrieval["context"]:
                combined_chunks.append(
                    f"[{domain.upper()} DOMAIN]\n"
                    f"{retrieval['context']}"
                )
                combined_sources.extend(retrieval["sources"])
                combined_concepts.extend(
                    [s["entity"] for s in retrieval["seed_nodes"]]
                )
                combined_relations.extend(
                    [f"{r[0]} -> {r[1]}"
                    for r in retrieval["relationships"][:3]]
                )
                domains_used.append(domain)

        if not combined_chunks:
            return {
                "question": "Could not retrieve context "
                            "from any domain.",
                "error": "No matching entities found."
            }

        # Step 5 — Combine all domain contexts
        full_context = "\n\n---\n\n".join(combined_chunks)

        print(f"[MultiDomain] Combined context from "
            f"{len(domains_used)} domains: {domains_used}")

        # Step 6 — Generate question from combined context
        question = generate_question(
            context            = full_context,
            learning_objective = learning_objective,
            model              = self.model,
            tokenizer          = self.tokenizer,
            difficulty         = difficulty,
            question_type      = question_type
        )

        # Step 7 — Generate sample answer
        sample_answer = generate_answer(
            context    = full_context,
            question   = question,
            model      = self.model,
            tokenizer  = self.tokenizer,
            difficulty = difficulty
        )

        # Step 8 — Format retrieved chunks for metadata
        retrieved_chunks = {
            f"chunk_{i+1}": {
                "text":   item["text"][:200].strip(),
                "source": item.get("source", "unknown"),
                "type":   item.get("type", "seed"),
                "domain": domains_used[
                    min(i // chunks_per_domain, len(domains_used)-1)
                ]
            }
            for i, item in enumerate(
                retrieval.get("chunk_pool", [])
            )
        }

        # Format concepts
        clean_concepts = [
            c for c in combined_concepts
            if len(c) > 3
            and not any(
                char in c for char in ['©', '\n', '\r'])
        ]

        return {
            "question": question,
            "metadata": {
                "learning_objective":    learning_objective,
                "difficulty":            difficulty,
                "question_type":         question_type,
                "domain":                "multi_domain",
                "domains_used":          domains_used,
                "domain_scores":         qualifying,
                "sources":               list(set(combined_sources)),
                "retrieval_method":      "GraphRAG-MultiDomain",
                "key_concepts":          clean_concepts[:8],
                "concept_relationships": combined_relations[:5],
                "retrieved_chunks":      retrieved_chunks,
                "sample_answer":         sample_answer,
                "domain_score_threshold": threshold,
            }
        }
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
        seed_ratio      = diff_cfg.get("seed_ratio", 0.6)
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
            max_context_chunks = max_chunks,
            seed_ratio = seed_ratio
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

        # Generate sample answer using same context
        sample_answer = generate_answer(
            context    = retrieval["context"],
            question   = question,
            model      = self.model,
            tokenizer  = self.tokenizer,
            difficulty = difficulty
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
        # Replace with:
        retrieved_chunks = {
            f"chunk_{i+1}": {
                "text":   item["text"][:200].strip(),
                "source": item.get("source", "unknown"),
                "type":   item.get("type", "seed")
            }
            for i, item in enumerate(retrieval.get("chunk_pool", []))
        }

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
                "sample_answer":         sample_answer,
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


