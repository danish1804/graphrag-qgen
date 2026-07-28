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
from utils.generator         import (
    load_model, generate_question, generate_answer, get_generation_trace)
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
        self.judge_model      = None
        self.judge_tokenizer  = None
        self._judge_loaded    = False

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

    # def load_judge_model(self):
    #     """Load judge model — called once at server startup."""
    #     if self._judge_loaded:
    #         print("[GraphRAGPipeline] Judge already loaded.")
    #         return

    #     from utils.judge import load_judge
    #     print("\n[GraphRAGPipeline] Loading judge model...")
    #     self.judge_model, self.judge_tokenizer, device = \
    #         load_judge()
    #     self._judge_loaded = True
    #     print(f"[GraphRAGPipeline] Judge ready on {device}.")

    def load_judge_model(self):
        """
        Reuse generator model as judge.
        Saves VRAM — runs on GPU.
        """
        if self._judge_loaded:
            return

        if self._models_loaded and self.model is not None:
            print("[GraphRAGPipeline] Reusing Qwen 7B "
                "as judge — GPU ready.")
            self.judge_model     = self.model
            self.judge_tokenizer = self.tokenizer
            self._judge_loaded   = True
            print("[GraphRAGPipeline] Judge ready on GPU.")
        else:
            from utils.judge import load_judge
            self.judge_model, self.judge_tokenizer, device = \
                load_judge()
            self._judge_loaded = True
            print(f"[GraphRAGPipeline] Judge ready on {device}.")

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

    # def generate_with_routing(self,
    #                        learning_objective: str,
    #                        difficulty:    str = "medium",
    #                        question_type: str = "conceptual",
    #                        top_k_nodes:   int = 3,
    #                        neighbour_depth: int = 1) -> dict:
    #     """
    #     Route learning objective to correct domain graph
    #     then generate question automatically.

    #     Uses embedding-based routing if domain profiles are
    #     available, falls back to keyword routing otherwise.
    #     """
    #     if not self._models_loaded:
    #         raise RuntimeError("Call load_models() first.")

    #     from graphrag.graphrag_domain_indexer import (
    #         load_domain_graph, route_by_embedding)

    #     # Use embedding-based routing (smarter than keywords)
    #     domain = route_by_embedding(
    #         learning_objective, self.embedder)

    #     print(f"[GraphRAGPipeline] Routed to: {domain}")

    #     self.index, self.graph, self.nodes = \
    #         load_domain_graph(domain)
    #     self.graph_name = f"graphrag_{domain}"

    #     result = self.generate(
    #         learning_objective = learning_objective,
    #         difficulty         = difficulty,
    #         question_type      = question_type,
    #         top_k_nodes        = top_k_nodes,
    #         neighbour_depth    = neighbour_depth
    #     )

    #     # Move domain into metadata
    #     result["metadata"]["domain"] = domain
    #     result["metadata"]["domain_routed_to"] = domain
    #     return result
    
    def generate_with_routing(self,
                          learning_objective: str,
                          difficulty:    str = "medium",
                          question_type: str = "conceptual",
                          domain:        str = None,
                          top_k_nodes:   int = None,
                          neighbour_depth: int = None,
                          force_single: bool = False) -> dict:

        if not self._models_loaded:
            raise RuntimeError("Call load_models() first.")

        from graphrag.graphrag_domain_indexer import (
            load_domain_graph, route_by_embedding, DOMAIN_MAP)

        diff_cfg = cfg["difficulty"][difficulty]

        # ── Domain selection logic ─────────────────────────────
        # Samar: "If user specifies domain → use it directly.
        #         If domain wrong/missing → fall back to router."
        _routing_conf = None   # populated only when auto-routed (1.3)

        if domain and domain in DOMAIN_MAP:
            # User specified valid domain → use directly
            print(f"[GraphRAGPipeline] Using specified domain: {domain}")
            selected_domain = domain

        elif difficulty == "hard" and diff_cfg.get("multi_domain", False) and not force_single:
            # Hard question + no domain specified → multi-domain
            print(f"[GraphRAGPipeline] Hard question → "
                f"attempting multi-domain retrieval")
            return self.generate_with_multi_domain(
                learning_objective = learning_objective,
                difficulty         = difficulty,
                question_type      = question_type
            )

        else:
            # Easy/medium or no domain → best match, with a confidence
            # margin check (GraphRAG 1.3). If the top two domains are
            # within margin_threshold, the query is boundary-spanning,
            # so fall back to multi-domain retrieval.
            scores = route_by_embedding(
                learning_objective, self.embedder, return_all_scores=True
            )
            ranked = sorted(scores.items(), key=lambda kv: kv[1],
                            reverse=True)
            best_domain, best_score = ranked[0]
            runner_up, runner_score = (ranked[1] if len(ranked) > 1
                                       else (None, 0.0))
            margin = round(best_score - runner_score, 3)
            mt = cfg.get("routing", {}).get("margin_threshold", 0.05)
            _routing_conf = {
                "top": best_domain, "top_score": best_score,
                "runner_up": runner_up, "runner_up_score": runner_score,
                "margin": margin, "threshold": mt,
                "ambiguous": margin < mt, "scores": scores,
            }

            if margin < mt and not force_single:
                print(f"[GraphRAGPipeline] Ambiguous routing: "
                      f"{best_domain}={best_score} vs "
                      f"{runner_up}={runner_score} "
                      f"(margin={margin} < {mt}) → multi-domain")
                result = self.generate_with_multi_domain(
                    learning_objective = learning_objective,
                    difficulty         = difficulty,
                    question_type      = question_type
                )
                result["metadata"]["routing_confidence"] = _routing_conf
                return result

            selected_domain = best_domain

        # Single domain retrieval (easy/medium/explicit domain)
        print(f"[GraphRAGPipeline] Routed to: {selected_domain}")

        self.index, self.graph, self.nodes = \
            load_domain_graph(selected_domain)
        self.graph_name = f"graphrag_{selected_domain}"

        result = self.generate(
            learning_objective = learning_objective,
            difficulty         = difficulty,
            question_type      = question_type,
            top_k_nodes        = top_k_nodes,
            neighbour_depth    = neighbour_depth
        )

        result["metadata"]["domain"]          = selected_domain
        result["metadata"]["domain_routed_to"] = selected_domain
        result["metadata"]["routing_confidence"] = _routing_conf
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

        from utils.multi_domain_retrieval import multi_domain_retrieval_params
        retrieval_params = multi_domain_retrieval_params(diff_cfg)
        chunks_per_domain = retrieval_params["max_context_chunks"]

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
                f"— using top scoring domain directly")

            best_domain = max(all_scores, key=all_scores.get)
            print(f"[MultiDomain] Falling back to: {best_domain}")

            self.index, self.graph, self.nodes = \
                load_domain_graph(best_domain)
            self.graph_name = f"graphrag_{best_domain}"

            result = self.generate(
                learning_objective = learning_objective,
                difficulty         = difficulty,
                question_type      = question_type
            )

            result["metadata"]["domain"]           = best_domain
            result["metadata"]["domain_routed_to"] = best_domain
            return result
        
        # Step 4 — Retrieve chunks from each qualifying domain
        combined_chunks   = []
        combined_sources  = []
        combined_concepts = []
        combined_relations = []
        domains_used      = []
        all_chunk_pool    = []

        for domain in qualifying:
            print(f"[MultiDomain] Retrieving from: {domain}")

            # Load domain graph
            index, graph, nodes = load_domain_graph(domain)

            # Same per-domain budget as single-domain hard (max_chunks)
            retrieval = retrieve_graph_context(
                query              = learning_objective,
                index              = index,
                graph              = graph,
                nodes              = nodes,
                embedder           = self.embedder,
                top_k_nodes        = retrieval_params["top_k_nodes"],
                neighbour_depth    = retrieval_params["neighbour_depth"],
                max_context_chunks = retrieval_params["max_context_chunks"],
                seed_ratio         = retrieval_params["seed_ratio"],
            )

            for item in retrieval.get("chunk_pool", []):
                all_chunk_pool.append({**item, "domain": domain})

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

        # Step 8 — Format retrieved chunks for metadata (all domains)
        retrieved_chunks = {
            f"chunk_{i+1}": {
                "text":   item["text"][:200].strip(),
                "source": item.get("source", "unknown"),
                "type":   item.get("type", "seed"),
                "domain": item.get("domain", "unknown"),
            }
            for i, item in enumerate(all_chunk_pool)
        }

        # Format concepts
        clean_concepts = [
            c for c in combined_concepts
            if len(c) > 3
            and not any(
                char in c for char in ['©', '\n', '\r'])
        ]

        # ── LLM as Judge evaluation ────────────────────────
        judge_scores = None

        if self._judge_loaded:
            from evaluation.evaluator import evaluate_generation

            threshold = cfg["judge"].get("score_threshold", 3.5)
            max_regen = cfg["judge"].get("max_regenerations", 2)
            attempts  = 1

            # Convert combined chunks to list for judge
            chunks_for_judge = [
                f"[{info.get('type', 'chunk').upper()} "
                f"from {info.get('source', 'unknown')} "
                f"domain={info.get('domain', 'unknown')}]\n"
                f"{info['text']}"
                for info in retrieved_chunks.values()
            ] if retrieved_chunks else [full_context]

            while attempts <= max_regen + 1:
                eval_result = evaluate_generation(
                    learning_objective = learning_objective,
                    retrieved_chunks   = chunks_for_judge,
                    question           = question,
                    answer             = sample_answer,
                    difficulty         = difficulty,
                    question_type      = question_type,
                    judge_model        = self.judge_model,
                    judge_tokenizer    = self.judge_tokenizer,
                    save               = True
                )

                # Replace with:
                q_score = eval_result["question"]["overall"]

                # Guard against None score (parse failure)
                if q_score is None:
                    print(f"[Judge] Score parse failed — "
                        f"skipping regeneration")
                    judge_scores = {
                        "question_overall":  None,
                        "answer_overall":
                            eval_result["answer"]["overall"],
                        "retrieval_overall":
                            eval_result["retrieval"]["overall"],
                        "attempts":  attempts,
                        "eval_id":   eval_result["eval_id"]
                    }
                    break

                if q_score >= threshold or attempts > max_regen:
                    judge_scores = {
                        "question_overall":
                            q_score,
                        "answer_overall":
                            eval_result["answer"]["overall"],
                        "retrieval_overall":
                            eval_result["retrieval"]["overall"],
                        "attempts":  attempts,
                        "eval_id":   eval_result["eval_id"]
                    }
                    break

                print(f"[Judge] Score below threshold "
                      f"— regenerating...")
                question = generate_question(
                    context            = full_context,
                    learning_objective = learning_objective,
                    model              = self.model,
                    tokenizer          = self.tokenizer,
                    difficulty         = difficulty,
                    question_type      = question_type
                )
                sample_answer = generate_answer(
                    context    = full_context,
                    question   = question,
                    model      = self.model,
                    tokenizer  = self.tokenizer,
                    difficulty = difficulty
                )
                attempts += 1

        return {
            "question": question,
            "metadata": {
                **get_generation_trace(),
                "learning_objective":    learning_objective,
                "difficulty":            difficulty,
                "question_type":         question_type,
                "domain":                "multi_domain",
                "domains_used":          domains_used,
                "domain_scores":         qualifying,
                "sources":               list(set(combined_sources)),
                "retrieval_method":      "GraphRAG-MultiDomain",
                "retrieval_params": {
                    "top_k_nodes":         retrieval_params["top_k_nodes"],
                    "neighbour_depth":     retrieval_params["neighbour_depth"],
                    "max_chunks_per_domain": chunks_per_domain,
                    "max_chunks_single_domain": int(diff_cfg["max_chunks"]),
                    "domains_count":       len(domains_used),
                },
                "key_concepts":          clean_concepts[:8],
                "concept_relationships": combined_relations[:5],
                "retrieved_chunks":      retrieved_chunks,
                "sample_answer":         sample_answer,
                "domain_score_threshold": threshold,
                "judge_scores":          judge_scores,
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

        # After generating question and answer, perform LLM-based evaluation with judge (if loaded)
        # ── LLM as Judge evaluation ────────────────────────────
        judge_scores = None
        attempts     = 1

        if self._judge_loaded:
            from evaluation.evaluator import evaluate_generation

            threshold  = cfg["judge"].get("score_threshold", 3.5)
            max_regen  = cfg["judge"].get("max_regenerations", 2)

            # Convert chunks to list of strings for judge
            chunks_for_judge = [
                f"[{info.get('type', 'chunk').upper()} "
                f"from {info.get('source', 'unknown')}]\n"
                f"{info['text']}"
                for info in retrieved_chunks.values()
            ] if retrieved_chunks else [retrieval["context"]]

            while attempts <= max_regen + 1:
                eval_result = evaluate_generation(
                    learning_objective = learning_objective,
                    retrieved_chunks   = chunks_for_judge,
                    question           = question,
                    answer             = sample_answer,
                    difficulty         = difficulty,
                    question_type      = question_type,
                    judge_model        = self.judge_model,
                    judge_tokenizer    = self.judge_tokenizer,
                    save               = True
                )

                q_score = eval_result["question"]["overall"]
                print(f"[Judge] Question score: {q_score} "
                    f"(threshold={threshold}, "
                    f"attempt={attempts})")

                # Guard against None score (parse failure)
                if q_score is None:
                    print(f"[Judge] Score parse failed — "
                        f"skipping regeneration")
                    judge_scores = {
                        "question_overall":  None,
                        "answer_overall":
                            eval_result["answer"]["overall"],
                        "retrieval_overall":
                            eval_result["retrieval"]["overall"],
                        "attempts":  attempts,
                        "eval_id":   eval_result["eval_id"]
                    }
                    break

                if q_score >= threshold or attempts > max_regen:
                    judge_scores = {
                        "question_overall":  q_score,
                        "answer_overall":
                            eval_result["answer"]["overall"],
                        "retrieval_overall":
                            eval_result["retrieval"]["overall"],
                        "attempts":          attempts,
                        "eval_id":
                            eval_result["eval_id"]
                    }
                    break

                print(f"[Judge] Score {q_score} below threshold "
                    f"{threshold} — regenerating...")
                question = generate_question(
                    context            = retrieval["context"],
                    learning_objective = learning_objective,
                    model              = self.model,
                    tokenizer          = self.tokenizer,
                    difficulty         = difficulty,
                    question_type      = question_type
                )
                sample_answer = generate_answer(
                    context    = retrieval["context"],
                    question   = question,
                    model      = self.model,
                    tokenizer  = self.tokenizer,
                    difficulty = difficulty
                )
                attempts += 1

        return {
            "question":  question,
            "metadata": {
                **get_generation_trace(),
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
                "judge_scores":          judge_scores,
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


