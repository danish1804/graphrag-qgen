from __future__ import annotations

from typing import Any


class GraphRAGGenerationService:
    """Application-layer orchestration for GraphRAG generation."""

    def __init__(self, graph_pipeline: Any):
        self.graph_pipeline = graph_pipeline

    def generate_with_domain_fallback(
        self,
        *,
        learning_objective: str,
        difficulty: str,
        question_type: str,
        domain: str | None,
        top_k_nodes: int,
        neighbour_depth: int,
        valid_domains: set[str],
    ) -> dict[str, Any]:
        """
        Execute GraphRAG generation while preserving legacy fallback behavior:
        invalid explicit domain -> auto-routing instead of request failure.
        """
        resolved_domain = domain
        if resolved_domain and resolved_domain not in valid_domains:
            print(
                f"[Warning] Invalid domain '{resolved_domain}' specified "
                f"— falling back to auto-routing. Valid domains: {valid_domains}"
            )
            resolved_domain = None

        return self.graph_pipeline.generate_with_routing(
            learning_objective=learning_objective,
            difficulty=difficulty,
            question_type=question_type,
            domain=resolved_domain,
            top_k_nodes=top_k_nodes,
            neighbour_depth=neighbour_depth,
            force_single=(domain is not None and resolved_domain is None),
        )
