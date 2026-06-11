from __future__ import annotations

import os
import time
from typing import Any


class IndexRepository:
    """Repository for GraphRAG index update/load operations."""

    def __init__(self, *, index_dir: str, graph_pipeline: Any):
        self.index_dir = index_dir
        self.graph_pipeline = graph_pipeline

    def incremental_update_domain(
        self,
        *,
        domain: str,
        new_chunks: list[str],
        new_sources: list[str],
    ) -> tuple[dict[str, Any], float]:
        from graphrag.graphrag_indexer import incremental_update_graph

        graph_name = f"graphrag_{domain}"
        started = time.time()
        summary = incremental_update_graph(
            new_chunks=new_chunks,
            new_sources=new_sources,
            index_dir=self.index_dir,
            graph_name=graph_name,
            embedder=self.graph_pipeline.embedder,
        )
        elapsed = round(time.time() - started, 3)
        return summary, elapsed

    def reload_domain(self, domain: str) -> None:
        self.graph_pipeline.load_domain_index(domain)

    def domain_graph_exists(self, domain: str) -> bool:
        graph_path = os.path.join(self.index_dir, f"graphrag_{domain}_graph.pkl")
        return os.path.exists(graph_path)
