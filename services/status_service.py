from __future__ import annotations

import json
import os
from typing import Any


class StatusService:
    """Builds health and GraphRAG status payloads."""

    def __init__(
        self,
        *,
        index_dir: str,
        domain_map: dict[str, Any],
        single_pipeline: Any,
        multi_pipeline: Any,
        graph_pipeline: Any,
    ):
        self.index_dir = index_dir
        self.domain_map = domain_map
        self.single_pipeline = single_pipeline
        self.multi_pipeline = multi_pipeline
        self.graph_pipeline = graph_pipeline

    def build_health(self, valid_domains: set[str]) -> dict[str, Any]:
        single_loaded = self.single_pipeline.index is not None
        single_name = getattr(self.single_pipeline, "index_name", None)

        multi_loaded = self.multi_pipeline.index is not None
        multi_chunks = 0
        if multi_loaded:
            try:
                multi_chunks = len(self.multi_pipeline.chunks)
            except Exception:
                pass

        graph_domains = {}
        for domain_name in sorted(valid_domains):
            graph_path = os.path.join(
                self.index_dir,
                f"graphrag_{domain_name}_graph.pkl",
            )
            graph_domains[domain_name] = os.path.exists(graph_path)

        return {
            "status": "ok",
            "models_loaded": self.single_pipeline._models_loaded,
            "standard_rag": {
                "single_doc": {"loaded": single_loaded, "index_name": single_name},
                "multi_doc": {"loaded": multi_loaded, "total_chunks": multi_chunks},
            },
            "graphrag": {
                "domains_available": graph_domains,
                "current_index": self.graph_pipeline.graph_name,
            },
        }

    def build_graphrag_status(self, valid_domains: set[str]) -> dict[str, Any]:
        domain_stats: dict[str, Any] = {}

        for domain_name in sorted(valid_domains):
            meta_path = os.path.join(
                self.index_dir,
                f"graphrag_{domain_name}_meta.json",
            )
            domain_info = self.domain_map.get(domain_name, {})
            description = domain_info.get("description", f"{domain_name} domain")
            fallback_docs = domain_info.get("files", [])

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                domain_stats[domain_name] = {
                    "available": True,
                    "graph_nodes": meta.get("total_nodes", 0),
                    "graph_edges": meta.get("total_edges", 0),
                    "cross_source_edges": meta.get("cross_source_edges", 0),
                    "total_chunks": meta.get("total_chunks", 0),
                    "documents": meta.get("documents", fallback_docs),
                    "description": description,
                }
            else:
                domain_stats[domain_name] = {
                    "available": False,
                    "description": description,
                    "message": f"Run POST /graphrag/rebuild/{domain_name} to build.",
                }

        return {
            "graphrag_available": any(v["available"] for v in domain_stats.values()),
            "current_index": self.graph_pipeline.graph_name,
            "domains": domain_stats,
        }
