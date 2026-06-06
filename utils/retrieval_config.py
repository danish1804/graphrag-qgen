"""Retrieval parameter helpers driven by config.yml difficulty blocks."""

from __future__ import annotations

try:
    from utils.config import cfg
except ImportError:
    from config import cfg


def get_difficulty_retrieval(difficulty: str) -> dict[str, int]:
    """Return top_k_nodes and neighbour_depth for a difficulty level."""
    diff_cfg = cfg.get("difficulty", {}).get(difficulty, {})
    retrieval = cfg.get("retrieval", {})
    return {
        "top_k_nodes": int(
            diff_cfg.get("top_k_nodes", retrieval.get("top_k_nodes", 3))
        ),
        "neighbour_depth": int(
            diff_cfg.get("neighbour_depth", retrieval.get("neighbour_depth", 1))
        ),
        "max_chunks": int(diff_cfg.get("max_chunks", retrieval.get("max_context_chunks", 6))),
    }


def get_standard_rag_top_k() -> int:
    """Top K chunks for single/multi standard RAG."""
    return int(cfg.get("retrieval", {}).get("top_k_results", 3))
