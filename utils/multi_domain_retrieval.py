"""Multi-domain GraphRAG retrieval parameter helpers."""

from __future__ import annotations

from typing import Any


def per_domain_chunk_limit(diff_cfg: dict[str, Any]) -> int:
    """
    Chunks retrieved from each qualifying domain in multi-domain mode.

    Defaults to max_chunks so hard+auto-route matches hard+fixed-domain
    (single-domain generate() uses max_chunks as max_context_chunks).
    """
    max_chunks = int(diff_cfg["max_chunks"])
    override = diff_cfg.get("chunks_per_domain")
    if override is None:
        return max_chunks
    return max(int(override), max_chunks)


def multi_domain_retrieval_params(
    diff_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Shared retrieval knobs for each domain in a multi-domain call."""
    return {
        "top_k_nodes": int(diff_cfg["top_k_nodes"]),
        "neighbour_depth": int(diff_cfg["neighbour_depth"]),
        "max_context_chunks": per_domain_chunk_limit(diff_cfg),
        "seed_ratio": float(diff_cfg.get("seed_ratio", 0.6)),
    }
