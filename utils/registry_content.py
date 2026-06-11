"""
Registry helpers for per-file content fingerprints (deduplication).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from utils.registry import load_registry, save_registry


def list_indexed_files(domain: str) -> list[dict[str, Any]]:
    """Return metadata dicts for indexed files in a domain."""
    registry = load_registry()
    domain_data = registry.get("domains", {}).get(domain, {})
    files = domain_data.get("files", {})
    rows: list[dict[str, Any]] = []
    for filename, info in files.items():
        if not info.get("indexed"):
            continue
        rows.append({"filename": filename, **info})
    return rows


def update_file_fields(domain: str, filename: str, fields: dict[str, Any]) -> None:
    """Merge fields into an existing registry file entry."""
    registry = load_registry()
    entry = registry["domains"][domain]["files"].get(filename)
    if entry is None:
        entry = {"indexed": False, "added": str(date.today())}
        registry["domains"][domain]["files"][filename] = entry
    entry.update(fields)
    save_registry(registry)


def record_content_fingerprints(
    domain: str,
    filename: str,
    *,
    content_hash_bytes: str,
    content_hash_text: str,
) -> None:
    """Persist hashes after a successful index (used for future dedup)."""
    update_file_fields(
        domain,
        filename,
        {
            "content_hash_bytes": content_hash_bytes,
            "content_hash_text": content_hash_text,
        },
    )


def mark_duplicate_indexed(
    domain: str,
    filename: str,
    *,
    matched_file: str,
    reason: str,
    similarity_score: float | None = None,
) -> None:
    """
    Mark a file as indexed without graph merge (duplicate of existing content).
    """
    fields: dict[str, Any] = {
        "indexed": True,
        "duplicate_of": matched_file,
        "dedup_reason": reason,
        "added": str(date.today()),
    }
    if similarity_score is not None:
        fields["similarity_score"] = round(similarity_score, 4)
    update_file_fields(domain, filename, fields)
