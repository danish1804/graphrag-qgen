"""
Incremental rebuild helpers: dedup filter, chunk loading, fingerprint storage.
"""

from __future__ import annotations

import os
from typing import Any

from services.document_dedup_service import DocumentDedupService
from utils.content_hash import sha256_file, sha256_text
from utils.document_loader import load_and_chunk, load_document
from utils.registry_content import (
    mark_duplicate_indexed,
    record_content_fingerprints,
)
from utils.text_normalization import normalize_document_text


def filter_and_skip_duplicates(
    dedup_service: DocumentDedupService,
    domain: str,
    new_files: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Remove near-duplicate files before expensive graph merge."""
    if not dedup_service.is_enabled():
        return new_files, []
    return dedup_service.filter_files_for_indexing(domain, new_files)


def mark_skipped_duplicates(
    domain: str,
    skipped: list[dict[str, Any]],
) -> None:
    """Persist registry state for files skipped as duplicates."""
    for item in skipped:
        match = item.get("match") or {}
        mark_duplicate_indexed(
            domain,
            item["filename"],
            matched_file=match.get("matched_file", "unknown"),
            reason=match.get("reason", "duplicate"),
            similarity_score=match.get("similarity_score"),
        )


def load_chunks_for_files(
    tested_dir: str,
    filenames: list[str],
) -> tuple[list[str], list[str]]:
    """Load and chunk PDFs; return parallel chunk and source lists."""
    all_chunks: list[str] = []
    all_sources: list[str] = []
    for pdf_file in filenames:
        pdf_path = os.path.join(tested_dir, pdf_file)
        if not os.path.exists(pdf_path):
            print(f"[Admin] Warning: {pdf_file} not found in tested/")
            continue
        result = load_and_chunk(pdf_path)
        all_chunks.extend(result["chunks"])
        all_sources.extend([pdf_file] * len(result["chunks"]))
    return all_chunks, all_sources


def store_fingerprints_for_files(
    tested_dir: str,
    domain: str,
    filenames: list[str],
) -> None:
    """Record byte/text hashes after successful indexing."""
    for name in filenames:
        path = os.path.join(tested_dir, name)
        if not os.path.exists(path):
            continue
        text = load_document(path)
        record_content_fingerprints(
            domain,
            name,
            content_hash_bytes=sha256_file(path),
            content_hash_text=sha256_text(normalize_document_text(text)),
        )
