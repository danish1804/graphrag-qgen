"""
Document deduplication for admin upload and incremental rebuild.

Strategy (Samar / demo feedback):
  1. Byte-level SHA-256 — skip identical files before chunking/graph work.
  2. Normalised text SHA-256 — skip when extracted text matches an indexed doc.
  3. SequenceMatcher similarity — flag/skip near-duplicates (e.g. minor header edits,
     different publisher) above configurable threshold.

Comparisons are limited to the target domain and capped by max_similarity_comparisons
so cost stays bounded as the library grows.
"""

from __future__ import annotations

import os
from typing import Any

from utils.config import cfg
from utils.content_hash import sha256_file, sha256_text
from utils.document_loader import load_document
from utils.registry_content import list_indexed_files
from utils.text_normalization import normalize_document_text
from utils.text_similarity import similarity_ratio

from services.document_dedup_models import DedupAssessment, DedupMatch


class DocumentDedupService:
    """Assess whether a new file should be indexed or skipped as a duplicate."""

    def __init__(self, tested_dir: str, dedup_config: dict[str, Any] | None = None):
        self.tested_dir = tested_dir
        self.cfg = dedup_config or cfg.get("deduplication", {})

    def is_enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def assess_file(self, file_path: str, domain: str) -> DedupAssessment:
        """Run hash + optional similarity checks against indexed domain files."""
        if not self.is_enabled():
            return DedupAssessment(
                is_duplicate=False,
                should_skip_indexing=False,
                content_hash_bytes=sha256_file(file_path),
            )

        bytes_hash = sha256_file(file_path)
        indexed = list_indexed_files(domain)

        byte_match = self._match_bytes_hash(bytes_hash, indexed, file_path)
        if byte_match:
            return self._duplicate_result(bytes_hash, byte_match, None)

        text = load_document(file_path)
        text_hash = sha256_text(normalize_document_text(text))

        text_match = self._match_text_hash(text_hash, indexed)
        if text_match:
            return self._duplicate_result(bytes_hash, text_match, text_hash)

        sim_match = self._match_similarity(text, indexed, file_path)
        if sim_match:
            return self._duplicate_result(bytes_hash, sim_match, text_hash)

        return DedupAssessment(
            is_duplicate=False,
            should_skip_indexing=False,
            content_hash_bytes=bytes_hash,
            content_hash_text=text_hash,
        )

    def filter_files_for_indexing(
        self,
        domain: str,
        filenames: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Split unindexed filenames into those to process vs duplicates to skip.
        """
        to_index: list[str] = []
        skipped: list[dict[str, Any]] = []

        for name in filenames:
            path = os.path.join(self.tested_dir, name)
            if not os.path.exists(path):
                continue
            assessment = self.assess_file(path, domain)
            if assessment.should_skip_indexing and assessment.match:
                skipped.append({"filename": name, **assessment.to_dict()})
            else:
                to_index.append(name)

        return to_index, skipped

    def _duplicate_result(
        self,
        bytes_hash: str,
        match: DedupMatch,
        text_hash: str | None,
    ) -> DedupAssessment:
        action = self.cfg.get("on_exact_hash", "skip")
        if match.reason == "high_similarity":
            action = self.cfg.get("on_high_similarity", "skip")
        skip = action == "skip"
        return DedupAssessment(
            is_duplicate=True,
            should_skip_indexing=skip,
            match=match,
            content_hash_bytes=bytes_hash,
            content_hash_text=text_hash,
        )

    def _bytes_hash_for_row(self, row: dict[str, Any]) -> str | None:
        """Registry hash if present, otherwise hash the file on disk."""
        stored = row.get("content_hash_bytes")
        if stored:
            return stored
        path = os.path.join(self.tested_dir, row["filename"])
        if not os.path.exists(path):
            return None
        return sha256_file(path)

    def _match_bytes_hash(
        self,
        bytes_hash: str,
        indexed: list[dict[str, Any]],
        new_path: str,
    ) -> DedupMatch | None:
        # Skip comparing file to itself when re-registering same path/name.
        new_name = os.path.basename(new_path)
        for row in indexed:
            if row["filename"] == new_name:
                continue
            candidate = self._bytes_hash_for_row(row)
            if candidate and candidate == bytes_hash:
                return DedupMatch(
                    matched_file=row["filename"],
                    reason="exact_bytes",
                )
        return None

    def _match_text_hash(
        self,
        text_hash: str,
        indexed: list[dict[str, Any]],
    ) -> DedupMatch | None:
        for row in indexed:
            if row.get("content_hash_text") == text_hash:
                return DedupMatch(
                    matched_file=row["filename"],
                    reason="exact_text",
                )
        return None

    def _match_similarity(
        self,
        new_text: str,
        indexed: list[dict[str, Any]],
        new_path: str,
    ) -> DedupMatch | None:
        threshold = float(self.cfg.get("similarity_threshold", 0.95))
        sample_chars = int(self.cfg.get("similarity_sample_chars", 15000))
        max_cmp = int(self.cfg.get("max_similarity_comparisons", 25))
        new_bytes_hash = sha256_file(new_path)

        candidates = []
        for row in indexed[:max_cmp]:
            if row["filename"] == os.path.basename(new_path):
                continue
            candidate_hash = self._bytes_hash_for_row(row)
            if candidate_hash == new_bytes_hash:
                continue
            candidates.append(row)

        best_file = None
        best_score = 0.0

        for row in candidates:
            candidate_path = os.path.join(self.tested_dir, row["filename"])
            if not os.path.exists(candidate_path):
                continue
            candidate_text = load_document(candidate_path)
            score = similarity_ratio(
                new_text,
                candidate_text,
                sample_chars=sample_chars,
            )
            if score > best_score:
                best_score = score
                best_file = row["filename"]

        if best_file and best_score >= threshold:
            return DedupMatch(
                matched_file=best_file,
                reason="high_similarity",
                similarity_score=best_score,
            )
        return None
