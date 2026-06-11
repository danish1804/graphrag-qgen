"""
Data models for document deduplication results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DedupMatch:
    """Describes why a file was treated as a duplicate."""

    matched_file: str
    reason: str  # exact_bytes | exact_text | high_similarity
    similarity_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "matched_file": self.matched_file,
            "reason": self.reason,
        }
        if self.similarity_score is not None:
            payload["similarity_score"] = self.similarity_score
        return payload


@dataclass(frozen=True)
class DedupAssessment:
    """Outcome of checking one file before indexing."""

    is_duplicate: bool
    should_skip_indexing: bool
    match: DedupMatch | None = None
    content_hash_bytes: str | None = None
    content_hash_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_duplicate": self.is_duplicate,
            "should_skip_indexing": self.should_skip_indexing,
            "match": self.match.to_dict() if self.match else None,
            "content_hash_bytes": self.content_hash_bytes,
            "content_hash_text": self.content_hash_text,
        }
