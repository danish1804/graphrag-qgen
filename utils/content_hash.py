"""
Content hashing for document deduplication.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Stream a file from disk and return its SHA-256 hex digest."""
    digest = hashlib.sha256()
    path = Path(file_path)
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Hash normalised UTF-8 text (caller should normalise first)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
