"""
Text similarity helpers for near-duplicate document detection.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from utils.text_normalization import normalize_document_text


def similarity_ratio(text_a: str, text_b: str, sample_chars: int = 15000) -> float:
    """
    Return a ratio in [0, 1] between two documents.

    Uses SequenceMatcher on a prefix sample to bound cost when corpora are large.
    """
    norm_a = normalize_document_text(text_a)[:sample_chars]
    norm_b = normalize_document_text(text_b)[:sample_chars]
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()
