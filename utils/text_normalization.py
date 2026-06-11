"""
Text normalisation for similarity and content fingerprints.
"""

from __future__ import annotations

import re


def normalize_document_text(text: str) -> str:
    """
    Collapse whitespace and lower-case for stable hashing / similarity.
    Does not stem or remove stopwords — keeps implementation predictable.
    """
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed
