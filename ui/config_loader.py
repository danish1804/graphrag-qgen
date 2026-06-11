"""Load retrieval defaults from project config.yml for the Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.retrieval_config import (  # noqa: E402
    get_difficulty_retrieval,
    get_standard_rag_top_k,
)

__all__ = ["get_difficulty_retrieval", "get_standard_rag_top_k"]
