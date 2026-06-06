"""Question type helpers driven by config.yml."""

from __future__ import annotations

try:
    from utils.config import cfg
except ImportError:
    from config import cfg


def valid_question_types() -> set[str]:
    """Return enabled question type keys from config."""
    return set(cfg.get("question_types", {}).keys())
