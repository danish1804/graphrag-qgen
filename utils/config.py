# ============================================================
# utils/config.py
# Configuration loader — reads config.yml
#
# Usage:
#   from utils.config import cfg
#   chunk_size = cfg["document"]["chunk_size"]
#   hard_chunks = cfg["difficulty"]["hard"]["max_chunks"]
# ============================================================

import os
import yaml

_config = None

def load_config(config_path: str = None) -> dict:
    """
    Load config.yml from project root.
    Cached after first load — subsequent calls return cached version.
    """
    global _config

    if _config is not None:
        return _config

    if config_path is None:
        # Auto-detect project root
        root = os.getenv(
            "GRAPHRAG_BASE_DIR",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(root, "config.yml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yml not found at {config_path}. "
            f"Copy config.yml.example to config.yml and edit."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    print(f"[Config] Loaded from {config_path}")
    return _config


# Convenience — import cfg directly
cfg = load_config()