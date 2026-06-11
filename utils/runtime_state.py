import json
import os
from utils.atomic_io import atomic_write_json


BASE_DIR = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
STATE_PATH = os.path.join(BASE_DIR, "data", "runtime_state.json")


def _default_state() -> dict:
    return {"domains": {}}


def load_runtime_state() -> dict:
    """Load mutable runtime state from disk."""
    if not os.path.exists(STATE_PATH):
        return _default_state()
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "domains" not in data or not isinstance(data["domains"], dict):
        data["domains"] = {}
    return data


def save_runtime_state(state: dict) -> None:
    """Persist mutable runtime state to disk."""
    atomic_write_json(STATE_PATH, state)


def add_runtime_domain(domain_name: str, description: str, keywords: list[str]) -> dict:
    """Register/update a runtime domain without mutating config.yml."""
    state = load_runtime_state()
    state["domains"][domain_name] = {
        "description": description,
        "keywords": keywords,
        "enabled": True
    }
    save_runtime_state(state)
    return state["domains"][domain_name]


def get_runtime_domains() -> dict:
    """Return runtime-defined domains as a name->settings map."""
    return load_runtime_state().get("domains", {})
