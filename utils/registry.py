# utils/registry.py
# ============================================================
# Document Registry — single source of truth for all PDFs
#
# Tracks:
#   - Which PDFs exist in the system
#   - Which domain each PDF belongs to
#   - Whether each PDF has been indexed
#   - Last source number for auto-naming
#
# Used by:
#   - Admin API (upload, rebuild)
#   - Domain indexer (build_domain_graph)
#   - rename_docs.py (get next number)
# ============================================================

import os
import json
from datetime import date
from typing import Optional

try:
    from utils.config import cfg
    BASE_DIR = os.getenv(
        "GRAPHRAG_BASE_DIR",
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
    )
except ImportError:
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))

REGISTRY_PATH = os.path.join(
    BASE_DIR, "data", "document_registry.json")


def load_registry() -> dict:
    """Load registry from disk."""
    if not os.path.exists(REGISTRY_PATH):
        raise FileNotFoundError(
            f"Registry not found at {REGISTRY_PATH}. "
            f"Run scripts/create_registry.py first."
        )
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    """Save registry to disk."""
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2,
                  ensure_ascii=False)


def get_next_source_number() -> int:
    """
    Get next available source number for renaming.
    Increments last_source_number in registry.
    """
    registry = load_registry()
    next_num = registry["last_source_number"] + 1
    registry["last_source_number"] = next_num
    save_registry(registry)
    return next_num


def register_document(filename:   str,
                      domain:     Optional[str] = None,
                      indexed:    bool = False) -> None:
    """
    Register a new document in the registry.
    Called when:
      - rename_docs.py renames a new file
      - admin uploads a document via API
    """
    registry = load_registry()

    if filename in registry["documents"]:
        # Update existing entry
        if domain:
            registry["documents"][filename]["domain"] = domain
        print(f"[Registry] Updated: {filename} "
              f"→ domain={domain}")
    else:
        # New entry
        registry["documents"][filename] = {
            "domain":  domain,
            "indexed": indexed,
            "added":   str(date.today())
        }
        print(f"[Registry] Registered: {filename} "
              f"→ domain={domain}")

    save_registry(registry)


def mark_as_indexed(filename: str) -> None:
    """
    Mark a document as indexed after
    incremental update completes.
    """
    registry = load_registry()

    if filename not in registry["documents"]:
        print(f"[Registry] Warning: {filename} "
              f"not in registry — adding now")
        registry["documents"][filename] = {
            "domain":  None,
            "indexed": True,
            "added":   str(date.today())
        }
    else:
        registry["documents"][filename]["indexed"] = True

    save_registry(registry)
    print(f"[Registry] Marked indexed: {filename}")


def get_domain_files(domain: str) -> list:
    """
    Get all indexed files for a specific domain.
    Replaces hardcoded DOMAIN_MAP file lists.
    Used by build_domain_graph() for full rebuilds.
    """
    registry = load_registry()
    return [
        filename
        for filename, info in registry["documents"].items()
        if info["domain"] == domain
        and info["indexed"]
    ]


def get_unindexed_files(domain: Optional[str] = None
                        ) -> list:
    """
    Get files not yet indexed.
    Used by /admin/new-documents endpoint.
    """
    registry = load_registry()
    return [
        {"filename": filename, "domain": info["domain"]}
        for filename, info in registry["documents"].items()
        if not info["indexed"]
        and (domain is None or info["domain"] == domain)
    ]


def get_all_domains() -> dict:
    """
    Get summary of all domains and their file counts.
    """
    registry = load_registry()
    domains = {}
    for filename, info in registry["documents"].items():
        domain = info["domain"] or "unassigned"
        if domain not in domains:
            domains[domain] = {"total": 0, "indexed": 0}
        domains[domain]["total"] += 1
        if info["indexed"]:
            domains[domain]["indexed"] += 1
    return domains