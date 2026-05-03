# ============================================================
# utils/registry.py
# Document Registry Helper
#
# Single source of truth for all PDF tracking.
# Tracks which PDFs exist, their domain, and index status.
#
# Used by:
#   - Admin API (upload, rebuild endpoints)
#   - graphrag_domain_indexer.py (full rebuilds)
#   - rename_docs.py (auto-registration)
# ============================================================

import os
import json
from datetime import date
from typing import Optional

try:
    BASE_DIR = os.getenv(
        "GRAPHRAG_BASE_DIR",
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
    )
except Exception:
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
    Get next available source number.
    Used by rename_docs.py for consistent naming.
    """
    registry = load_registry()
    next_num  = registry["last_source_number"] + 1
    registry["last_source_number"] = next_num
    save_registry(registry)
    print(f"[Registry] Next source number: {next_num}")
    return next_num


def register_document(filename:  str,
                      domain:    str,
                      indexed:   bool = False) -> None:
    """
    Register a document under a specific domain.
    Called by admin upload endpoint only.
    Domain must be specified — use /admin/upload-document
    to assign domain before calling this.
    """
    registry = load_registry()

    if domain not in registry["domains"]:
        raise ValueError(
            f"Domain '{domain}' not found in registry. "
            f"Valid: {list(registry['domains'].keys())}"
        )

    files = registry["domains"][domain]["files"]

    if filename in files:
        # Update existing
        files[filename]["indexed"] = indexed
        print(f"[Registry] Updated: {filename} "
              f"→ {domain} indexed={indexed}")
    else:
        # New entry
        files[filename] = {
            "indexed": indexed,
            "added":   str(date.today())
        }
        print(f"[Registry] Registered: {filename} "
              f"→ {domain} indexed={indexed}")

    save_registry(registry)


def mark_as_indexed(filename: str,
                    domain:   str) -> None:
    """
    Mark a document as indexed after
    incremental update completes.
    """
    registry = load_registry()

    if domain not in registry["domains"]:
        print(f"[Registry] Warning: domain '{domain}' "
              f"not found")
        return

    if filename not in registry["domains"][domain]["files"]:
        print(f"[Registry] Warning: {filename} not in "
              f"{domain} — adding now")
        registry["domains"][domain]["files"][filename] = {
            "indexed": True,
            "added":   str(date.today())
        }
    else:
        registry["domains"][domain]["files"][filename][
            "indexed"] = True

    save_registry(registry)
    print(f"[Registry] Marked indexed: {filename} "
          f"in {domain}")


def get_domain_files(domain:         str,
                     indexed_only:   bool = True) -> list:
    """
    Get all files for a specific domain.
    Replaces hardcoded DOMAIN_MAP file lists.

    Args:
        domain:       domain name
        indexed_only: if True returns only indexed files
                      if False returns all registered files
    """
    registry = load_registry()

    if domain not in registry["domains"]:
        print(f"[Registry] Domain '{domain}' not found")
        return []

    files = registry["domains"][domain]["files"]

    if indexed_only:
        return [
            filename for filename, info in files.items()
            if info["indexed"]
        ]
    return list(files.keys())


def get_unindexed_files(domain: Optional[str] = None
                        ) -> list:
    """
    Get all files not yet indexed.
    Used by /admin/new-documents endpoint.

    Args:
        domain: if specified filter by domain
                if None returns all unindexed files
    """
    registry = load_registry()
    unindexed = []

    for dom, info in registry["domains"].items():
        if domain and dom != domain:
            continue
        for filename, finfo in info["files"].items():
            if not finfo["indexed"]:
                unindexed.append({
                    "filename": filename,
                    "domain":   dom,
                    "added":    finfo["added"]
                })

    return unindexed


def get_registry_summary() -> dict:
    """
    Get summary of all domains and file counts.
    Used by /health and /graphrag/status endpoints.
    """
    registry = load_registry()
    summary  = {}

    for domain, info in registry["domains"].items():
        files   = info["files"]
        indexed = sum(
            1 for f in files.values() if f["indexed"])
        summary[domain] = {
            "total_files":   len(files),
            "indexed_files": indexed,
            "pending_files": len(files) - indexed
        }

    return {
        "last_source_number": registry["last_source_number"],
        "domains": summary,
        "total_files": sum(
            d["total_files"] for d in summary.values()),
        "total_indexed": sum(
            d["indexed_files"] for d in summary.values())
    }

# ============================================================
# TEST — python utils/registry.py
# ============================================================
if __name__ == "__main__":

    print("=" * 50)
    print("REGISTRY TEST")
    print("=" * 50)

    # Test 1 — Load registry
    print("\n1. Loading registry...")
    registry = load_registry()
    print(f"   last_source_number: "
          f"{registry['last_source_number']}")
    print(f"   domains found: "
          f"{list(registry['domains'].keys())}")

    # Test 2 — Summary
    print("\n2. Registry summary...")
    summary = get_registry_summary()
    print(f"   Total files:   {summary['total_files']}")
    print(f"   Total indexed: {summary['total_indexed']}")
    for domain, info in summary["domains"].items():
        print(f"   {domain}: "
              f"{info['indexed_files']} indexed, "
              f"{info['pending_files']} pending")

    # Test 3 — Get domain files
    print("\n3. Get deep_learning files...")
    files = get_domain_files("deep_learning")
    print(f"   Files found: {len(files)}")
    print(f"   First 3: {files[:3]}")

    # Test 4 — Get unindexed files
    print("\n4. Check unindexed files...")
    unindexed = get_unindexed_files()
    print(f"   Unindexed: {len(unindexed)}")
    if unindexed:
        print(f"   First: {unindexed[0]}")
    else:
        print("   All files indexed")

    # Test 5 — Register new document
    print("\n5. Test register_document...")
    register_document(
        "test_document.pdf",
        domain  = "database",
        indexed = False
    )

    # Test 6 — Verify it appears in unindexed
    print("\n6. Verify test doc is unindexed...")
    unindexed = get_unindexed_files(domain="database")
    found = any(
        f["filename"] == "test_document.pdf"
        for f in unindexed
    )
    print(f"   test_document.pdf found in unindexed: "
          f"{found} test_document" if found else f"{found} test_document")

    # Test 7 — Mark as indexed
    print("\n7. Mark test doc as indexed...")
    mark_as_indexed("test_document.pdf", domain="database")

    # Test 8 — Verify it is now indexed
    print("\n8. Verify test doc is now indexed...")
    files = get_domain_files("database")
    found = "test_document.pdf" in files
    print(f"   test_document.pdf in indexed files: "
          f"{found} indexed document" if found else f"{found} indexed document")

    # Cleanup — remove test document from registry
    print("\n9. Cleaning up test document...")
    registry = load_registry()
    if "test_document.pdf" in \
            registry["domains"]["database"]["files"]:
        del registry["domains"]["database"]["files"][
            "test_document.pdf"]
        save_registry(registry)
        print("   test_document.pdf removed from registry")

    print("\n" + "=" * 50)
    print("REGISTRY TEST COMPLETE")
    print("=" * 50)