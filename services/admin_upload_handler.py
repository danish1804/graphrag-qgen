"""
Admin upload: dedup BEFORE writing to data/tested/ or registry.

Partner requirement: if content is duplicate, do not save and do not register.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from services.document_dedup_service import DocumentDedupService
from utils.registry import register_document


def duplicate_response_without_save(
    *,
    original_filename: str,
    domain: str,
    assessment: Any,
) -> dict[str, Any]:
    """API payload when upload is rejected as duplicate (no disk/registry changes)."""
    match = assessment.match
    return {
        "status": "ok",
        "original_filename": original_filename,
        "saved_as": None,
        "domain": domain,
        "already_indexed": True,
        "saved_to_disk": False,
        "registered": False,
        "dedup": assessment.to_dict(),
        "message": (
            f"Upload rejected: content matches existing indexed file "
            f"'{match.matched_file}' ({match.reason}). "
            "File was not saved to tested/ and was not registered."
        ),
    }


def register_new_file_for_rebuild(
    *,
    tested_dir: str,
    domain: str,
    final_filename: str,
    original_filename: str,
    assessment: Any,
) -> dict[str, Any]:
    """Register a new file already saved under tested/."""
    register_document(final_filename, domain=domain, indexed=False)
    return {
        "status": "ok",
        "original_filename": original_filename,
        "saved_as": final_filename,
        "domain": domain,
        "already_indexed": False,
        "saved_to_disk": True,
        "registered": True,
        "dedup": assessment.to_dict(),
        "message": (
            f"Saved as {final_filename} in {domain}. "
            "Call POST /admin/rebuild-domain to index."
        ),
    }


def assess_before_save(
    dedup_service: DocumentDedupService,
    file_path: str,
    domain: str,
) -> Any:
    """Run hash/similarity checks on a temp or existing path."""
    return dedup_service.assess_file(file_path, domain)


def save_bytes_to_tested(
    *,
    tested_dir: str,
    final_filename: str,
    content: bytes,
) -> str:
    """Write uploaded bytes to data/tested/ and return full path."""
    final_path = os.path.join(tested_dir, final_filename)
    with open(final_path, "wb") as handle:
        handle.write(content)
    return final_path


def save_temp_file(content: bytes, suffix: str) -> str:
    """Write bytes to a temporary file for pre-save dedup checks."""
    import tempfile

    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
    except Exception:
        os.close(fd)
        raise
    return temp_path


def remove_temp_file(temp_path: str | None) -> None:
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)


def move_temp_to_tested(temp_path: str, tested_dir: str, final_filename: str) -> str:
    """Move a verified temp upload into data/tested/."""
    final_path = os.path.join(tested_dir, final_filename)
    shutil.move(temp_path, final_path)
    return final_path
