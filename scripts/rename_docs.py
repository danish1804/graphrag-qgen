# ============================================================
# rename_docs.py
# Renames all documents in a folder to a consistent format:
# source_1.pdf, source_2.pdf, source_3.pdf ...
#
# Features:
# - Handles PDF and PPTX
# - Skips already-renamed files
# - Counts existing source_N files and continues from there
# - Logs every rename for traceability
# - Dry-run mode to preview changes before applying
#
# Usage:
#   python rename_docs.py              # renames files
#   python rename_docs.py --dry-run   # preview only, no changes
# ============================================================

import os
import sys
import re
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR      = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR      = os.path.join(BASE_DIR, "data", "raw")
TESTED_DIR    = os.path.join(BASE_DIR, "data", "tested")
LOG_FILE      = os.path.join(BASE_DIR, "data", "rename_log.txt")
ALLOWED_EXTS  = {".pdf", ".pptx"}
PREFIX        = "source"

# Registry integration
try:
    from utils.registry import (
        load_registry,
        save_registry)
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    print("[Registry] Not available — using folder scan")


def get_existing_counter(folder: str) -> int:
    """
    Scan folder for files already named source_N.*
    Return the highest N found, or 0 if none exist.
    """
    pattern = re.compile(rf"^{PREFIX}_(\d+)\.", re.IGNORECASE)
    max_n = 0
    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            n = int(match.group(1))
            if n > max_n:
                max_n = n
    return max_n


def get_files_to_rename(folder: str) -> list:
    """
    Return list of files that need renaming.
    Skips files already in source_N format.
    Skips unsupported file types.
    """
    pattern = re.compile(rf"^{PREFIX}_\d+\.", re.IGNORECASE)
    to_rename = []

    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)

        # Skip directories
        if os.path.isdir(fpath):
            continue

        # Skip unsupported types
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_EXTS:
            continue

        # Skip already renamed files
        if pattern.match(fname):
            continue

        to_rename.append((fpath, fname, ext))

    return to_rename


def write_log(lines: list):
    """Append rename log entries to log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Rename session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        for line in lines:
            f.write(line + "\n")


def rename_documents(dry_run: bool = False):
    """
    Main rename function.

    Args:
        dry_run: if True, prints what would happen without changing files
    """
    print("=" * 60)
    print("DOCUMENT RENAMER")
    print("=" * 60)
    print(f"Folder  : {DATA_DIR}")
    print(f"Mode    : {'DRY RUN - no changes will be made' if dry_run else 'LIVE - files will be renamed'}")
    print("=" * 60)

    # Check folder exists
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: Folder not found: {DATA_DIR}")
        print("Create the folder and add your documents first.")
        return

    # # Find highest existing source_N counter
    # existing_max = get_existing_counter(DATA_DIR)
    # print(f"\nExisting source_N files found: {existing_max}")
    # print(f"New files will start from    : source_{existing_max + 1}")

    # Replace with:
    if REGISTRY_AVAILABLE:
        from utils.registry import load_registry
        registry     = load_registry()
        existing_max = registry["last_source_number"]
        print(f"\n[Registry] Last source number: {existing_max}")
        print(f"New files will start from: source_{existing_max + 1}")
    else:
        existing_max = get_existing_counter(DATA_DIR)
        print(f"\nExisting source_N files found: {existing_max}")
        print(f"New files will start from    : source_{existing_max + 1}")

    # Get files that need renaming
    to_rename = get_files_to_rename(DATA_DIR)

    if not to_rename:
        print("\nNo new files to rename - everything is already consistent.")
        return

    print(f"\nFiles to rename: {len(to_rename)}")
    print("-" * 60)

    log_lines = []
    counter = existing_max + 1

    for old_path, old_name, ext in to_rename:
        new_name = f"{PREFIX}_{counter}{ext}"
        new_path = os.path.join(DATA_DIR, new_name)

        # Handle collision (extremely unlikely but safe)
        while os.path.exists(new_path):
            counter += 1
            new_name = f"{PREFIX}_{counter}{ext}"
            new_path = os.path.join(DATA_DIR, new_name)

        print(f"  [{counter:>3}] {old_name}")
        print(f"        -> {new_name}")

        log_entry = f"[{counter:>3}] {old_name} -> {new_name}"

        # if not dry_run:
        #     os.rename(old_path, new_path)
        #     log_entry += " [RENAMED]"
        # else:
        #     log_entry += " [DRY RUN]"

        # log_lines.append(log_entry)
        # counter += 1

        # Replace with:
        if not dry_run:
            # Step 1 — Rename in raw/
            os.rename(old_path, new_path)
            log_entry += " [RENAMED]"

            # Step 2 — Move to tested/ folder
            tested_path = os.path.join(TESTED_DIR, new_name)
            shutil.move(new_path, tested_path)
            log_entry += " [MOVED TO TESTED]"
            print(f"        -> moved to tested/")

            # Step 3 — Update registry counter only
            if REGISTRY_AVAILABLE:
                from utils.registry import load_registry, save_registry
                registry = load_registry()
                registry["last_source_number"] = counter
                save_registry(registry)
                log_entry += " [REGISTRY UPDATED]"
                print(f"        -> registry counter: {counter}")
        else:
            log_entry += " [DRY RUN]"

        log_lines.append(log_entry)
        counter += 1

    # Summary
    print("-" * 60)
    if dry_run:
        print(f"\nDRY RUN complete - {len(to_rename)} files would be renamed.")
        print("Run without --dry-run to apply changes.")
    else:
        print(f"\nDone - {len(to_rename)} files renamed successfully.")
        write_log(log_lines)
        print(f"Log saved: {LOG_FILE}")

    # Show final state of folder
    print("\nCurrent folder contents:")
    print("-" * 60)
    all_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTS
    ])
    for i, fname in enumerate(all_files, 1):
        fpath = os.path.join(DATA_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {i:>3}. {fname:<40} ({size_mb:.1f} MB)")

    print(f"\nTotal documents: {len(all_files)}")


# -- Entry point ----------------------------------------
if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    rename_documents(dry_run=dry_run)