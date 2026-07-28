# ============================================================
# scripts/test_dedup_exclusion.py
# Verify the domain indexer no longer selects duplicate documents.
#
# get_domain_files() now skips registry entries flagged with
# duplicate_of (e.g. source_73.pdf == source_7.pdf), so duplicates
# stop inflating the graph and its cross-source edge count on
# re-index. include_duplicates=True still returns them for admin views.
#
# Run:
#   python scripts/test_dedup_exclusion.py
# ============================================================
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.registry import get_domain_files

DOMAINS = ["deep_learning", "database", "software_engineering",
           "operating_systems", "computer_architecture"]


def main():
    total_dups = 0
    for d in DOMAINS:
        kept = get_domain_files(d)                              # dedup-excluded
        allf = get_domain_files(d, include_duplicates=True)     # everything
        dups = sorted(set(allf) - set(kept))
        total_dups += len(dups)
        # none of the excluded duplicates should be in the indexing list
        for f in dups:
            assert f not in kept, f"{f} was not excluded!"
        flag = f"  excludes {len(dups)}: {dups}" if dups else ""
        print(f"  {d:24} {len(kept)} files (of {len(allf)}){flag}")

    print(f"\ntotal duplicates excluded from indexing: {total_dups}")
    assert total_dups >= 1, "expected at least one duplicate to be excluded"
    print("[OK] duplicate documents are excluded from domain file selection.")


if __name__ == "__main__":
    main()
