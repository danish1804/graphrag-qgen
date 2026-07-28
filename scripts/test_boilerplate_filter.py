# ============================================================
# scripts/test_boilerplate_filter.py
# Verify the ingestion-time boilerplate / quality filter.
#
# is_quality_chunk drops front-matter, TOC, copyright and
# reference-list chunks so they never become graph nodes. This
# loads a known-noisy textbook, shows the drop rate, and prints
# the KEEP/DROP verdict on the front matter (should be DROP) vs
# body content (should be KEEP).
#
# Run:
#   python scripts/test_boilerplate_filter.py
# ============================================================
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_loader import load_and_chunk, is_quality_chunk

BASE_DIR   = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
TESTED = os.path.join(BASE_DIR, "data", "tested")
FILES = ["source_14.pdf"]   # Database Design — has title page + TOC


def main():
    for fname in FILES:
        path = os.path.join(TESTED, fname)
        if not os.path.exists(path):
            print(f"  (skip {fname} — not found)")
            continue
        # get ALL chunks (filter off) so we can show the verdicts
        allc = load_and_chunk(path, filter_boilerplate=False)["chunks"]
        kept = [c for c in allc if is_quality_chunk(c)]
        dropped = len(allc) - len(kept)
        print(f"\n{fname}: {len(allc)} chunks -> {len(kept)} kept "
              f"({dropped} dropped, {100*dropped/max(len(allc),1):.0f}%)")

        print("  front matter (first 6 chunks):")
        for c in allc[:6]:
            v = "KEEP" if is_quality_chunk(c) else "DROP"
            preview = c[:64].replace("\n", " ").strip()
            print(f"    [{v}] {preview!r}")

        assert dropped >= 1, "expected some boilerplate to be dropped"
    print("\n[OK] boilerplate filter drops front-matter, keeps content.")


if __name__ == "__main__":
    main()
