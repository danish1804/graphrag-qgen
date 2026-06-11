# save as test_chunk_filter.py
import os
import sys
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_fine_tune_data import NOISE_PATTERNS, SIGNAL_KEYWORDS, is_quality_chunk
from utils.document_loader import load_and_chunk

# def is_quality_chunk(text: str) -> bool:
#     """
#     Returns True if chunk is worth generating a question from.

#     Filters out:
#     - Publisher/copyright boilerplate
#     - Front matter (preface, acknowledgements, TOC)
#     - Administrative exam text
#     - Chunks with too little signal content
#     """
#     text_lower = text.lower().strip()

#     # Reject if too short
#     if len(text) < 200:
#         return False

#     # Reject if matches noise patterns
#     for pattern in NOISE_PATTERNS:
#         if re.search(pattern, text_lower):
#             return False

#     # Reject if mostly numbers/symbols — formula dumps
#     alpha_ratio = sum(c.isalpha() for c in text) / len(text)
#     if alpha_ratio < 0.4:
#         return False

#     # Reject if too many consecutive capital letters
#     # (likely a heading/title chunk)
#     words = text.split()
#     if len(words) < 10:
#         return False

#     # Reject if more than 60% of words are capitalised
#     # (likely a title page or all-caps header section)
#     cap_ratio = sum(1 for w in words if w.isupper()
#                     and len(w) > 1) / max(len(words), 1)
#     if cap_ratio > 0.4:
#         return False

#     # Bonus: check if chunk has domain signal keywords
#     # Not required but good indicator of quality
#     has_signal = any(kw in text_lower
#                      for kw in SIGNAL_KEYWORDS)

#     # Accept if has signal keywords OR is long enough
#     # (long chunks tend to be content, not boilerplate)
#     return has_signal or len(text) > 400


BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")

# Test on Silberschatz — known to have noisy front matter
result = load_and_chunk(os.path.join(TESTED_DIR, "source_61.pdf"))
chunks = result["chunks"]

print(f"Total chunks: {len(chunks)}")
print(f"\nFirst 5 chunks — likely noise:")
for i, c in enumerate(chunks[:19]):
    verdict = "KEEP" if is_quality_chunk(c) else "REJECT"
    print(f"\nChunk {i+1} [{verdict}]:")
    print(c[:500])
    print("-" * 40)

print(f"\nChunks 20-25 — likely content:")
for i, c in enumerate(chunks[20:30], 20):
    verdict = "KEEP" if is_quality_chunk(c) else "REJECT"
    print(f"\nChunk {i+1} [{verdict}]:")
    print(c[:500])
    print("-" * 40)