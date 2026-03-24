# Save as test_batch.py in project root
import os
import sys
sys.path.append(os.path.dirname(__file__))
from src.document_loader import load_and_chunk

RAW_DIR = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen\data\raw"

results = []
for fname in os.listdir(RAW_DIR):
    if fname.endswith(".pdf"):
        path = os.path.join(RAW_DIR, fname)
        try:
            result = load_and_chunk(path)
            status = "OK" if result["total_chunks"] > 5 else "LOW"
            results.append((status, fname, result["total_chunks"], result["total_chars"]))
        except Exception as e:
            results.append(("FAIL", fname, 0, str(e)))

print(f"\n{'Status':<6} {'Chunks':<8} {'Chars':<10} {'File'}")
print("-" * 70)
for status, fname, chunks, chars in sorted(results):
    print(f"{status:<6} {str(chunks):<8} {str(chars):<10} {fname}")

ok   = sum(1 for r in results if r[0] == "OK")
low  = sum(1 for r in results if r[0] == "LOW")
fail = sum(1 for r in results if r[0] == "FAIL")
print(f"\nSummary: {ok} OK  |  {low} LOW  |  {fail} FAIL  |  {len(results)} total")