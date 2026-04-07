import os, sys
sys.path.append(os.path.dirname(__file__))
from src.document_loader import load_and_chunk

BASE = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
TESTED = os.path.join(BASE, "data", "tested")

for fname in sorted(os.listdir(TESTED)):
    if fname.endswith(".pdf"):
        path = os.path.join(TESTED, fname)
        result = load_and_chunk(path)
        # Show first 200 chars of first chunk
        preview = result["chunks"][0][:200].replace("\n", " ")
        print(f"\n{fname} ({result['total_chunks']} chunks)")
        print(f"  {preview}")