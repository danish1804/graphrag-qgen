import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.document_loader import load_and_chunk

# -- Paths --------------------------------------------------------
BASE_DATA_DIR = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen\data"
RAW_DIR       = os.path.join(BASE_DATA_DIR, "raw")
TESTED_DIR    = os.path.join(BASE_DATA_DIR, "tested")
REJECTED_DIR  = os.path.join(BASE_DATA_DIR, "rejected")

# Ensure folders exist
for folder in [RAW_DIR, TESTED_DIR, REJECTED_DIR]:
    os.makedirs(folder, exist_ok=True)

results = []

print("=" * 70)
print("BATCH TEST AND ROUTING")
print("=" * 70)
print(f"Source  : {RAW_DIR}")
print(f"Tested  : {TESTED_DIR}")
print(f"Rejected: {REJECTED_DIR}")
print("=" * 70)

# -- Processing Loop ---------------------------------------------
pdf_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")]

if not pdf_files:
    print("No PDF files found in raw folder.")
    sys.exit(0)

print(f"\nFound {len(pdf_files)} PDF files. Processing...\n")

for fname in sorted(pdf_files):
    source_path = os.path.join(RAW_DIR, fname)
    print(f"Testing: {fname}")

    try:
        result = load_and_chunk(source_path)
        chunks = result["total_chunks"]
        chars  = result["total_chars"]

        if chunks > 5:
            status = "OK"
        elif chunks > 0:
            status = "LOW"
        else:
            status = "EMPTY"

        results.append((status, fname, chunks, chars))

        # Route file
        if status == "OK":
            dest = os.path.join(TESTED_DIR, fname)
            shutil.move(source_path, dest)
            print(f"  Status : OK ({chunks} chunks, {chars:,} chars) -> moved to tested/")
        else:
            dest = os.path.join(REJECTED_DIR, fname)
            shutil.move(source_path, dest)
            print(f"  Status : {status} ({chunks} chunks) -> moved to rejected/")

    except Exception as e:
        results.append(("FAIL", fname, 0, str(e)))
        # Still try to move to rejected
        try:
            dest = os.path.join(REJECTED_DIR, fname)
            shutil.move(source_path, dest)
            print(f"  Status : FAIL → moved to rejected/")
            print(f"  Error  : {str(e)[:80]}")
        except Exception as move_err:
            print(f"  Status : FAIL — could not move file either")
            print(f"  Error  : {str(e)[:80]}")

    print()

# -- Summary Report -----------------------------------------
print("=" * 70)
print(f"{'Status':<6} {'Chunks':<8} {'Chars':<12} {'File'}")
print("-" * 70)
for s, f, ck, ch in sorted(results, key=lambda x: x[0]):
    print(f"{s:<6} {str(ck):<8} {str(ch):<12} {f}")

ok     = sum(1 for r in results if r[0] == "OK")
low    = sum(1 for r in results if r[0] == "LOW")
empty  = sum(1 for r in results if r[0] == "EMPTY")
fail   = sum(1 for r in results if r[0] == "FAIL")

print("=" * 70)
print(f"OK      : {ok}  -> moved to tested/  (ready for pipeline)")
print(f"LOW     : {low}  -> moved to rejected/ (too few chunks - likely scanned)")
print(f"EMPTY   : {empty}  -> moved to rejected/ (no text extracted)")
print(f"FAIL    : {fail}  -> moved to rejected/ (load error)")
print(f"Total   : {len(results)}")
print("=" * 70)

if fail + low + empty > 0:
    print("\nRejected files are likely scanned PDFs (image-based).")
    print("These need OCR processing to extract text.")
    print("Tools: Adobe Acrobat, tesseract-ocr, or pymupdf with OCR.")