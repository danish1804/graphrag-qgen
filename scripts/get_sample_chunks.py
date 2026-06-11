# save as get_sample_chunks.py
import os
import sys
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.document_loader import load_and_chunk

BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")

# One representative file per domain
SAMPLES = {
    "deep_learning":         "source_59.pdf",
    "database":              "source_8.pdf",
    "software_engineering":  "source_22.pdf",
    "operating_systems":     "source_5.pdf",
    "computer_architecture": "source_38.pdf",
}

random.seed(42)

for domain, fname in SAMPLES.items():
    fpath = os.path.join(TESTED_DIR, fname)
    result = load_and_chunk(fpath)
    chunks = [c for c in result["chunks"] if len(c) > 300]
    sample = random.choice(chunks[:100])
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain} ({fname})")
    print(f"{'='*60}")
    print(sample[:500])