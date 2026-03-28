# ============================================================
# src/multi_indexer.py
# Builds a combined FAISS index from multiple documents
# Each chunk is tagged with its source document
# ============================================================

import os
import json
import numpy as np
import faiss
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.document_loader import load_and_chunk
from src.embedder import Embedder, DIMENSION

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
INDEX_NAME = "combined"


def build_combined_index(
        docs_dir: str   = TESTED_DIR,
        index_dir: str  = INDEX_DIR,
        index_name: str = INDEX_NAME
) -> dict:
    """
    Load all PDFs and PPTXs from docs_dir,
    chunk them, embed them, and build one
    combined FAISS index with source metadata.

    Each chunk stored with:
    - text content
    - source file name
    - chunk position within that file
    """
    os.makedirs(index_dir, exist_ok=True)

    # ── Step 1: Discover all documents ───────────────────
    supported = (".pdf", ".pptx")
    doc_files = sorted([
        f for f in os.listdir(docs_dir)
        if f.lower().endswith(supported)
    ])

    if not doc_files:
        raise FileNotFoundError(
            f"No PDF or PPTX files found in: {docs_dir}")

    print("=" * 60)
    print("MULTI-DOCUMENT INDEXER")
    print("=" * 60)
    print(f"Documents found : {len(doc_files)}")
    print(f"Source folder   : {docs_dir}")
    print(f"Index output    : {index_dir}/{index_name}")
    print("=" * 60)
