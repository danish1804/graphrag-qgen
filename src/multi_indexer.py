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


