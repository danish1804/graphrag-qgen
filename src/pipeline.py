# ============================================================
# src/pipeline.py
# Connects document loader + embedder + generator
# into a single end-to-end pipeline
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.document_loader import load_and_chunk
from src.embedder import build_index, load_index, search, Embedder
from src.generator import load_model, generate_question


INDEX_DIR = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen\indexes"
DATA_DIR  = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen\data"


