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


class Pipeline:
    def __init__(self):
        self.index        = None
        self.chunks       = None
        self.embedder     = None
        self.model        = None
        self.tokenizer    = None
        self.index_name   = None
        self._models_loaded = False

    # ----------------------------------------------------------
    # STAGE 1 - Load models into memory (call once at startup)
    # ----------------------------------------------------------
    def load_models(self):
        """Load embedding model and Qwen 2.5 into memory."""
        if self._models_loaded:
            print("Models already loaded.")
            return

        print("\n[Pipeline] Loading embedding model...")
        self.embedder = Embedder()

        print("\n[Pipeline] Loading Qwen 2.5 7B...")
        self.model, self.tokenizer = load_model()

        self._models_loaded = True
        print("\n[Pipeline] All models ready.")




