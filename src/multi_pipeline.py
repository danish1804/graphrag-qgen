# ============================================================
# src/multi_pipeline.py
# Multi-document pipeline — retrieves across all indexed PDFs
# Uses combined FAISS index built by multi_indexer.py
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.multi_indexer import (
    build_combined_index,
    load_combined_index,
    search_combined
)
from src.embedder import Embedder
from src.generator import load_model, generate_question


BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_NAME = "combined"


class MultiPipeline:
    def __init__(self):
        self.index        = None
        self.chunks       = None
        self.meta         = None
        self.embedder     = None
        self.model        = None
        self.tokenizer    = None
        self._models_loaded = False
    
    # ----------------------------------------------------------
    # STAGE 1 — Load models
    # ----------------------------------------------------------
    def load_models(self):
        """Load embedding model and Qwen 2.5 into memory."""
        if self._models_loaded:
            print("Models already loaded.")
            return

        print("\n[MultiPipeline] Loading embedding model...")
        self.embedder = Embedder()

        print("\n[MultiPipeline] Loading Qwen 2.5 7B...")
        self.model, self.tokenizer = load_model()

        self._models_loaded = True
        print("\n[MultiPipeline] All models ready.")

    # ----------------------------------------------------------
    # STAGE 2 — Load combined index
    # ----------------------------------------------------------
    def load_index(self, index_dir: str = INDEX_DIR,
                   index_name: str = INDEX_NAME):
        """Load the pre-built combined FAISS index."""
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before load_index()")

        self.index, self.chunks, self.meta = load_combined_index(
            index_dir, index_name)

        print(f"[MultiPipeline] Index loaded — "
              f"{len(self.chunks)} chunks across "
              f"{len(set(m['source'] for m in self.meta))} documents")