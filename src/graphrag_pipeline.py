# ============================================================
# src/graphrag_pipeline.py
# GraphRAG Pipeline — connects graph retrieval to Qwen 2.5
#
# Difference from pipeline.py:
#   pipeline.py         → flat chunk retrieval (standard RAG)
#   graphrag_pipeline.py → graph-aware retrieval (GraphRAG)
#
# The generator (Qwen 2.5) is identical in both.
# Only the retrieval layer changes.
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.document_loader  import load_and_chunk
from src.graphrag_indexer import (build_graphrag_index,
                                   load_graphrag_index)
from src.graphrag_retriever import retrieve_graph_context
from src.embedder          import Embedder
from src.generator         import load_model, generate_question


BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
DATA_DIR   = os.path.join(BASE_DIR, "data", "tested")


class GraphRAGPipeline:

    def __init__(self):
        self.index        = None
        self.graph        = None
        self.nodes        = None
        self.embedder     = None
        self.model        = None
        self.tokenizer    = None
        self.graph_name   = None
        self._models_loaded = False

    # ----------------------------------------------------------
    # STAGE 1 — Load models (call once at startup)
    # ----------------------------------------------------------
    def load_models(self):
        """Load embedding model and Qwen 2.5 into memory."""
        if self._models_loaded:
            print("Models already loaded.")
            return

        print("\n[GraphRAGPipeline] Loading embedding model...")
        self.embedder = Embedder()

        print("\n[GraphRAGPipeline] Loading Qwen 2.5 7B...")
        self.model, self.tokenizer = load_model()

        self._models_loaded = True
        print("\n[GraphRAGPipeline] All models ready.")
