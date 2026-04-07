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

    # ----------------------------------------------------------
    # STAGE 2 — Ingest a document
    # ----------------------------------------------------------
    def ingest(self, file_path: str) -> dict:
        """
        Load a PDF/PPTX, build knowledge graph, embed nodes,
        save GraphRAG index.

        Steps:
        1. Load and chunk document
        2. Extract entities + build graph (two-pass)
        3. Embed graph nodes
        4. Save FAISS index + graph + metadata
        """
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before ingest()")

        file_name = os.path.basename(file_path)
        self.graph_name = os.path.splitext(file_name)[0] \
                          + "_graphrag"

        print(f"\n[GraphRAGPipeline] Ingesting: {file_name}")

        # Load and chunk
        doc_result = load_and_chunk(file_path)
        chunks     = doc_result["chunks"]
        sources    = [file_name] * len(chunks)

        print(f"[GraphRAGPipeline] {len(chunks)} chunks loaded")

        # Build GraphRAG index
        summary = build_graphrag_index(
            chunks, sources,
            index_dir  = INDEX_DIR,
            graph_name = self.graph_name
        )

        # Load into memory
        self.index, self.graph, self.nodes = \
            load_graphrag_index(INDEX_DIR, self.graph_name)

        print(f"[GraphRAGPipeline] Graph ready — "
              f"{summary['graph_nodes']} nodes, "
              f"{summary['graph_edges']} edges")

        return {
            "file_name":   file_name,
            "graph_name":  self.graph_name,
            "graph_nodes": summary["graph_nodes"],
            "graph_edges": summary["graph_edges"],
            "chunks":      len(chunks)
        }
    
    # ----------------------------------------------------------
    # STAGE 3 — Load existing index
    # ----------------------------------------------------------
    def load_existing_index(self, graph_name: str):
        """Load a previously built GraphRAG index."""
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before loading index.")

        self.graph_name = graph_name
        self.index, self.graph, self.nodes = \
            load_graphrag_index(INDEX_DIR, graph_name)

        print(f"[GraphRAGPipeline] Loaded: {graph_name}")

