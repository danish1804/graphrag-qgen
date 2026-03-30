# ============================================================
# main.py
# FastAPI application
# Supports both single-document and multi-document pipelines
#
# Single document endpoints:
#   GET  /health
#   POST /upload
#   POST /generate
#
# Multi document endpoints:
#   GET  /multi/status
#   POST /multi/generate
#   POST /multi/rebuild
# ============================================================

import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(__file__))

from src.pipeline       import Pipeline,      DATA_DIR
from src.multi_pipeline import MultiPipeline, INDEX_DIR, INDEX_NAME


# ── Global pipeline instances ─────────────────────────────
single_pipeline = Pipeline()
multi_pipeline  = MultiPipeline()


# ── Startup ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[Server] Starting up...")

    # Load models once — shared between both pipelines
    print("[Server] Loading embedding model and Qwen 2.5...")
    single_pipeline.load_models()

    # Share the already-loaded models with multi pipeline
    # No need to load twice
    multi_pipeline.embedder   = single_pipeline.embedder
    multi_pipeline.model      = single_pipeline.model
    multi_pipeline.tokenizer  = single_pipeline.tokenizer
    multi_pipeline._models_loaded = True
    print("[Server] Models shared between pipelines.")

    # Load combined index if it exists
    combined_index_path = os.path.join(
        INDEX_DIR, f"{INDEX_NAME}.faiss")

    if os.path.exists(combined_index_path):
        try:
            multi_pipeline.load_index()
            print("[Server] Combined index loaded.")
        except Exception as e:
            print(f"[Server] Could not load combined index: {e}")
            print("[Server] Run POST /multi/rebuild to build it.")
    else:
        print("[Server] No combined index found.")
        print("[Server] Run POST /multi/rebuild to build it.")

    print("[Server] Server is live.\n")
    yield
    print("[Server] Shutting down.")



