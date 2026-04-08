# ============================================================
# main.py
# FastAPI server — Standard RAG + GraphRAG endpoints
#
# Endpoint groups:
#   Standard RAG (single doc) : /upload, /generate
#   Standard RAG (multi doc)  : /multi/status, /multi/generate,
#                                /multi/rebuild
#   GraphRAG                  : /graphrag/status, /graphrag/upload
#                                /graphrag/generate
#   System                    : /health
# ============================================================

import os
import sys
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))

from src.pipeline              import Pipeline
from src.multi_pipeline        import MultiPipeline
from src.graphrag_pipeline     import GraphRAGPipeline
from src.graphrag_domain_indexer import (
    DOMAIN_MAP, route_query_to_domain)

BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Pipeline instances ────────────────────────────────────
single_pipeline = Pipeline()
multi_pipeline  = MultiPipeline()
graph_pipeline  = GraphRAGPipeline()


# ── Startup / shutdown ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and indexes at startup."""

    print("[Server] Starting up...")

    # Load models once — shared across all pipelines
    print("[Server] Loading embedding model and Qwen 2.5...")
    single_pipeline.load_models()

    # Share models — no double loading
    multi_pipeline.embedder       = single_pipeline.embedder
    multi_pipeline.model          = single_pipeline.model
    multi_pipeline.tokenizer      = single_pipeline.tokenizer
    multi_pipeline._models_loaded = True

    graph_pipeline.embedder       = single_pipeline.embedder
    graph_pipeline.model          = single_pipeline.model
    graph_pipeline.tokenizer      = single_pipeline.tokenizer
    graph_pipeline._models_loaded = True

    print("[Server] Models shared across all pipelines.")

    # Load combined standard RAG index
    try:
        multi_pipeline.load_index()
        print("[Server] Combined standard RAG index loaded.")
    except Exception as e:
        print(f"[Server] Combined index not found: {e}")
        print("[Server] Call POST /multi/rebuild to build it.")

    # Load all domain GraphRAG indexes
    print("[Server] Loading domain GraphRAG indexes...")
    domains_loaded = []
    for domain_name in DOMAIN_MAP:
        graph_path = os.path.join(
            INDEX_DIR, f"graphrag_{domain_name}_graph.pkl")
        if os.path.exists(graph_path):
            try:
                graph_pipeline.load_domain_index(domain_name)
                domains_loaded.append(domain_name)
            except Exception as e:
                print(f"[Server] {domain_name} graph failed: {e}")
        else:
            print(f"[Server] {domain_name} graph not found — "
                  f"call POST /graphrag/rebuild/{domain_name}")

    if domains_loaded:
        print(f"[Server] GraphRAG domains loaded: "
              f"{domains_loaded}")

    print("[Server] Server is live.")
    print("[Server] Endpoints ready:")
    print("[Server]   GET  /health")
    print("[Server]   POST /upload")
    print("[Server]   POST /generate")
    print("[Server]   GET  /multi/status")
    print("[Server]   POST /multi/generate")
    print("[Server]   POST /multi/rebuild")
    print("[Server]   GET  /graphrag/status")
    print("[Server]   POST /graphrag/upload")
    print("[Server]   POST /graphrag/generate")

    yield

    print("[Server] Shutting down.")


# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title       = "AI Exam Question Generator",
    description = "GraphRAG-powered exam question generation "
                  "for EdGenAI Technologies — P000233DS",
    version     = "2.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Request / Response models ─────────────────────────────
class GenerateRequest(BaseModel):
    learning_objective: str
    difficulty:    str = "medium"
    question_type: str = "conceptual"
    top_k:         int = 3

class GraphRAGGenerateRequest(BaseModel):
    learning_objective: str
    difficulty:     str = "medium"
    question_type:  str = "conceptual"
    top_k_nodes:    int = 3
    neighbour_depth: int = 1
    domain:         Optional[str] = None  # None = auto-route

# ── Validation helpers ────────────────────────────────────
VALID_DIFFICULTIES  = {"easy", "medium", "hard"}
VALID_TYPES         = {"recall", "conceptual", "application"}
VALID_DOMAINS       = set(DOMAIN_MAP.keys())

def validate_params(difficulty: str, question_type: str):
    if difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(
            status_code=400,
            detail=f"difficulty must be one of "
                   f"{VALID_DIFFICULTIES}"
        )
    if question_type not in VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"question_type must be one of "
                   f"{VALID_TYPES}"
        )
    
# ============================================================
# SYSTEM ENDPOINTS
# ============================================================

@app.get("/health", tags=["System"])
async def health():
    """
    Server health check.
    Shows status of all three pipelines.
    """
    # Standard RAG single doc
    single_loaded = single_pipeline.index is not None
    single_name   = getattr(single_pipeline, "index_name", None)