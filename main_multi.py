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


app = FastAPI(
    title="Exam Question Generator — P000233DS",
    description=(
        "AI-powered exam question generation using RAG + Qwen 2.5 7B.\n\n"
        "**Single document:** Upload one PDF and generate questions from it.\n\n"
        "**Multi document:** Generate questions across all indexed course material."
    ),
    version="0.2.0",
    lifespan=lifespan
)


# ── Request / Response models ─────────────────────────────

class GenerateRequest(BaseModel):
    learning_objective: str
    difficulty: str         = "medium"
    question_type: str      = "conceptual"
    top_k: int              = 3

class GenerateResponse(BaseModel):
    question: str
    learning_objective: str
    difficulty: str
    question_type: str
    top_chunks_used: int
    avg_retrieval_score: float

class MultiGenerateResponse(BaseModel):
    question: str
    learning_objective: str
    difficulty: str
    question_type: str
    sources_used: list
    chunks_retrieved: int
    avg_retrieval_score: float

class UploadResponse(BaseModel):
    message: str
    file_name: str
    total_chunks: int
    index_name: str

class MultiStatusResponse(BaseModel):
    index_loaded: bool
    total_chunks: int
    total_documents: int
    documents: list

class RebuildResponse(BaseModel):
    message: str
    total_documents: int
    total_chunks: int


# ── Single Document Endpoints ─────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Check server and model status."""
    return {
        "status":               "ok",
        "models_loaded":        single_pipeline._models_loaded,
        "single_index_loaded":  single_pipeline.index is not None,
        "single_index_name":    single_pipeline.index_name,
        "multi_index_loaded":   multi_pipeline.index is not None,
        "multi_total_chunks":   len(multi_pipeline.chunks)
                                if multi_pipeline.chunks else 0,
    }


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Single Document"]
)
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF or PPTX file.
    Ingests, chunks, embeds and indexes it for single-doc generation.
    """
    allowed = [".pdf", ".pptx"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use PDF or PPTX."
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[Server] File saved: {save_path}")

    try:
        result = single_pipeline.ingest(save_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

    return UploadResponse(
        message="File ingested successfully",
        file_name=result["file_name"],
        total_chunks=result["total_chunks"],
        index_name=result["index_name"]
    )



