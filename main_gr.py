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
import argparse

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.config import cfg

sys.path.append(os.path.dirname(__file__))

from standard_rag.pipeline              import Pipeline
from standard_rag.multi_pipeline        import MultiPipeline
from graphrag.graphrag_pipeline     import GraphRAGPipeline
from graphrag.graphrag_domain_indexer import (
    DOMAIN_MAP, route_query_to_domain)

BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Pipeline instances ────────────────────────────────────
single_pipeline = Pipeline()
multi_pipeline  = MultiPipeline()
graph_pipeline  = GraphRAGPipeline()

# ── Server mode ───────────────────────────────────────────
# standard : loads standard RAG only (fast startup ~15s)
# full      : loads standard RAG + all GraphRAG domains (~45s)
SERVER_MODE = "full"  # overridden by --mode argument at startup


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

    # Load GraphRAG only in full mode
    if SERVER_MODE == "full":
        print("[Server] Loading domain GraphRAG indexes...")
        domains_loaded = []
        for domain_name in DOMAIN_MAP:
            try:
                graph_pipeline.load_domain_index(domain_name)
                domains_loaded.append(domain_name)
                print(f"[Server]   {domain_name} loaded.")
            except Exception as e:
                print(f"[Server]   {domain_name} failed: {e}")

        print(f"[Server] GraphRAG domains loaded: {domains_loaded}")

        # Build domain profile embeddings for semantic routing
        print("[Server] Building domain profile embeddings...")
        from graphrag.graphrag_domain_indexer import build_domain_profiles
        build_domain_profiles(single_pipeline.embedder)
        print("[Server] Semantic routing ready.")
    else:
        print("[Server] Standard mode — GraphRAG skipped.")
        print("[Server] /graphrag/* endpoints will return 503.")

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


# Replace hardcoded VALID_DOMAINS with config-driven version
VALID_DOMAINS = set(
    domain for domain, settings in cfg["domains"].items()
    if settings["enabled"]
)

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

    # Standard RAG multi doc
    multi_loaded  = multi_pipeline.index is not None
    multi_chunks  = 0
    if multi_loaded:
        try:
            multi_chunks = len(multi_pipeline.chunks)
        except Exception:
            pass

    # GraphRAG domain status
    graph_domains = {}
    for domain_name in DOMAIN_MAP:
        graph_path = os.path.join(
            INDEX_DIR,
            f"graphrag_{domain_name}_graph.pkl")
        graph_domains[domain_name] = os.path.exists(graph_path)

    return {
        "status":              "ok",
        "models_loaded":       single_pipeline._models_loaded,
        "standard_rag": {
            "single_doc": {
                "loaded":     single_loaded,
                "index_name": single_name
            },
            "multi_doc": {
                "loaded":       multi_loaded,
                "total_chunks": multi_chunks
            }
        },
        "graphrag": {
            "domains_available": graph_domains,
            "current_index":     graph_pipeline.graph_name
        }
    }

# ============================================================
# STANDARD RAG — SINGLE DOCUMENT
# ============================================================

@app.post("/upload", tags=["Standard RAG — Single Doc"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or PPTX.
    Chunks and indexes it for single-document RAG generation.
    """
    filename = file.filename
    if not filename.lower().endswith((".pdf", ".pptx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and PPTX files supported."
        )

    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = single_pipeline.ingest(save_path)
        return {
            "status":       "ok",
            "file_name":    result["file_name"],
            "total_chunks": result["total_chunks"],
            "index_name":   result["index_name"],
            "message":      "Document indexed. "
                            "Call POST /generate to generate questions."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", tags=["Standard RAG — Single Doc"])
async def generate_question(req: GenerateRequest):
    """
    Generate an exam question from the uploaded document.
    Requires a prior call to POST /upload.
    """
    validate_params(req.difficulty, req.question_type)

    if single_pipeline.index is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded. Call POST /upload first."
        )

    try:
        result = single_pipeline.generate(
            learning_objective = req.learning_objective,
            difficulty         = req.difficulty,
            question_type      = req.question_type,
            top_k              = req.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# STANDARD RAG — MULTI DOCUMENT
# ============================================================

@app.get("/multi/status", tags=["Standard RAG — Multi Doc"])
async def multi_status():
    """Show combined index stats — all 22 documents."""
    if multi_pipeline.index is None:
        return {
            "index_loaded":  False,
            "message":       "Call POST /multi/rebuild to build index."
        }

    # Get unique documents from chunk metadata
    try:
        docs = sorted(set(
            m["source"] for m in multi_pipeline.meta
        ))
    except Exception:
        docs = []

    return {
        "index_loaded":    True,
        "total_chunks":    len(multi_pipeline.chunks),
        "total_documents": len(docs),
        "documents":       sorted(docs)
    }

@app.post("/multi/generate", tags=["Standard RAG — Multi Doc"])
async def multi_generate(req: GenerateRequest):
    """
    Generate question from combined index across all 22 docs.
    Auto-routes to most relevant documents.
    """
    validate_params(req.difficulty, req.question_type)

    if multi_pipeline.index is None:
        raise HTTPException(
            status_code=400,
            detail="Combined index not loaded. "
                   "Call POST /multi/rebuild first."
        )

    try:
        result = multi_pipeline.generate(
            learning_objective = req.learning_objective,
            difficulty         = req.difficulty,
            question_type      = req.question_type,
            top_k              = req.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/multi/rebuild", tags=["Standard RAG — Multi Doc"])
async def multi_rebuild():
    """
    Rebuild the combined FAISS index from all documents
    in data/tested/. Run after adding new documents.
    """
    try:
        multi_pipeline.rebuild_index()
        return {
            "status":       "ok",
            "total_chunks": len(multi_pipeline.chunks),
            "message":      "Combined index rebuilt successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ============================================================
# GRAPHRAG ENDPOINTS
# ============================================================

@app.get("/graphrag/status", tags=["GraphRAG"])
async def graphrag_status():
    """
    Show status of all domain GraphRAG indexes.
    Includes node/edge counts for each domain.
    """
    import json

    domain_stats = {}

    for domain_name in DOMAIN_MAP:
        meta_path = os.path.join(
            INDEX_DIR,
            f"graphrag_{domain_name}_meta.json")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            domain_stats[domain_name] = {
                "available":    True,
                "graph_nodes":  meta.get("total_nodes", 0),
                "graph_edges":  meta.get("total_edges", 0),
                "cross_source_edges": meta.get(
                    "cross_source_edges", 0),
                "total_chunks": meta.get("total_chunks", 0),
                "documents":    meta.get("documents",
                    DOMAIN_MAP[domain_name]["files"]),
                "description":  DOMAIN_MAP[domain_name][
                    "description"]
            }
        else:
            domain_stats[domain_name] = {
                "available":   False,
                "description": DOMAIN_MAP[domain_name][
                    "description"],
                "message":     f"Run POST /graphrag/rebuild/"
                               f"{domain_name} to build."
            }

    return {
        "graphrag_available": any(
            v["available"]
            for v in domain_stats.values()
        ),
        "current_index": graph_pipeline.graph_name,
        "domains":       domain_stats
    }


@app.post("/graphrag/upload", tags=["GraphRAG"])
async def graphrag_upload(file: UploadFile = File(...)):
    """
    Upload a PDF or PPTX and build a GraphRAG index for it.
    The domain is auto-detected from the filename and content.
    The document is added to the appropriate domain graph.

    Note: For full domain graph rebuilding use
    POST /graphrag/rebuild/{domain_name}.
    This endpoint builds a single-document GraphRAG index
    for immediate use.
    """
    filename = file.filename
    if not filename.lower().endswith((".pdf", ".pptx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and PPTX files supported."
        )

    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = graph_pipeline.ingest(save_path)
        return {
            "status":       "ok",
            "file_name":    result["file_name"],
            "graph_name":   result["graph_name"],
            "graph_nodes":  result["graph_nodes"],
            "graph_edges":  result["graph_edges"],
            "chunks":       result["chunks"],
            "message":      "GraphRAG index built. "
                            "Call POST /graphrag/generate "
                            "to generate questions."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graphrag/generate", tags=["GraphRAG"])
async def graphrag_generate(req: GraphRAGGenerateRequest):
    """
    Generate an exam question using GraphRAG.

    Automatically routes the learning objective to the correct
    domain graph (deep_learning, database, software_engineering,
    operating_systems) using keyword matching.

    You can override auto-routing by specifying domain explicitly.

    Returns the question plus graph metadata:
    - seed_nodes: entities matched to the query
    - neighbour_nodes: related concepts from graph traversal
    - relationships: entity connections used for context
    - domain_routed_to: which domain graph was used
    """
    validate_params(req.difficulty, req.question_type)

    # Validate explicit domain if provided
    if req.domain and req.domain not in VALID_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"domain must be one of {VALID_DOMAINS} "
                   f"or null for auto-routing."
        )

    try:
        if req.domain:
            # Explicit domain specified — load that graph
            graph_pipeline.load_domain_index(req.domain)
            result = graph_pipeline.generate(
                learning_objective = req.learning_objective,
                difficulty         = req.difficulty,
                question_type      = req.question_type,
                top_k_nodes        = req.top_k_nodes,
                neighbour_depth    = req.neighbour_depth
            )
            result["domain_routed_to"] = req.domain
        else:
            # Auto-route to correct domain
            result = graph_pipeline.generate_with_routing(
                learning_objective = req.learning_objective,
                difficulty         = req.difficulty,
                question_type      = req.question_type,
                top_k_nodes        = req.top_k_nodes,
                neighbour_depth    = req.neighbour_depth
            )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graphrag/rebuild/{domain_name}",
          tags=["GraphRAG"])
async def graphrag_rebuild(domain_name: str):
    """
    Rebuild the GraphRAG index for a specific domain.
    Use after adding new documents to a domain.

    domain_name must be one of:
      deep_learning, database,
      software_engineering, operating_systems
    """
    if domain_name not in VALID_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"domain_name must be one of {VALID_DOMAINS}"
        )

    try:
        from graphrag.graphrag_domain_indexer import build_domain_graph
        summary = build_domain_graph(domain_name)

        # Reload into pipeline
        graph_pipeline.load_domain_index(domain_name)

        return {
            "status":       "ok",
            "domain":       domain_name,
            "graph_nodes":  summary["graph_nodes"],
            "graph_edges":  summary["graph_edges"],
            "files_loaded": summary["files_loaded"],
            "message":      f"GraphRAG index rebuilt for "
                            f"{domain_name}."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Exam Question Generator API")
    parser.add_argument(
        "--mode",
        choices=["standard", "full"],
        default="full",
        help=(
            "standard: load Standard RAG only (fast, ~15s). "
            "full: load Standard RAG + GraphRAG domains (~45s)."
        )
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    args = parser.parse_args()

    # Set global mode before lifespan runs
    SERVER_MODE = args.mode

    print(f"[Server] Starting in {SERVER_MODE.upper()} mode...")
    if SERVER_MODE == "standard":
        print("[Server] Standard mode: Standard RAG endpoints only")
        print("[Server] Endpoints: /health /upload /generate "
              "/multi/status /multi/generate /multi/rebuild")
    else:
        print("[Server] Full mode: Standard RAG + GraphRAG endpoints")

    uvicorn.run(
        "main_gr:app",
        host   = "0.0.0.0",
        port   = args.port,
        reload = False
    )