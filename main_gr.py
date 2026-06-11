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
from typing import Optional, Literal, Any
import argparse

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from utils.config import cfg
from utils.question_types import valid_question_types

sys.path.append(os.path.dirname(__file__))

from standard_rag.pipeline              import Pipeline
from standard_rag.multi_pipeline        import MultiPipeline
from graphrag.graphrag_pipeline     import GraphRAGPipeline
from graphrag.graphrag_domain_indexer import DOMAIN_MAP
from utils import runtime_state
from services.domain_catalog_service import DomainCatalogService
from services.graphrag_generation_service import GraphRAGGenerationService
from services.status_service import StatusService
from services.admin_domain_service import AdminDomainService
from services.index_repository import IndexRepository

BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Pipeline instances ────────────────────────────────────
single_pipeline = Pipeline()
multi_pipeline  = MultiPipeline()
graph_pipeline  = GraphRAGPipeline()
domain_catalog_service = DomainCatalogService(
    cfg=cfg,
    domain_map=DOMAIN_MAP,
    runtime_state_module=runtime_state,
)
graphrag_generation_service = GraphRAGGenerationService(graph_pipeline)
status_service = None
admin_domain_service = None

# ── Server mode ───────────────────────────────────────────
# standard : loads standard RAG only (fast startup ~15s)
# full      : loads standard RAG + all GraphRAG domains (~45s)
SERVER_MODE = "full"  # overridden by --mode argument at startup


# ── Domain catalog sync ─────────────────────────────────────
def _sync_runtime_domains() -> set[str]:
    """Backwards-compatible wrapper around the domain catalog service."""
    return domain_catalog_service.sync_domains()


status_service = StatusService(
    index_dir=INDEX_DIR,
    domain_map=DOMAIN_MAP,
    single_pipeline=single_pipeline,
    multi_pipeline=multi_pipeline,
    graph_pipeline=graph_pipeline,
)
index_repository = IndexRepository(
    index_dir=INDEX_DIR,
    graph_pipeline=graph_pipeline,
)
admin_domain_service = AdminDomainService(
    tested_dir=TESTED_DIR,
    graph_pipeline=graph_pipeline,
    index_repository=index_repository,
    runtime_state_module=runtime_state,
    sync_domains_callable=_sync_runtime_domains,
)


# ── Startup / shutdown ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and indexes at startup."""

    print("[Server] Starting up...")

    # Load models once — shared across all pipelines
    print(f"[Server] Loading embedding model and {cfg['model']['llm_name']}...")
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
        global VALID_DOMAINS
        VALID_DOMAINS = _sync_runtime_domains()
        print("[Server] Loading domain GraphRAG indexes...")
        domains_loaded = []
        for domain_name in sorted(VALID_DOMAINS):
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

    # ── Load judge model ──────────────────────────────────
    print("[Server] Loading judge model...")
    try:
        graph_pipeline.load_judge_model()
        print("[Server] Judge model ready.")
    except Exception as e:
        print(f"[Server] Judge model failed to load: {e}")
        print("[Server] Evaluations will be skipped.")

    print("[Server] Server is live.")

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
    print("[Server]   POST /admin/upload-document")
    print("[Server]   POST /admin/rebuild-domain")
    print("[Server]   POST /admin/add-domain")
    print("[Server]   GET  /admin/new-documents")
    print("[Server]   GET  /evaluate") 

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
    learning_objective: str = Field(..., min_length=3)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_type: Literal[
        "recall", "conceptual", "application", "case_study"
    ] = "conceptual"
    top_k: int = Field(default=3, ge=1, le=10)

class GraphRAGGenerateRequest(BaseModel):
    learning_objective: str = Field(..., min_length=3)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_type: Literal[
        "recall", "conceptual", "application", "case_study"
    ] = "conceptual"
    top_k_nodes: int = Field(default=3, ge=1, le=20)
    neighbour_depth: int = Field(default=1, ge=1, le=3)
    domain: Optional[str] = None  # None = auto-route

# Add after existing request models:

class AdminUploadRequest(BaseModel):
    domain: str  # which domain to add document to

class AdminRebuildRequest(BaseModel):
    domain: str = Field(..., min_length=2)  # which domain to rebuild

class AdminAddDomainRequest(BaseModel):
    domain_name: str = Field(..., min_length=2)  # new domain identifier
    description: str = Field(..., min_length=3)  # human readable description
    keywords: list[str] = Field(default_factory=list)  # routing keywords


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    standard_rag: dict[str, Any]
    graphrag: dict[str, Any]


class StandardUploadResponse(BaseModel):
    status: Literal["ok"]
    file_name: str
    total_chunks: int
    index_name: str
    message: str


class RebuildResponse(BaseModel):
    status: Literal["ok"]
    total_chunks: int
    message: str


class GraphRAGGenerateResponse(BaseModel):
    question: str
    metadata: dict[str, Any]

    class Config:
        extra = "allow"


class AdminRebuildResponse(BaseModel):
    status: Literal["ok"]
    domain: str
    update_type: str
    message: str
    new_nodes_added: int
    new_edges_added: int
    nodes_updated: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    elapsed_seconds: float = 0.0
    new_files: list[str] = Field(default_factory=list)
    skipped_duplicates: list[dict[str, Any]] = Field(default_factory=list)

# ── Validation helpers ────────────────────────────────────
VALID_DIFFICULTIES  = {"easy", "medium", "hard"}
VALID_TYPES         = valid_question_types()


# Replace hardcoded VALID_DOMAINS with config-driven version
VALID_DOMAINS = _sync_runtime_domains()

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

@app.get("/health", tags=["System"], response_model=HealthResponse)
async def health():
    """
    Server health check.
    Shows status of all three pipelines.
    """
    global VALID_DOMAINS
    VALID_DOMAINS = _sync_runtime_domains()
    return status_service.build_health(VALID_DOMAINS)

# ============================================================
# STANDARD RAG — SINGLE DOCUMENT
# ============================================================

@app.post(
    "/upload",
    tags=["Standard RAG — Single Doc"],
    response_model=StandardUploadResponse
)
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
    
@app.post(
    "/multi/rebuild",
    tags=["Standard RAG — Multi Doc"],
    response_model=RebuildResponse
)
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
    global VALID_DOMAINS
    VALID_DOMAINS = _sync_runtime_domains()
    return status_service.build_graphrag_status(VALID_DOMAINS)


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


# @app.post("/graphrag/generate", tags=["GraphRAG"])
# async def graphrag_generate(req: GraphRAGGenerateRequest):
#     """
#     Generate an exam question using GraphRAG.

#     Automatically routes the learning objective to the correct
#     domain graph (deep_learning, database, software_engineering,
#     operating_systems) using keyword matching.

#     You can override auto-routing by specifying domain explicitly.

#     Returns the question plus graph metadata:
#     - seed_nodes: entities matched to the query
#     - neighbour_nodes: related concepts from graph traversal
#     - relationships: entity connections used for context
#     - domain_routed_to: which domain graph was used
#     """
#     validate_params(req.difficulty, req.question_type)

#     # Validate explicit domain if provided
#     if req.domain and req.domain not in VALID_DOMAINS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"domain must be one of {VALID_DOMAINS} "
#                    f"or null for auto-routing."
#         )

#     try:
#         if req.domain:
#             # Explicit domain specified — load that graph
#             graph_pipeline.load_domain_index(req.domain)
#             result = graph_pipeline.generate(
#                 learning_objective = req.learning_objective,
#                 difficulty         = req.difficulty,
#                 question_type      = req.question_type,
#                 top_k_nodes        = req.top_k_nodes,
#                 neighbour_depth    = req.neighbour_depth
#             )
#             result["domain_routed_to"] = req.domain
#         else:
#             # Auto-route to correct domain
#             result = graph_pipeline.generate_with_routing(
#                 learning_objective = req.learning_objective,
#                 difficulty         = req.difficulty,
#                 question_type      = req.question_type,
#                 top_k_nodes        = req.top_k_nodes,
#                 neighbour_depth    = req.neighbour_depth
#             )

#         return result

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/graphrag/generate",
    tags=["GraphRAG"],
    response_model=GraphRAGGenerateResponse
)
async def graphrag_generate(req: GraphRAGGenerateRequest):
    """
    Generate an exam question using GraphRAG.

    Domain routing logic:
      - Valid domain specified → use that domain directly
      - Invalid domain specified → warn and fall back to auto-routing
      - No domain + hard difficulty → multi-domain retrieval
        (combines context from all domains above score threshold)
      - No domain + easy/medium → embedding-based auto-routing
        (picks single best matching domain)

    Returns question + full metadata including:
      - retrieved_chunks: text chunks passed to LLM
      - sample_answer: reference answer for educators
      - domain_routed_to / domains_used: routing decision
      - concept_relationships: graph edges used for context
    """
    validate_params(req.difficulty, req.question_type)

    global VALID_DOMAINS
    VALID_DOMAINS = _sync_runtime_domains()

    try:
        result = graphrag_generation_service.generate_with_domain_fallback(
            learning_objective=req.learning_objective,
            difficulty=req.difficulty,
            question_type=req.question_type,
            domain=req.domain,
            top_k_nodes=req.top_k_nodes,
            neighbour_depth=req.neighbour_depth,
            valid_domains=VALID_DOMAINS,
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
    global VALID_DOMAINS
    VALID_DOMAINS = _sync_runtime_domains()
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
# ADMIN ENDPOINTS
# Think of two types of users — end user and admin.
#         Admin has more access. Anything exposed to users
#         should be in the form of an API.
# ============================================================

# @app.post("/admin/upload-document", tags=["Admin"])
# async def admin_upload_document(
#     domain: str,
#     file:   UploadFile = File(...)
# ):
#     """
#     Admin endpoint — upload a new PDF/PPTX to a domain.

#     Adds the document to the domain's data folder.
#     Call POST /admin/rebuild-domain after uploading
#     to update the knowledge graph.

#     Args:
#         domain: target domain (database, deep_learning, etc.)
#         file:   PDF or PPTX file to upload
#     """
#     # Validate domain
#     if domain not in VALID_DOMAINS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid domain '{domain}'. "
#                    f"Valid domains: {VALID_DOMAINS}"
#         )

#     # Validate file type
#     filename = file.filename
#     if not filename.lower().endswith((".pdf", ".pptx")):
#         raise HTTPException(
#             status_code=400,
#             detail="Only PDF and PPTX files supported."
#         )

#     # Save to domain tested folder
#     domain_dir = os.path.join(TESTED_DIR)
#     os.makedirs(domain_dir, exist_ok=True)
#     save_path  = os.path.join(domain_dir, filename)

#     with open(save_path, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     print(f"[Admin] Uploaded {filename} -> domain: {domain}")
#     register_document(filename, domain=domain, indexed=False)

#     return {
#         "status":   "ok",
#         "message":  f"Document uploaded successfully. "
#                     f"Call POST /admin/rebuild-domain to "
#                     f"update the {domain} knowledge graph.",
#         "file":     filename,
#         "domain":   domain,
#         "saved_to": save_path
#     }

@app.post("/admin/upload-document", tags=["Admin"])
async def admin_upload_document(
    domain: str,
    file:   UploadFile = File(...)
):
    global VALID_DOMAINS
    VALID_DOMAINS = _sync_runtime_domains()
    try:
        return await admin_domain_service.upload_document(
            domain=domain,
            file=file,
            valid_domains=VALID_DOMAINS,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/admin/rebuild-domain",
    tags=["Admin"],
    response_model=AdminRebuildResponse
)
async def admin_rebuild_domain(req: AdminRebuildRequest):
    global VALID_DOMAINS
    VALID_DOMAINS = _sync_runtime_domains()
    try:
        return admin_domain_service.rebuild_domain(
            req.domain,
            VALID_DOMAINS,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/add-domain", tags=["Admin"])
async def admin_add_domain(req: AdminAddDomainRequest):
    """
    Admin endpoint — register a new subject domain.

    Suppose we want to add Information Systems.
            Just add the name here and write a build command.

    Adds domain to runtime state (not config.yml) and
    updates domain catalog for routing/rebuild operations.

    Note: Upload PDFs to data/tested/ first, then call
    this endpoint to register and build the domain.

    Args:
        domain_name:  identifier e.g. "information_systems"
        description:  human readable e.g. "Information Systems"
        keywords:     routing keywords for embedding profile
    """
    try:
        result = admin_domain_service.add_domain(
            domain_name=req.domain_name,
            description=req.description,
            keywords=req.keywords,
        )
        global VALID_DOMAINS
        VALID_DOMAINS = _sync_runtime_domains()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/admin/new-documents", tags=["Admin"])
async def get_new_documents():
    """
    Show documents in tested/ folder not yet
    assigned to any domain in the registry.
    Admin should call POST /admin/upload-document
    to assign domain and trigger indexing.
    """
    try:
        from utils.registry import load_registry

        registry = load_registry()

        # All files registered in any domain
        registered_files = set()
        for domain, info in registry["domains"].items():
            registered_files.update(
                info["files"].keys())

        # All PDFs in tested/ folder
        tested_files = set(
            f for f in os.listdir(TESTED_DIR)
            if f.endswith((".pdf", ".pptx"))
        )

        # Files in tested/ but not in any domain
        unregistered = sorted(
            tested_files - registered_files)

        return {
            "status":             "ok",
            "total_unregistered": len(unregistered),
            "unregistered_files": unregistered,
            "total_in_tested":    len(tested_files),
            "total_registered":   len(registered_files),
            "message":
                "All documents registered. "
                "No pending files."
                if not unregistered else
                f"{len(unregistered)} file(s) in tested/ "
                f"not yet assigned to a domain. "
                f"Call POST /admin/upload-document "
                f"?domain=<domain>&file=<filename>"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e))

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