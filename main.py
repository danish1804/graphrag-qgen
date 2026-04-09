# ============================================================
# main.py
# FastAPI application - wraps the pipeline
# POST /upload   -> ingest a PDF or PPTX
# POST /generate -> generate an exam question
# GET  /health   -> check server status
# ============================================================

import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(__file__))
from standard_rag.pipeline import Pipeline, DATA_DIR


# -- Global pipeline instance ------------------------
pipeline = Pipeline()


# -- Startup: load models once when server starts ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[Server] Starting up - loading models...")
    pipeline.load_models()
    print("[Server] Models ready. Server is live.\n")
    yield
    print("[Server] Shutting down.")

app = FastAPI(
    title="Exam Question Generator",
    description="GraphRAG + Qwen 2.5 -> generates exam questions from course material",
    version="0.1.0",
    lifespan=lifespan
)


# -- Request / Response models -------------------------------
class GenerateRequest(BaseModel):
    learning_objective: str
    difficulty: str = "medium"      # easy / medium / hard
    question_type: str = "conceptual"  # recall / conceptual / application / analytical
    top_k: int = 3


class GenerateResponse(BaseModel): 
    question: str
    metadata: dict

class UploadResponse(BaseModel):
    message: str
    file_name: str
    total_chunks: int
    index_name: str

# -- Endpoints ------------------------------------------------

@app.get("/health")
def health():
    """Check if server and models are ready."""
    return {
        "status": "ok",
        "models_loaded": pipeline._models_loaded,
        "index_loaded": pipeline.index is not None,
        "index_name": pipeline.index_name
    }

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF or PPTX course material file.
    The file is ingested, chunked, embedded,
    and a FAISS index is built and saved.
    """
    # Validate file type
    allowed = [".pdf", ".pptx"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use PDF or PPTX."
        )

    # Save uploaded file to data/
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[Server] File saved: {save_path}")

    # Ingest
    try:
        result = pipeline.ingest(save_path)
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

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Generate an exam question from a learning objective.
    Requires a document to be uploaded first.
    """
    if pipeline.index is None:
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet. POST to /upload first."
        )

    try:
        result = pipeline.generate(
            learning_objective=req.learning_objective,
            difficulty=req.difficulty,
            question_type=req.question_type,
            top_k=req.top_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

    return result


# -- Run --------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


