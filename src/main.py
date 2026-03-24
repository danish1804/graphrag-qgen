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
from src.pipeline import Pipeline, DATA_DIR


# -- Global pipeline instance ------------------------
pipeline = Pipeline()


# -- Startup: load models once when server starts ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[Server] Starting up — loading models...")
    pipeline.load_models()
    print("[Server] Models ready. Server is live.\n")
    yield
    print("[Server] Shutting down.")

app = FastAPI(
    title="Exam Question Generator",
    description="GraphRAG + Qwen 2.5 — generates exam questions from course material",
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
    learning_objective: str
    difficulty: str
    question_type: str
    top_chunks_used: int
    avg_retrieval_score: float



