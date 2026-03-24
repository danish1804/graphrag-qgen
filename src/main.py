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
