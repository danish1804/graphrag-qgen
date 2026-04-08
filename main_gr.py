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
