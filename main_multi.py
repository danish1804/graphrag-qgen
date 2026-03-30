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


