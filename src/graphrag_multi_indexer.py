# ============================================================
# src/graphrag_multi_indexer.py
# Multi-Document GraphRAG Indexer
#
# Builds ONE combined knowledge graph from all 22 PDFs.
#
# Key differences from graphrag_indexer.py:
#   graphrag_indexer.py      → one document at a time
#   graphrag_multi_indexer.py → all documents → one graph
#
# Architecture:
#   1. Load all PDFs from data/tested/
#   2. Chunk each document (reuse document_loader)
#   3. Two-pass entity extraction across ALL chunks
#      - Global DF calculated across entire corpus
#      - Same entity from different docs → ONE node
#   4. Build combined weighted graph
#      - Cross-document edges tagged and weighted
#   5. Embed graph nodes
#   6. Save combined GraphRAG index
#
# Why this fixes the pooling weakness:
#   Single doc graph: 'pooling' had 2 neighbours
#   Combined graph:   'pooling' has neighbours from
#   multiple textbooks → FAISS finds it reliably
# ============================================================

import os
import re
import json
import pickle
import numpy as np
import faiss
import spacy
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.document_loader  import load_and_chunk
from src.embedder         import Embedder, DIMENSION
from src.graphrag_indexer import (
    normalize_entity,
    is_valid_entity,
    extract_raw_entities,
    build_knowledge_graph,
    embed_graph_nodes
)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
GRAPH_NAME = "graphrag_combined"

# ── Config ─────────────────────────────────────────────────
# Slightly lower DF threshold for multi-doc
# because course codes appear in fewer total chunks
MAX_DOC_FREQ    = 0.35
MIN_ENTITY_FREQ = 3      # higher threshold for larger corpus
MIN_EDGE_WEIGHT = 2


def discover_documents(docs_dir: str) -> list:
    """Find all PDF and PPTX files in the tested folder."""
    supported = (".pdf", ".pptx")
    docs = sorted([
        f for f in os.listdir(docs_dir)
        if f.lower().endswith(supported)
    ])
    return docs


def load_all_chunks(docs_dir: str) -> tuple:
    """
    Load and chunk all documents.

    Returns:
        all_chunks  : flat list of all chunk strings
        all_sources : source filename per chunk
        doc_stats   : dict of per-document stats
    """
    docs      = discover_documents(docs_dir)
    all_chunks  = []
    all_sources = []
    doc_stats   = {}
    failed      = []

    print(f"\nLoading {len(docs)} documents...")

    for fname in tqdm(docs, desc="Loading documents"):
        fpath = os.path.join(docs_dir, fname)
        try:
            result  = load_and_chunk(fpath)
            chunks  = result["chunks"]

            all_chunks.extend(chunks)
            all_sources.extend([fname] * len(chunks))

            doc_stats[fname] = {
                "chunks": len(chunks),
                "chars":  result["total_chars"]
            }

        except Exception as e:
            print(f"  FAIL {fname}: {e}")
            failed.append(fname)

    print(f"\nDocuments loaded  : {len(docs) - len(failed)}")
    print(f"Documents failed  : {len(failed)}")
    print(f"Total chunks      : {len(all_chunks)}")

    if failed:
        print(f"Failed: {failed}")

    return all_chunks, all_sources, doc_stats
