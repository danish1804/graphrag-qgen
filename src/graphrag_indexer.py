# ============================================================
# src/graphrag_indexer.py
# Builds a knowledge graph from document chunks using spaCy
# for entity extraction and NetworkX for graph construction.
#
# What it does differently from embedder.py:
#   embedder.py  → stores flat text chunks in FAISS
#   this file    → extracts entities + relationships,
#                  builds a graph, then embeds graph nodes
#
# Graph structure:
#   Nodes = named entities (CNN, Pooling, Hash Table...)
#   Edges = co-occurrence relationships between entities
#           within the same chunk
# ============================================================

import os
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
from src.embedder import Embedder, DIMENSION

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
GRAPH_NAME = "graphrag"


# ── Entity extraction config ───────────────────────────────
# spaCy entity types we care about for IT domain
# PERSON, ORG = skip (not useful for exam questions)
# We keep noun chunks as pseudo-entities for technical terms
KEEP_ENTITY_TYPES = {
    "ORG", "PRODUCT", "WORK_OF_ART", "EVENT", "LAW",
    "LANGUAGE", "GPE", "NORP"
}

# Minimum entity frequency to be included in graph
MIN_ENTITY_FREQ = 2

# Minimum co-occurrence count to create an edge
MIN_EDGE_WEIGHT = 2
