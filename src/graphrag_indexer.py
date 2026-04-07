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


def extract_entities_from_chunk(chunk: str, nlp) -> list:
    """
    Extract meaningful entities from a text chunk.

    Uses two strategies:
    1. spaCy NER — named entities (proper nouns, products etc.)
    2. Noun chunks — technical terms not caught by NER
       e.g. "convolutional layer", "hash table", "binary tree"

    Returns list of entity strings (lowercased, cleaned).
    """
    doc = nlp(chunk)
    entities = set()

    # Strategy 1: Named entities
    for ent in doc.ents:
        if len(ent.text.strip()) > 2:
            entities.add(ent.text.strip().lower())

    # Strategy 2: Noun chunks — critical for IT domain
    # Catches: "convolutional neural network", "parameter sharing"
    for chunk_np in doc.noun_chunks:
        text = chunk_np.text.strip().lower()
        # Filter: must be 2-50 chars, not just stopwords
        if 2 < len(text) < 50 and not chunk_np.root.is_stop:
            entities.add(text)

    return list(entities)


def build_knowledge_graph(
        chunks: list,
        nlp,
        min_entity_freq: int = MIN_ENTITY_FREQ,
        min_edge_weight: int = MIN_EDGE_WEIGHT
) -> tuple:
    """
    Build a knowledge graph from text chunks.

    Algorithm:
    1. Extract entities from every chunk
    2. Count entity frequencies
    3. Filter low-frequency entities (noise)
    4. Create edges between entities that co-occur
       in the same chunk (they are related)
    5. Return NetworkX graph + entity-to-chunks mapping

    Returns:
        graph        — NetworkX undirected weighted graph
        entity_chunks — dict: entity → list of chunks it appears in
    """
    print("Extracting entities from chunks...")

    entity_freq     = defaultdict(int)
    chunk_entities  = []   # entities per chunk
    entity_chunks   = defaultdict(list)  # entity → chunks

    # Pass 1: extract entities from all chunks
    for i, chunk in enumerate(tqdm(chunks, desc="Entity extraction")):
        entities = extract_entities_from_chunk(chunk, nlp)
        chunk_entities.append(entities)

        for ent in entities:
            entity_freq[ent] += 1
            entity_chunks[ent].append(chunk)

    # Filter: keep only entities that appear enough times
    valid_entities = {
        e for e, freq in entity_freq.items()
        if freq >= min_entity_freq
    }
    print(f"Entities before filtering : {len(entity_freq)}")
    print(f"Entities after filtering  : {len(valid_entities)}")

    # Pass 2: build graph
    print("Building knowledge graph...")
    graph = nx.Graph()

    # Add nodes
    for entity in valid_entities:
        graph.add_node(
            entity,
            frequency=entity_freq[entity],
            chunks=entity_chunks[entity][:5]  # store up to 5 example chunks
        )

    # Add edges — co-occurrence within same chunk
    edge_weights = defaultdict(int)

    for entities in tqdm(chunk_entities, desc="Building edges"):
        # Filter to valid entities only
        valid = [e for e in entities if e in valid_entities]

        # Create edges between all pairs in this chunk
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pair = tuple(sorted([valid[i], valid[j]]))
                edge_weights[pair] += 1

    # Add edges that meet minimum weight threshold
    edges_added = 0
    for (e1, e2), weight in edge_weights.items():
        if weight >= min_edge_weight:
            graph.add_edge(e1, e2, weight=weight)
            edges_added += 1

    print(f"Graph nodes : {graph.number_of_nodes()}")
    print(f"Graph edges : {graph.number_of_edges()}")

    return graph, dict(entity_chunks)

