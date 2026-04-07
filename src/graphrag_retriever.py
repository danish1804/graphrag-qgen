# ============================================================
# src/graphrag_retriever.py
# GraphRAG Retriever
#
# How it differs from standard FAISS retrieval:
#
# Standard RAG:
#   query → embed → find similar chunks → return chunks
#
# GraphRAG:
#   query → embed → find similar NODES (entities)
#          → traverse graph to get neighbours
#          → collect context chunks from node + neighbours
#          → return enriched context with relationships
#
# This enables multi-hop retrieval:
#   "parameter sharing" → neighbours include
#   "sparse connectivity", "weight filter", "feature extraction"
#   → question can reference ALL these related concepts
# ============================================================

import numpy as np
import networkx as nx
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.embedder import Embedder


def retrieve_graph_context(
        query: str,
        index,
        graph: nx.Graph,
        nodes: list,
        embedder: Embedder,
        top_k_nodes: int = 3,
        neighbour_depth: int = 1,
        max_context_chunks: int = 6
) -> dict:
    """
    GraphRAG retrieval — find seed nodes then traverse graph.

    Steps:
    1. Embed query
    2. Find top_k_nodes most similar graph nodes (seed nodes)
    3. For each seed node, get its neighbours up to neighbour_depth
    4. Collect context chunks from seed + neighbour nodes
    5. Deduplicate and assemble enriched context

    Args:
        query            : learning objective string
        index            : FAISS index over graph node embeddings
        graph            : NetworkX knowledge graph
        nodes            : ordered list of node names
        embedder         : BAAI embedding model
        top_k_nodes      : how many seed nodes to find (default 3)
        neighbour_depth  : how many hops to traverse (default 1)
        max_context_chunks: max chunks to include in context

    Returns:
        dict with:
          context       — assembled text for Qwen prompt
          seed_nodes    — entities directly matched to query
          neighbour_nodes — related entities found by traversal
          relationships — list of (entity1, entity2, weight) tuples
          sources       — source documents contributing context
    """
    # Step 1: Embed query
    query_embedding = embedder.model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    # Step 2: Find seed nodes
    k = min(top_k_nodes, len(nodes))
    distances, indices = index.search(query_embedding, k)

    seed_nodes = []
    for score, idx in zip(distances[0], indices[0]):
        if idx >= 0 and idx < len(nodes):
            node_name = nodes[idx]
            if node_name in graph:
                seed_nodes.append({
                    "entity": node_name,
                    "score":  float(score),
                    "freq":   graph.nodes[node_name].get(
                                  "frequency", 0)
                })

    if not seed_nodes:
        return {
            "context":          "",
            "seed_nodes":       [],
            "neighbour_nodes":  [],
            "relationships":    [],
            "sources":          []
        }
    
    # ── Step 3: Graph traversal ───────────────────────────
    visited        = set()
    neighbour_nodes = []
    relationships  = []

    for seed in seed_nodes:
        entity = seed["entity"]
        visited.add(entity)

        # Get immediate neighbours (depth=1)
        if entity in graph:
            neighbours = list(graph.neighbors(entity))

            # Sort neighbours by edge weight
            # (higher weight = stronger relationship)
            neighbours_sorted = sorted(
                neighbours,
                key=lambda n: graph[entity][n].get("weight", 1),
                reverse=True
            )

            for neighbour in neighbours_sorted[:5]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    edge_data = graph[entity][neighbour]

                    neighbour_nodes.append({
                        "entity":  neighbour,
                        "related_to": entity,
                        "weight":  edge_data.get("weight", 1),
                        "cross_source": edge_data.get(
                            "cross_source", False)
                    })

                    relationships.append((
                        entity,
                        neighbour,
                        edge_data.get("weight", 1)
                    ))

                    # Depth 2 traversal if requested
                    if neighbour_depth >= 2:
                        for n2 in graph.neighbors(neighbour):
                            if n2 not in visited:
                                visited.add(n2)
                                e2 = graph[neighbour][n2]
                                neighbour_nodes.append({
                                    "entity":     n2,
                                    "related_to": neighbour,
                                    "weight":     e2.get("weight", 1),
                                    "cross_source": e2.get(
                                        "cross_source", False)
                                })

     # ── Step 4: Collect context chunks ───────────────────
    chunk_pool  = []
    seen_chunks = set()
    sources     = set()

    # Seed node chunks first (highest priority)
    for seed in seed_nodes:
        node_data = graph.nodes.get(seed["entity"], {})
        for chunk in node_data.get("chunks", []):
            key = chunk[:80].strip()
            if key not in seen_chunks:
                seen_chunks.add(key)
                chunk_pool.append({
                    "text":   chunk,
                    "entity": seed["entity"],
                    "type":   "seed",
                    "score":  seed["score"]
                })

    # Neighbour chunks second
    for nb in neighbour_nodes:
        node_data = graph.nodes.get(nb["entity"], {})
        for chunk in node_data.get("chunks", []):
            key = chunk[:80].strip()
            if key not in seen_chunks:
                seen_chunks.add(key)
                chunk_pool.append({
                    "text":    chunk,
                    "entity":  nb["entity"],
                    "type":    "neighbour",
                    "score":   nb["weight"] / 10.0
                })

    # Collect sources
    for item in chunk_pool:
        node_data = graph.nodes.get(item["entity"], {})
        for src in node_data.get("sources", []):
            sources.add(src)

    # Limit chunks
    chunk_pool = chunk_pool[:max_context_chunks]

