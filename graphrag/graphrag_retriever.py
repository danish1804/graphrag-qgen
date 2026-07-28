# ============================================================
# src/graphrag_retriever.py
# GraphRAG Retriever
#
# How it differs from standard FAISS retrieval:
#
# Standard RAG:
#   query -> embed -> find similar chunks -> return chunks
#
# GraphRAG:
#   query -> embed -> find similar NODES (entities)
#          -> traverse graph to get neighbours
#          -> collect context chunks from node + neighbours
#          -> return enriched context with relationships
#
# This enables multi-hop retrieval:
#   "parameter sharing" -> neighbours include
#   "sparse connectivity", "weight filter", "feature extraction"
#   -> question can reference ALL these related concepts
# ============================================================

import numpy as np
import networkx as nx
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.embedder import Embedder


def _degree_discounted_weight(graph, src, nbr, penalty):
    """Edge weight discounted by the neighbour's overall degree
    (GraphRAG 1.2, TF-IDF style), so generic hub terms — very high
    degree — stop dominating neighbour selection. penalty=0 reproduces
    the raw edge-weight ranking; higher penalty favours specific,
    lower-degree neighbours.
    """
    w = graph[src][nbr].get("weight", 1)
    deg = max(graph.degree(nbr), 1)
    return w / (deg ** penalty)


def retrieve_graph_context(
        query: str,
        index,
        graph: nx.Graph,
        nodes: list,
        embedder: Embedder,
        top_k_nodes: int = 3,
        neighbour_depth: int = 1,
        max_context_chunks: int = 6,
        seed_ratio: float = 0.6,
        rerank: bool = True,
        depth2_cap: int = 50,
        degree_penalty: float = 0.5
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
    
    # Step 3: Graph traversal
    visited        = set()
    neighbour_nodes = []
    relationships  = []

    for seed in seed_nodes:
        entity = seed["entity"]
        visited.add(entity)

        # Get immediate neighbours (depth=1)
        if entity in graph:
            neighbours = list(graph.neighbors(entity))

            # Sort neighbours by degree-discounted edge weight so
            # generic hub terms stop dominating (GraphRAG 1.2).
            neighbours_sorted = sorted(
                neighbours,
                key=lambda n: _degree_discounted_weight(
                    graph, entity, n, degree_penalty),
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

                    # Depth 2 traversal if requested — sorted by edge
                    # weight and capped (depth2_cap). Re-ranking already
                    # handles relevance, so this cap is a compute /
                    # robustness bound, not a quality lever. cap=50 keeps
                    # quality parity with the uncapped pool on the tuning
                    # query while shrinking it ~6x (741 -> 129).
                    # Previously unbounded (Bani review, Problem B).
                    if neighbour_depth >= 2:
                        n2_sorted = sorted(
                            graph.neighbors(neighbour),
                            key=lambda x: _degree_discounted_weight(
                                graph, neighbour, x, degree_penalty),
                            reverse=True
                        )
                        for n2 in n2_sorted[:depth2_cap]:
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

     # Step 4: Collect context chunks
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
                "score":  seed["score"],
                "source": list(graph.nodes.get(
                    seed["entity"], {}
                ).get("sources", ["unknown"]))[0]  # ← add this
            })

    # Neighbour chunks second
    for nb in neighbour_nodes:
        node_data = graph.nodes.get(nb["entity"], {})
        for chunk in node_data.get("chunks", []):
            key = chunk[:80].strip()
            if key not in seen_chunks:
                seen_chunks.add(key)
                # And for neighbour chunks:
                chunk_pool.append({
                    "text":    chunk,
                    "entity":  nb["entity"],
                    "type":    "neighbour",
                    "score":   nb["weight"] / 10.0,
                    "source":  list(graph.nodes.get(
                        nb["entity"], {}
                    ).get("sources", ["unknown"]))[0]  # ← add this
                })

    # Collect sources
    for item in chunk_pool:
        node_data = graph.nodes.get(item["entity"], {})
        for src in node_data.get("sources", []):
            sources.add(src)

    # Apply seed ratio from config to limit how many chunks come from seed vs neighbour nodes
    # Seed chunks are directly relevant to query
    # Neighbour chunks are related concepts via graph edges
    seed_chunks      = [c for c in chunk_pool
                        if c.get("type") == "seed"]
    neighbour_chunks = [c for c in chunk_pool
                        if c.get("type") == "neighbour"]

    # --- Query-time re-ranking (Bani review, Problem B / option 3) ---
    # Order chunks by cosine similarity to the LIVE query instead of
    # graph-insertion order, so off-topic neighbours are pushed below
    # query-relevant chunks before the budget is applied. Preserves the
    # seed vs neighbour split (seed_ratio) — we only re-order WITHIN each
    # group. Set rerank=False to reproduce the old behaviour.
    if rerank and chunk_pool:
        chunk_embs = embedder.model.encode(
            [c["text"] for c in chunk_pool],
            normalize_embeddings=True,
            batch_size=32
        ).astype(np.float32)
        qv = query_embedding[0]  # already L2-normalised
        for c, emb in zip(chunk_pool, chunk_embs):
            c["relevance"] = float(np.dot(emb, qv))  # cosine similarity
        seed_chunks.sort(key=lambda c: c.get("relevance", 0.0),
                         reverse=True)
        neighbour_chunks.sort(key=lambda c: c.get("relevance", 0.0),
                              reverse=True)

    # Calculate slot allocation from config seed_ratio
    seed_limit      = max(1, round(max_context_chunks * seed_ratio))
    neighbour_limit = max_context_chunks - seed_limit

    print(f"[Retriever] Chunk allocation: "
        f"{seed_limit} seed + {neighbour_limit} neighbour "
        f"(ratio={seed_ratio}, max={max_context_chunks})")

    chunk_pool = (
        seed_chunks[:seed_limit] +
        neighbour_chunks[:neighbour_limit]
    )

    print(f"[Retriever] Final pool: "
        f"{len([c for c in chunk_pool if c.get('type')=='seed'])} seed, "
        f"{len([c for c in chunk_pool if c.get('type')=='neighbour'])} neighbour")

    # Step 5: Assemble context
    context_parts = []

    # Header showing what was found
    seed_names = [s["entity"] for s in seed_nodes]
    nb_names   = [n["entity"] for n in neighbour_nodes[:5]]

    context_parts.append(
        f"[Key concepts identified: "
        f"{', '.join(seed_names)}]"
    )

    if nb_names:
        context_parts.append(
            f"[Related concepts: "
            f"{', '.join(nb_names)}]"
        )

    if relationships:
        rel_strs = [
            f"{r[0]} ↔ {r[1]}"
            for r in relationships[:5]
        ]
        context_parts.append(
            f"[Relationships: {'; '.join(rel_strs)}]"
        )

    context_parts.append("")  # blank line

    # Add chunks
    for item in chunk_pool:
        context_parts.append(item["text"])
        context_parts.append("")

    context = "\n".join(context_parts).strip()

    return {
        "context":          context,
        "seed_nodes":       seed_nodes,
        "neighbour_nodes":  neighbour_nodes,
        "relationships":    relationships,
        "sources":          list(sources),
        "chunks_used":      len(chunk_pool),
        "chunk_pool":       chunk_pool     # for retrieving the chunks used to generate the question (for testing/debugging)
    }


# ============================================================
# TEST — python graphrag/graphrag_retriever.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from graphrag.graphrag_indexer import load_graphrag_index
    from utils.embedder import Embedder

    BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INDEX_DIR  = os.path.join(BASE_DIR, "indexes")

    print("=" * 60)
    print("GRAPHRAG RETRIEVER TEST")
    print("=" * 60)

    # Load the index we just built
    print("Loading GraphRAG index...")
    index, graph, nodes = load_graphrag_index(
        INDEX_DIR, "graphrag_test")

    print("Loading embedding model...")
    embedder = Embedder()

    # Test queries
    test_queries = [
        "What is parameter sharing in CNNs?",
        "How does sparse connectivity reduce parameters?",
        "Explain how pooling achieves translation invariance",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)

        result = retrieve_graph_context(
            query, index, graph, nodes, embedder,
            top_k_nodes=3,
            neighbour_depth=1,
            max_context_chunks=5
        )

        print(f"Seed nodes     : "
              f"{[s['entity'] for s in result['seed_nodes']]}")
        print(f"Neighbours     : "
              f"{[n['entity'] for n in result['neighbour_nodes'][:5]]}")
        print(f"Relationships  : "
              f"{[(r[0], r[1]) for r in result['relationships'][:3]]}")
        print(f"Chunks used    : {result['chunks_used']}")
        print(f"Sources        : {result['sources']}")
        print(f"\nContext preview (first 400 chars):")
        print("-" * 40)
        print(result["context"][:400])
        print("-" * 40)

    print("\nGraphRAG retriever test complete.")

