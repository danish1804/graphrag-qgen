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

from utils.document_loader  import load_and_chunk
from utils.embedder         import Embedder, DIMENSION
from graphrag.graphrag_indexer import (
    normalize_entity,
    is_valid_entity,
    extract_raw_entities,
    build_knowledge_graph,
    embed_graph_nodes
)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def build_combined_graphrag_index(
        docs_dir:   str = TESTED_DIR,
        index_dir:  str = INDEX_DIR,
        graph_name: str = GRAPH_NAME
) -> dict:
    """
    Full multi-document GraphRAG indexing pipeline.

    Steps:
    1. Discover and load all documents
    2. Build combined knowledge graph (two-pass)
    3. Embed graph nodes
    4. Build FAISS index
    5. Save everything to disk

    Returns summary dict.
    """
    os.makedirs(index_dir, exist_ok=True)

    print("=" * 60)
    print("MULTI-DOCUMENT GRAPHRAG INDEXER")
    print("=" * 60)
    print(f"Source folder : {docs_dir}")
    print(f"Output dir    : {index_dir}")
    print(f"Graph name    : {graph_name}")
    print("=" * 60)

    # ── Step 1: Load all documents ─────────────────────
    all_chunks, all_sources, doc_stats = load_all_chunks(
        docs_dir)

    if not all_chunks:
        raise ValueError(
            f"No chunks loaded from {docs_dir}. "
            f"Run test_document_batch.py first."
        )

    unique_docs = list(set(all_sources))
    print(f"\nUnique documents  : {len(unique_docs)}")
    print(f"Total chunks      : {len(all_chunks)}")
    print(f"\nPer-document stats:")
    for fname, stats in sorted(doc_stats.items()):
        print(f"  {fname:<30} "
              f"{stats['chunks']:>5} chunks  "
              f"{stats['chars']:>10,} chars")


# ── Step 2: Build combined knowledge graph ─────────
    print("\nLoading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10_000_000
    print("spaCy loaded.")

    graph, entity_chunks = build_knowledge_graph(
        all_chunks,
        all_sources,
        nlp,
        max_doc_freq    = MAX_DOC_FREQ,
        min_entity_freq = MIN_ENTITY_FREQ,
        min_edge_weight = MIN_EDGE_WEIGHT
    )

    if graph.number_of_nodes() == 0:
        raise ValueError("No valid entities found.")

    # ── Cross-document analysis ────────────────────────
    cross_source_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get("num_sources", 0) > 1
    ]
    cross_source_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get("cross_source", False)
    ]

    print(f"\nCross-document nodes  : {len(cross_source_nodes)} "
          f"(concepts appearing in multiple docs)")
    print(f"Cross-document edges  : {len(cross_source_edges)} "
          f"(concepts co-occurring across docs)")

    if cross_source_nodes:
        print(f"\nTop 10 cross-document concepts:")
        top_cross = sorted(
            cross_source_nodes,
            key=lambda n: graph.nodes[n].get("num_sources", 0),
            reverse=True
        )[:10]
        for node in top_cross:
            d = graph.nodes[node]
            print(f"  '{node}' — "
                  f"appears in {d.get('num_sources', 0)} documents: "
                  f"{d.get('sources', [])}")


# ── Step 3: Embed graph nodes ──────────────────────
    print("\nLoading embedding model...")
    embedder = Embedder()
    node_list, embeddings = embed_graph_nodes(graph, embedder)
    print(f"Node embeddings shape: {embeddings.shape}")

    # ── Step 4: Build FAISS index ──────────────────────
    print("\nBuilding FAISS index...")
    n_nodes = len(node_list)

    if n_nodes > 1000:
        n_clusters = min(int(np.sqrt(n_nodes)), 256)
        print(f"Using IndexIVFFlat "
              f"(n_clusters={n_clusters}, {n_nodes} nodes)")
        quantizer = faiss.IndexFlatIP(DIMENSION)
        index     = faiss.IndexIVFFlat(
            quantizer, DIMENSION, n_clusters,
            faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(n_clusters, 32)
    else:
        print(f"Using IndexFlatIP ({n_nodes} nodes)")
        index = faiss.IndexFlatIP(DIMENSION)
        index.add(embeddings)

    print(f"FAISS index built — {index.ntotal} vectors")

# ── Step 5: Save everything ────────────────────────
    index_path = os.path.join(
        index_dir, f"{graph_name}.faiss")
    graph_path = os.path.join(
        index_dir, f"{graph_name}_graph.pkl")
    meta_path  = os.path.join(
        index_dir, f"{graph_name}_meta.json")

    faiss.write_index(index, index_path)
    print(f"FAISS index saved : {index_path}")

    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Graph saved       : {graph_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "nodes":                node_list,
            "total_nodes":          len(node_list),
            "total_edges":          graph.number_of_edges(),
            "cross_source_nodes":   len(cross_source_nodes),
            "cross_source_edges":   len(cross_source_edges),
            "total_chunks":         len(all_chunks),
            "total_documents":      len(unique_docs),
            "documents":            sorted(unique_docs),
            "doc_stats":            doc_stats
        }, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved    : {meta_path}")

    # ── Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MULTI-DOCUMENT GRAPHRAG INDEX COMPLETE")
    print("=" * 60)
    print(f"Documents indexed     : {len(unique_docs)}")
    print(f"Total chunks          : {len(all_chunks)}")
    print(f"Graph nodes           : {graph.number_of_nodes()}")
    print(f"Graph edges           : {graph.number_of_edges()}")
    print(f"Cross-document nodes  : {len(cross_source_nodes)}")
    print(f"Cross-document edges  : {len(cross_source_edges)}")
    print(f"FAISS vectors         : {index.ntotal}")

    return {
        "total_documents":    len(unique_docs),
        "total_chunks":       len(all_chunks),
        "graph_nodes":        graph.number_of_nodes(),
        "graph_edges":        graph.number_of_edges(),
        "cross_source_nodes": len(cross_source_nodes),
        "cross_source_edges": len(cross_source_edges),
        "index_path":         index_path,
        "graph_path":         graph_path,
        "meta_path":          meta_path
    }

def load_combined_graphrag_index(
        index_dir:  str = INDEX_DIR,
        graph_name: str = GRAPH_NAME
) -> tuple:
    """Load the combined GraphRAG index from disk."""
    index_path = os.path.join(
        index_dir, f"{graph_name}.faiss")
    graph_path = os.path.join(
        index_dir, f"{graph_name}_graph.pkl")
    meta_path  = os.path.join(
        index_dir, f"{graph_name}_meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Combined GraphRAG index not found: {index_path}\n"
            f"Run build_combined_graphrag_index() first."
        )

    index = faiss.read_index(index_path)

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if hasattr(index, "nprobe"):
        index.nprobe = 32

    print(f"Combined GraphRAG index loaded — "
          f"{meta['total_nodes']} nodes, "
          f"{meta['total_edges']} edges, "
          f"{meta['cross_source_nodes']} cross-doc nodes, "
          f"{meta['total_documents']} documents")

    return index, graph, meta["nodes"]

# ============================================================
# TEST — python graphrag/graphrag_multi_indexer.py
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("MULTI-DOCUMENT GRAPHRAG INDEXER TEST")
    print("=" * 60)

    # Build the combined index
    summary = build_combined_graphrag_index()

    # Load back and test
    print("\nLoading index back to verify...")
    from utils.embedder import Embedder
    from graphrag.graphrag_retriever import retrieve_graph_context

    index, graph, nodes = load_combined_graphrag_index()
    embedder = Embedder()

    # Show top entities by degree
    print(f"\nTop 20 entities by degree:")
    print("-" * 55)
    degrees = sorted(
        graph.degree(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    for entity, degree in degrees:
        d    = graph.nodes[entity]
        nsrc = d.get("num_sources", 0)
        freq = d.get("frequency", 0)
        cross = "*" if nsrc > 1 else " "
        print(f"  {cross} {entity:<38} "
              f"degree={degree:<4} "
              f"freq={freq:<5} "
              f"docs={nsrc}")

    print(f"\n* = appears in multiple documents")

    # Test retrieval on previously weak query
    print(f"\nTesting previously weak query: pooling")
    print("-" * 55)
    result = retrieve_graph_context(
        "What is the purpose of pooling in CNNs?",
        index, graph, nodes, embedder,
        top_k_nodes=3,
        neighbour_depth=1
    )
    print(f"Seed nodes   : "
          f"{[s['entity'] for s in result['seed_nodes']]}")
    print(f"Neighbours   : "
          f"{[n['entity'] for n in result['neighbour_nodes'][:5]]}")
    print(f"Context preview:")
    print(result["context"][:300])

    print("\nMulti-document GraphRAG indexer test complete.")
