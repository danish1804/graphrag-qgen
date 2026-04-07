# ============================================================
# src/graphrag_indexer.py
# Builds a knowledge graph from document chunks using spaCy
# for entity extraction and NetworkX for graph construction.
#
# What it does differently from embedder.py:
#   embedder.py  - stores flat text chunks in FAISS
#   this file    - extracts entities + relationships,
#                  builds a graph, then embeds graph nodes
# Architecture:
#   Two-pass frequency-based pruning +
#   heuristic entity resolution +
#   canonical multi-document knowledge graph
#
# Pass 1: Extract entities from all chunks, build
#         global corpus statistics (document frequency)
# Pass 2: Filter noise via DF threshold, normalise
#         entity names, build weighted graph
#
# Graph structure:
#   Nodes = named entities (CNN, Pooling, Hash Table...)
#   Edges = co-occurrence relationships between entities
#           within the same chunk
# Works across any number of documents without
# hardcoded noise patterns.
# ============================================================
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
from src.embedder import Embedder, DIMENSION

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
GRAPH_NAME = "graphrag"

# ── Tuning parameters ─────────────────────────────────────
# Entity appears in more than this fraction of chunks → noise
MAX_DOC_FREQ     = 0.40

# Entity must appear in at least this many chunks → signal
MIN_ENTITY_FREQ  = 2

# Two entities must co-occur in at least this many chunks
MIN_EDGE_WEIGHT  = 2

# Entity text length limits
MIN_ENTITY_LEN   = 3
MAX_ENTITY_LEN   = 60


# Step 1: Entity normalisation
def normalize_entity(text: str) -> str:
    """
    Normalise entity text for heuristic entity resolution.

    Handles morphological variants:
      "Neural Networks" -> "neural network"
      "Hash Tables"     -> "hash table"
      "CNNs"            -> "cnn"

    Does NOT handle semantic synonyms:
      "CNN" ≠ "convolutional neural network"
    This is a known limitation - see report section 4.
    """
    text = text.lower().strip()

    # Remove possessives
    text = re.sub(r"'s$", "", text)

    # Remove leading articles
    text = re.sub(r"^(the|a|an)\s+", "", text)

    # Strip trailing punctuation
    text = text.strip(".,;:!?\"'()")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Simple singularisation - handles most IT terms
    if text.endswith("ies"):
        text = text[:-3] + "y"       # "trees" -> nope, "communities" -> "community"
    elif text.endswith("ses"):
        text = text[:-2]             # "processes" -> "process"
    elif (text.endswith("s")
          and not text.endswith("ss")
          and len(text) > 4):
        text = text[:-1]             # "networks" -> "network"

    return text.strip()

def is_valid_entity(text: str) -> bool:
    """
    Rule-based validity check applied BEFORE frequency filtering.
    Removes obvious non-entities regardless of frequency.
    """
    # Length check
    if not (MIN_ENTITY_LEN <= len(text) <= MAX_ENTITY_LEN):
        return False

    # Reject if contains newlines (slide outline artifacts)
    if "\n" in text or "\r" in text:
        return False

    # Reject pure numbers
    if re.match(r"^\d+(\.\d+)?$", text):
        return False

    # Reject if more than 40% digits (page refs, equation labels)
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.4:
        return False

    # Reject typical academic PDF artifacts
    artifact_patterns = [
        r"^\d+\s*/\s*\d+$",          # "4 / 42" page numbers
        r"^lecture\s*\d*$",           # "lecture 4"
        r"^chapter\s*\d*$",           # "chapter 3"
        r"^figure\s*\d*$",            # "figure 1"
        r"^table\s*\d*$",             # "table 2"
        r"^section\s*\d*$",           # "section 2"
        r"^part\s*\d*$",              # "part 1"
        r"^page\s*\d*$",              # "page 5"
        r"^cosc\d+",                  # "cosc2779"
        r"^[a-z]\d+[a-z]?\d*$",      # "w11", "h2", "p3"
        r"^\d+[a-z]+\d*$",            # "2d", "3x3"
    ]
    for pattern in artifact_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    return True



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


def embed_graph_nodes(
        graph: nx.Graph,
        embedder: Embedder
) -> tuple:
    """
    Embed all graph nodes (entity names + their context chunks).

    Each node is embedded as:
    "entity_name: chunk1_text chunk2_text..."
    This gives richer embeddings than just the entity name alone.

    Returns:
        node_list   — ordered list of node names
        embeddings  — numpy array (N, 1024)
    """
    print("Embedding graph nodes...")
    node_list = list(graph.nodes())
    node_texts = []

    for node in node_list:
        # Combine entity name with its context chunks
        node_data = graph.nodes[node]
        context_chunks = node_data.get("chunks", [])

        # Build rich text representation
        context = " ".join(context_chunks[:2])[:500]
        node_text = f"{node}: {context}"
        node_texts.append(node_text)

    embeddings = embedder.model.encode(
        node_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    return node_list, np.array(embeddings, dtype=np.float32)


def build_graphrag_index(
        chunks: list,
        index_dir: str = INDEX_DIR,
        graph_name: str = GRAPH_NAME
) -> dict:
    """
    Full GraphRAG indexing pipeline:
    1. Load spaCy model
    2. Extract entities and build knowledge graph
    3. Embed graph nodes
    4. Build FAISS index over node embeddings
    5. Save graph + index + metadata to disk

    Returns summary dict.
    """
    os.makedirs(index_dir, exist_ok=True)

    print("=" * 60)
    print("GRAPHRAG INDEXER")
    print("=" * 60)
    print(f"Input chunks : {len(chunks)}")
    print(f"Output dir   : {index_dir}")
    print("=" * 60)

    # ── Step 1: Load spaCy ────────────────────────────────
    print("\nLoading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    # Increase max length for large chunks
    nlp.max_length = 2_000_000
    print("spaCy loaded.")

    # ── Step 2: Build knowledge graph ─────────────────────
    graph, entity_chunks = build_knowledge_graph(chunks, nlp)

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "No entities extracted. Check your chunks have enough text."
        )

    # ── Step 3: Embed graph nodes ─────────────────────────
    print("\nLoading embedding model...")
    embedder = Embedder()
    node_list, embeddings = embed_graph_nodes(graph, embedder)
    print(f"Node embeddings shape: {embeddings.shape}")

    # ── Step 4: Build FAISS index ─────────────────────────
    print("\nBuilding FAISS index over graph nodes...")

    if len(node_list) > 1000:
        n_clusters = min(int(np.sqrt(len(node_list))), 128)
        print(f"Using IndexIVFFlat (n_clusters={n_clusters})")
        quantizer = faiss.IndexFlatIP(DIMENSION)
        index = faiss.IndexIVFFlat(
            quantizer, DIMENSION, n_clusters,
            faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(n_clusters, 16)
    else:
        print("Using IndexFlatIP (exact search)")
        index = faiss.IndexFlatIP(DIMENSION)
        index.add(embeddings)

    print(f"FAISS index built — {index.ntotal} node vectors")

    
    # ── Step 5: Save everything ───────────────────────────
    # Save FAISS index
    index_path = os.path.join(index_dir, f"{graph_name}.faiss")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved : {index_path}")

    # Save NetworkX graph
    graph_path = os.path.join(index_dir, f"{graph_name}_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Graph saved       : {graph_path}")

    # Save node list and metadata
    meta_path = os.path.join(index_dir, f"{graph_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "nodes":        node_list,
            "total_nodes":  len(node_list),
            "total_edges":  graph.number_of_edges(),
            "total_chunks": len(chunks),
        }, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved    : {meta_path}")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRAPHRAG INDEX COMPLETE")
    print("=" * 60)
    print(f"Graph nodes  : {graph.number_of_nodes()}")
    print(f"Graph edges  : {graph.number_of_edges()}")
    print(f"FAISS vectors: {index.ntotal}")

    return {
        "graph_nodes":  graph.number_of_nodes(),
        "graph_edges":  graph.number_of_edges(),
        "index_path":   index_path,
        "graph_path":   graph_path,
        "meta_path":    meta_path
    }



def load_graphrag_index(
        index_dir: str = INDEX_DIR,
        graph_name: str = GRAPH_NAME
) -> tuple:
    """
    Load a saved GraphRAG index from disk.

    Returns:
        index   — FAISS index
        graph   — NetworkX graph
        nodes   — list of node names
    """
    index_path = os.path.join(index_dir, f"{graph_name}.faiss")
    graph_path = os.path.join(index_dir, f"{graph_name}_graph.pkl")
    meta_path  = os.path.join(index_dir, f"{graph_name}_meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"GraphRAG index not found at {index_path}. "
            f"Run build_graphrag_index() first."
        )

    index = faiss.read_index(index_path)

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if hasattr(index, "nprobe"):
        index.nprobe = 16

    print(f"GraphRAG index loaded — "
          f"{meta['total_nodes']} nodes, "
          f"{meta['total_edges']} edges")

    return index, graph, meta["nodes"]


# ============================================================
# TEST — run directly on source_1.pdf
# python src/graphrag_indexer.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.document_loader import load_and_chunk

    print("=" * 60)
    print("GRAPHRAG INDEXER TEST")
    print("=" * 60)

    # Test on source_1.pdf first — small and fast
    PDF_PATH = os.path.join(BASE_DIR, "data", "tested", "source_1.pdf")

    print(f"\nLoading: {PDF_PATH}")
    result = load_and_chunk(PDF_PATH)
    chunks = result["chunks"]
    print(f"Chunks loaded: {len(chunks)}")

    # Build graph index
    summary = build_graphrag_index(
        chunks,
        index_dir=INDEX_DIR,
        graph_name="graphrag_test"
    )

    print("\nLoading index back to verify...")
    index, graph, nodes = load_graphrag_index(
        INDEX_DIR, "graphrag_test")

    print(f"\nTop 20 entities by degree (most connected):")
    print("-" * 40)
    degrees = sorted(
        graph.degree(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    for entity, degree in degrees:
        freq = graph.nodes[entity].get("frequency", 0)
        print(f"  {entity:<35} degree={degree}, freq={freq}")

    print("\nSample edges (relationships found):")
    print("-" * 40)
    edges = list(graph.edges(data=True))[:15]
    for e1, e2, data in edges:
        print(f"  '{e1}' ↔ '{e2}'  "
              f"(co-occurrences: {data['weight']})")

    print("\nGraphRAG indexer test complete.")

