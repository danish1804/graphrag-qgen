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
from utils.config import cfg

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.embedder import Embedder, DIMENSION

# Paths 
BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
GRAPH_NAME = "graphrag"

# Tuning parameters
# Entity appears in more than this fraction of chunks → noise
MAX_DOC_FREQ    = cfg["graph"]["max_doc_freq"]

# Entity must appear in at least this many chunks → signal
MIN_ENTITY_FREQ = cfg["graph"]["min_entity_freq"]

# Two entities must co-occur in at least this many chunks
MIN_EDGE_WEIGHT = cfg["graph"]["min_edge_weight"]

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

    # Reject code fragments — contain = or ( or ) characters
    # These are Keras/Python argument artifacts
    if any(c in text for c in ['=', '(', ')', '{', '}', '[', ']']):
        return False

    # Reject strings starting with numbers followed by letters
    # Catches "22d", "1motivation", "3basic"
    if re.match(r'^\d+[a-z]', text, re.IGNORECASE):
        return False

    # Reject very generic single words that aren't IT concepts
    generic_words = {
        "zero", "one", "two", "data", "input", "output",
        "layer", "lets", "example", "result", "value",
        "weight", "image", "network", "model", "function",
        "method", "approach", "system", "process", "step",
        "type", "size", "number", "set", "list", "case",
        "way", "use", "note", "left", "right", "true", "false",
        "none", "same", "different", "new", "good", "large",
        "small", "high", "low", "next", "first", "last",
        "addison-wesley", "wiley", "springer", "kluwer",
        "pearson", "mcgraw", "elsevier", "cambridge",
        "morgan kaufmann", "prentice hall", "mit press",
        "academic publisher", "academic press",
    }
    if text in generic_words:
        return False

    # Reject if contains " - " (slide artifact like "locally - sparse")
    # We keep the individual parts but not the joined form
    # Actually keep "locally - sparse connectivity" as it's meaningful
    # Just reject slide outline patterns: "1motivation", "2d convolution"
    if re.match(r'^\d+[a-z\s]', text):
        return False

    # Reject diagram node labels — repeated single letters with numbers
    # Catches: "p1 p2 p3 p4", "h2 h3 h4 h5", "w4,4w 4,5w5,4"
    if re.search(r'\b[a-z]\d+\b.*\b[a-z]\d+\b', text):
        return False

    # Reject if contains commas with numbers (matrix notation)
    # Catches: "w4,4w", "1,1", "2,2"
    if re.search(r'\d,\d', text):
        return False

    # Reject if too many words (likely merged entities or sentence fragments)
    # More than 5 words = not a concept, it's a phrase
    word_count = len(text.split())
    if word_count > 5:
        return False

    # Fix broken singularisation — "hypothesi" should be rejected
    # (it's not a real word — our singulariser over-fired)
    if text.endswith("si") or text.endswith("xi"):
        return False
    
    # Reject bullet point prefixes
    if text.startswith("•") or text.startswith("–"):
        return False

    # Reject concatenated words (no space, mixed case mid-word)
    if re.search(r'[a-z][A-Z]', text):
        return False

    # Reject section heading patterns
    if re.match(r'^\d+\.\d+\s+[a-z]', text):
        return False
    
    # Reject words with spaces inserted mid-token
    # Catches "becaus e", "your s ystem", "sys- tem"
    # Pattern: short word fragment + space + short fragment
    if re.search(r'\b\w{1,4}\s+\w{1,3}\b', text) and len(text.split()) == 2:
        if all(len(w) <= 4 for w in text.split()):
            return False
    
    return True
    



# Step 2: Entity extraction
def extract_raw_entities(chunk: str, nlp) -> list:
    """
    Extract candidate entities from one chunk using spaCy.
    No filtering at this stage — just extraction.
    Returns normalised entity strings.
    """
    # Clean slide/PDF artifacts before NLP
    cleaned = re.sub(r'\d+\s*/\s*\d+', ' ', chunk)      # "4 / 42"
    cleaned = re.sub(r'Lecture\s+\d+[^\n]*\n', ' ', cleaned)
    cleaned = re.sub(r'COSC\d+/\d+', ' ', cleaned, flags=re.IGNORECASE)
    
    # NEW — Remove Keras/Python code blocks
    # These appear as inline code in the PDF slides
    cleaned = re.sub(r'tf\.keras\.\w+[^)]*\)', ' ', cleaned)
    cleaned = re.sub(r'\w+=\w+', ' ', cleaned)           # key=value pairs
    cleaned = re.sub(r'\w+=\([\d,\s]+\)', ' ', cleaned)  # key=(1,1)
    
    # NEW — Remove slide outline numbering artifacts
    # "1Motivation 2D Convolution 3Basic..."
    cleaned = re.sub(r'\d+[A-Z][a-z]', ' ', cleaned)
    
    # NEW — Remove diagram labels like "p1 p2 p3 p4 p5"
    cleaned = re.sub(r'\b[a-z]\d+\s+[a-z]\d+\s+[a-z]\d+\b', ' ', cleaned)
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if len(cleaned) < 20:
        return []

    doc = nlp(cleaned)
    candidates = set()

    # Named entities from spaCy NER
    for ent in doc.ents:
        norm = normalize_entity(ent.text)
        if is_valid_entity(norm):
            candidates.add(norm)

    # Noun chunks — critical for IT technical terms
    for np_chunk in doc.noun_chunks:
        norm = normalize_entity(np_chunk.text)
        if (is_valid_entity(norm)
                and not np_chunk.root.is_stop):
            candidates.add(norm)

    return list(candidates)



# Step 3: Two-pass graph builder
def build_knowledge_graph(
        all_chunks: list,
        chunk_sources: list,
        nlp,
        max_doc_freq: float  = MAX_DOC_FREQ,
        min_entity_freq: int = MIN_ENTITY_FREQ,
        min_edge_weight: int = MIN_EDGE_WEIGHT
) -> tuple:
    """
    Two-pass knowledge graph construction.

    Pass 1 — Global corpus statistics:
        Extract entities from every chunk
        Count document frequency (DF) per entity
        Count total occurrences per entity

    Pass 2 — Filtered graph construction:
        Remove entities above DF threshold (noise)
        Remove entities below min frequency (rare)
        Build edges from co-occurrence within chunks
        Weight edges by cross-source co-occurrence
        Track which source documents contribute each edge

    Args:
        all_chunks   : list of text chunks
        chunk_sources: list of source filenames per chunk
        nlp          : spaCy language model
        ...

    Returns:
        graph        : NetworkX weighted graph
        entity_chunks: dict mapping entity → example chunks
    """
    total_chunks = len(all_chunks)
    print(f"\nPASS 1 — Global corpus statistics")
    print(f"Total chunks to process: {total_chunks}")

    # Pass 1 - extract entities and build corpus stats
    entity_chunk_count  = defaultdict(int)    # DF per entity
    entity_total_count  = defaultdict(int)    # total occurrences
    entity_sources      = defaultdict(set)    # which docs
    chunk_entity_lists  = []                  # entities per chunk
    entity_example_chunks = defaultdict(list) # for context

    for i, (chunk, source) in enumerate(
            tqdm(zip(all_chunks, chunk_sources),
                 total=total_chunks,
                 desc="Pass 1: extracting entities")):

        entities = extract_raw_entities(chunk, nlp)
        chunk_entity_lists.append((entities, source))

        seen_in_chunk = set()
        for ent in entities:
            entity_total_count[ent] += 1
            entity_sources[ent].add(source)

            if ent not in seen_in_chunk:
                entity_chunk_count[ent] += 1
                seen_in_chunk.add(ent)

            if len(entity_example_chunks[ent]) < 3:
                entity_example_chunks[ent].append(chunk)

    print(f"Unique entities found    : {len(entity_chunk_count)}")

    # Frequency filtering
    max_allowed_chunks = max_doc_freq * total_chunks

    valid_entities = {
        ent for ent, count in entity_chunk_count.items()
        if (count >= min_entity_freq          # appears enough
            and count <= max_allowed_chunks)  # not too common
    }

    removed_noise = {
        ent for ent, count in entity_chunk_count.items()
        if count > max_allowed_chunks
    }
    removed_rare = {
        ent for ent, count in entity_chunk_count.items()
        if count < min_entity_freq
    }

    print(f"Removed as noise (>{max_doc_freq*100:.0f}% DF): "
          f"{len(removed_noise)}")
    print(f"Removed as rare  (<{min_entity_freq} occurrences): "
          f"{len(removed_rare)}")
    print(f"Valid entities remaining : {len(valid_entities)}")

    if len(removed_noise) > 0:
        top_noise = sorted(
            removed_noise,
            key=lambda e: entity_chunk_count[e],
            reverse=True
        )[:5]
        print(f"Top noise entities removed: {top_noise}")

    # Pass 2: Build graph
    print(f"\nPASS 2 — Building knowledge graph")
    graph = nx.Graph()

    # Add nodes
    for ent in valid_entities:
        graph.add_node(
            ent,
            frequency    = entity_total_count[ent],
            doc_freq     = entity_chunk_count[ent],
            sources      = list(entity_sources[ent]),
            num_sources  = len(entity_sources[ent]),
            chunks       = entity_example_chunks[ent][:3]
        )

    # Build edges from co-occurrence
    edge_weights  = defaultdict(int)
    edge_sources  = defaultdict(set)

    for (entities, source) in tqdm(
            chunk_entity_lists,
            desc="Pass 2: building edges"):

        # Filter to valid only
        valid = [e for e in entities if e in valid_entities]

        # All pairs in this chunk co-occur
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pair = tuple(sorted([valid[i], valid[j]]))
                edge_weights[pair] += 1
                edge_sources[pair].add(source)

    # Add edges meeting threshold
    edges_added = 0
    for (e1, e2), weight in edge_weights.items():
        if weight >= min_edge_weight:
            graph.add_edge(
                e1, e2,
                weight       = weight,
                sources      = list(edge_sources[(e1, e2)]),
                cross_source = len(edge_sources[(e1, e2)]) > 1
            )
            edges_added += 1

    cross_source_edges = sum(
        1 for _, _, d in graph.edges(data=True)
        if d.get("cross_source", False)
    )

    print(f"Graph nodes         : {graph.number_of_nodes()}")
    print(f"Graph edges         : {graph.number_of_edges()}")
    print(f"Cross-source edges  : {cross_source_edges} "
          f"(connect concepts across documents)")

    return graph, dict(entity_example_chunks)


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


# Main build function 
def build_graphrag_index(
        chunks: list,
        sources: list = None,
        index_dir: str  = INDEX_DIR,
        graph_name: str = GRAPH_NAME
) -> dict:
    """
    Full GraphRAG indexing pipeline.

    Args:
        chunks    : list of text strings
        sources   : list of source filenames per chunk
                    (optional — defaults to "unknown")
        index_dir : where to save the index
        graph_name: name prefix for saved files

    Returns:
        summary dict with graph stats and file paths
    """
    os.makedirs(index_dir, exist_ok=True)

    # Default sources if not provided
    if sources is None:
        sources = ["unknown"] * len(chunks)

    print("=" * 60)
    print("GRAPHRAG INDEXER")
    print("=" * 60)
    print(f"Chunks          : {len(chunks)}")
    print(f"Unique sources  : {len(set(sources))}")
    print(f"Output dir      : {index_dir}")
    print("=" * 60)

    # Load spaCy
    print("\nLoading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000
    print("spaCy loaded.")

    # Build graph
    graph, entity_chunks = build_knowledge_graph(
        chunks, sources, nlp)

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "No valid entities found after filtering. "
            "Check that your chunks contain enough text."
        )

    # Embed nodes
    print("\nLoading embedding model...")
    embedder = Embedder()
    node_list, embeddings = embed_graph_nodes(graph, embedder)
    print(f"Node embeddings shape: {embeddings.shape}")

    # Build FAISS index
    print("\nBuilding FAISS index over graph nodes...")
    n_nodes = len(node_list)

    if n_nodes > 1000:
        n_clusters = min(int(np.sqrt(n_nodes)), 128)
        print(f"Using IndexIVFFlat (n_clusters={n_clusters})")
        quantizer = faiss.IndexFlatIP(DIMENSION)
        index     = faiss.IndexIVFFlat(
            quantizer, DIMENSION, n_clusters,
            faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(n_clusters, 16)
    else:
        print(f"Using IndexFlatIP (exact search, {n_nodes} nodes)")
        index = faiss.IndexFlatIP(DIMENSION)
        index.add(embeddings)

    print(f"FAISS index built — {index.ntotal} node vectors")

    # Save everything
    index_path = os.path.join(index_dir, f"{graph_name}.faiss")
    graph_path = os.path.join(index_dir, f"{graph_name}_graph.pkl")
    meta_path  = os.path.join(index_dir, f"{graph_name}_meta.json")

    faiss.write_index(index, index_path)
    print(f"FAISS index saved : {index_path}")

    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Graph saved       : {graph_path}")

    cross_source = sum(
        1 for _, _, d in graph.edges(data=True)
        if d.get("cross_source", False)
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "nodes":               node_list,
            "total_nodes":         len(node_list),
            "total_edges":         graph.number_of_edges(),
            "cross_source_edges":  cross_source,
            "total_chunks":        len(chunks),
            "unique_sources":      list(set(sources)),
        }, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved    : {meta_path}")

    print("\n" + "=" * 60)
    print("GRAPHRAG INDEX COMPLETE")
    print("=" * 60)
    print(f"Graph nodes         : {graph.number_of_nodes()}")
    print(f"Graph edges         : {graph.number_of_edges()}")
    print(f"Cross-source edges  : {cross_source}")
    print(f"FAISS vectors       : {index.ntotal}")

    return {
        "graph_nodes":         graph.number_of_nodes(),
        "graph_edges":         graph.number_of_edges(),
        "cross_source_edges":  cross_source,
        "index_path":          index_path,
        "graph_path":          graph_path,
        "meta_path":           meta_path
    }



# Load Function
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
          f"{meta['total_edges']} edges, "
          f"{meta['cross_source_edges']} cross-source edges")

    return index, graph, meta["nodes"]


# ============================================================
# TEST — python graphrag/graphrag_indexer.py
# ============================================================
if __name__ == "__main__":
    from utils.document_loader import load_and_chunk

    print("=" * 60)
    print("GRAPHRAG INDEXER TEST — source_1.pdf")
    print("=" * 60)

    PDF = os.path.join(BASE_DIR, "data", "tested", "source_1.pdf")
    result = load_and_chunk(PDF)
    chunks  = result["chunks"]
    sources = ["source_1.pdf"] * len(chunks)

    print(f"Chunks loaded: {len(chunks)}")

    summary = build_graphrag_index(
        chunks, sources,
        index_dir  = INDEX_DIR,
        graph_name = "graphrag_test"
    )

    # Load back and inspect
    print("\nLoading index back to verify...")
    index, graph, nodes = load_graphrag_index(
        INDEX_DIR, "graphrag_test")

    print(f"\nTop 20 entities by degree (most connected):")
    print("-" * 50)
    degrees = sorted(
        graph.degree(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    for entity, degree in degrees:
        freq    = graph.nodes[entity].get("frequency", 0)
        n_src   = graph.nodes[entity].get("num_sources", 0)
        print(f"  {entity:<40} "
              f"degree={degree:<4} "
              f"freq={freq:<4} "
              f"sources={n_src}")

    print(f"\nSample meaningful edges:")
    print("-" * 50)
    # Show edges with highest weight
    top_edges = sorted(
        graph.edges(data=True),
        key=lambda x: x[2]["weight"],
        reverse=True
    )[:15]
    for e1, e2, data in top_edges:
        cross = "*" if data.get("cross_source") else " "
        print(f"  {cross} '{e1}' ↔ '{e2}'  "
              f"(weight={data['weight']}, "
              f"sources={data['sources']})")

    print(f"\n* = cross-source edge (connects concepts across documents)")
    print("\nGraphRAG indexer test complete.")
