# ============================================================
# src/graphrag_domain_indexer.py
# Per-domain GraphRAG indexer
#
# Builds separate knowledge graphs per subject domain.
# Routes queries to the correct domain graph at generation time.
#
# Why per-domain graphs:
#   "pooling" in Deep Learning ≠ "pooling" in Databases
#   "transaction" in OS ≠ "transaction" in Databases
#   Mixed corpus corrupts entity relationships
#   Domain-specific graphs produce accurate retrieval
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from graphrag.graphrag_indexer import build_graphrag_index
from graphrag.graphrag_indexer import load_graphrag_index
from utils.document_loader  import load_and_chunk
import numpy as np

# Paths 
BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")

# Domain definitions
DOMAIN_MAP = {
    "deep_learning": {
        "files": [
            "source_50.pdf",
            "source_51.pdf",
            "source_52.pdf",
            "source_53.pdf",
            "source_54.pdf",
            "source_55.pdf",
            "source_56.pdf",
            "source_57.pdf",
            "source_58.pdf",
            "source_59.pdf",
            "source_60.pdf",
            "source_61.pdf",
            "source_62.pdf",

        ],
        "description": "Deep Learning - CNNs, pooling, backprop",
        "keywords": [
            "cnn", "convolutional", "neural network", "pooling",
            "backpropagation", "gradient", "deep learning",
            "convolution", "relu", "activation", "batch",
            "epoch", "training", "feature map", "kernel",
            "stride", "padding", "weight", "bias", "layer",
            "softmax", "dropout", "overfitting", "regularisation", "weight sharing"
        ]
    },
    "database": {
        "files": [
            "source_7.pdf",
            "source_8.pdf",
            "source_13.pdf",
            "source_14.pdf",
            "source_17.pdf",
            "source_18.pdf",
            "source_31.pdf",
            "source_36.pdf",
            "source_49.pdf",
        ],
        "description": "Database Systems - SQL, transactions, indexing, design",
        "keywords": [
            "sql", "database", "query", "transaction", "relation",
            "table", "index", "schema", "normalisation", "join",
            "primary key", "foreign key", "acid", "concurrency",
            "deadlock", "hash table", "b-tree", "indexing",
            "tuple", "attribute", "entity", "relational",
            "dbms", "er diagram", "integrity", "constraint",
            "functional dependency", "decomposition", "algebra"
        ]
    },
    "software_engineering": {
        "files": [
            "source_1.pdf",
            "source_2pdf",
            "source_3.pdf",
            "source_4.pdf",
            "source_11.pdf",
            "source_15.pdf",
            "source_16.pdf",
            "source_21.pdf",
            "source_22.pdf",
            "source_37.pdf",
            "source_45.pdf",
            "source_46.pdf",
            "source_47.pdf",
            "source_48.pdf",

        ],
        "description": "Software Engineering - processes, patterns, testing",
        "keywords": [
            "software", "agile", "design pattern", "testing",
            "requirement", "architecture", "scrum", "sprint",
            "refactoring", "oop", "inheritance", "polymorphism",
            "abstraction", "encapsulation", "uml", "use case",
            "waterfall", "prototype", "class diagram", "module",
            "interface", "component", "deployment", "maintenance",
            "build", "version control", "safety", "process",
            "object oriented", "object-oriented", "class",
            "gang of four", "design patterns", "factory",
            "singleton", "observer", "mvc", "solid",
            "unit test", "integration test", "test driven",
            "continuous integration", "devops", "git",
            "coupling", "cohesion", "modular", "reuse",
            "software development", "software process",
            "software quality", "software safety",
        ]
    },
    "operating_systems": {
        "files": [
            "source_5.pdf",    
            "source_10.pdf",   
            "source_19.pdf",   
            "source_20.pdf",   
            "source_33.pdf",   
            "source_35.pdf",   
            "source_39.pdf",   
            "source_40.pdf",   
            "source_41.pdf", 
            "source_42.pdf",   
            "source_43.pdf", 
        ],
        "description": "Operating Systems - processes, memory, scheduling",
        "keywords": [
            "process", "thread", "scheduler", "memory",
            "virtual memory", "page", "segment", "kernel",
            "interrupt", "system call", "file system", "disk",
            "cache", "buffer", "semaphore", "mutex", "deadlock",
            "unix", "linux", "fork", "pipe", "signal", "socket",
            "i/o", "device driver", "ipc", "context switch",
            "scheduling algorithm", "round robin", "priority",
            "xv6", "trap", "page table", "tlb"
        ]
    },
    "computer_architecture": {
        "files": [
            "source_6.pdf",    
            "source_9.pdf",    
            "source_12.pdf",   
            "source_23.pdf",   
            "source_24.pdf",   
            "source_25.pdf",   
            "source_26.pdf",   
            "source_27.pdf",   
            "source_28.pdf",   
            "source_29.pdf",   
            "source_30.pdf",   
            "source_32.pdf",   
            "source_38.pdf",   
            "source_44.pdf",   
        ],
        "description": "Computer Architecture - CPU, memory hierarchy, instruction sets, pipelining",
        "keywords": [
            "cpu", "instruction set", "pipeline", "cache",
            "memory hierarchy", "register", "alu", "risc", "cisc",
            "branch prediction", "out of order", "superscalar",
            "amdahl", "addressing mode", "interrupt", "dma",
            "bus", "clock cycle", "cpi", "throughput", "hazard",
            "forwarding", "stall", "fetch", "decode", "execute",
            "writeback", "direct mapped", "associative", "tlb",
            "virtual address", "physical address", "cache hit",
            "cache miss", "write back", "write through"
        ]
    }

}

# Domain profile embeddings (built once at startup)
_domain_profiles = None  # cached after first build

def build_domain_profiles(embedder) -> dict:
    """
    Build one embedding per domain using domain description
    + representative keywords as profile text.

    Called once at server startup. Cached in _domain_profiles.
    Much richer than document titles alone - uses the full
    keyword vocabulary that defines each domain.
    """
    global _domain_profiles
    profiles = {}

    for domain_name, config in DOMAIN_MAP.items():
        # Combine description + all keywords into profile text
        keywords   = " ".join(config["keywords"])
        desc       = config["description"]
        files_text = " ".join(config["files"])

        profile_text = f"{desc}. {keywords}. {files_text}"

        embedding = embedder.model.encode(
            [profile_text],
            normalize_embeddings=True
        )
        profiles[domain_name] = embedding[0]

    _domain_profiles = profiles
    print(f"[Router] Domain profiles built for: "
          f"{list(profiles.keys())}")
    return profiles


# def route_by_embedding(
#         query:    str,
#         embedder,
#         profiles: dict = None
# ) -> str:
#     """
#     Route a learning objective to the correct domain graph
#     using embedding cosine similarity.

#     More robust than keyword matching:
#     - Works for queries that don't contain exact keywords
#     - Handles paraphrased domain terminology
#     - Scales to new domains without keyword list maintenance

#     Falls back to keyword routing if profiles not available.
#     """
#     global _domain_profiles

#     # Use cached profiles if available
#     if profiles is None:
#         profiles = _domain_profiles

#     if profiles is None:
#         # Fallback to keyword routing
#         print("[Router] Profiles not built - "
#               "falling back to keyword routing")
#         return route_query_to_domain(query)

#     # Embed the query
#     query_emb = embedder.model.encode(
#         [query],
#         normalize_embeddings=True
#     )[0]

#     # Cosine similarity (vectors are normalised so dot product = cosine)
#     scores = {
#         domain: float(np.dot(query_emb, profile_emb))
#         for domain, profile_emb in profiles.items()
#     }

#     best_domain = max(scores, key=scores.get)
#     best_score  = scores[best_domain]

#     print(f"[Router] Embedding scores: "
#           f"{ {d: round(s, 3) for d, s in scores.items()} }")
#     print(f"[Router] Routed to: {best_domain} "
#           f"(score={best_score:.3f})")

#     return best_domain

def route_by_embedding(
    query:        str,
    embedder,
    return_all_scores: bool = False   # ← NEW parameter
) -> str | dict:
    """
    Route query to correct domain using embedding similarity.

    Args:
        query:             Learning objective text
        embedder:          BAAI embedder instance
        return_all_scores: If True returns all domain scores dict
                           If False returns winning domain string

    Returns:
        If return_all_scores=False: domain name string
        If return_all_scores=True:  dict of {domain: score}
    """
    global _domain_profiles

    if _domain_profiles is None:
        build_domain_profiles(embedder)

    # Embed query
    query_embedding = embedder.embed([query])[0]
    query_embedding = query_embedding / (
        np.linalg.norm(query_embedding) + 1e-10
    )

    # Score all domains
    scores = {}
    for domain, profile_embedding in _domain_profiles.items():
        score = float(np.dot(query_embedding, profile_embedding))
        scores[domain] = round(score, 3)

    print(f"[Router] Embedding scores: {scores}")

    # Return all scores or just winner
    if return_all_scores:
        return scores

    best_domain = max(scores, key=scores.get)
    print(f"[Router] Routed to: {best_domain} "
          f"(score={scores[best_domain]})")
    return best_domain


def build_domain_graph(
        domain_name: str,
        docs_dir:   str = TESTED_DIR,
        index_dir:  str = INDEX_DIR
) -> dict:
    """
    Build a GraphRAG index for one domain.

    Loads all PDFs for the domain, builds combined graph,
    saves as graphrag_{domain_name}.

    Returns summary dict.
    """
    if domain_name not in DOMAIN_MAP:
        raise ValueError(
            f"Unknown domain: {domain_name}. "
            f"Choose from: {list(DOMAIN_MAP.keys())}"
        )

    domain = DOMAIN_MAP[domain_name]
    files  = domain["files"]

    print(f"\n{'='*60}")
    print(f"Building graph for domain: {domain_name}")
    print(f"Description: {domain['description']}")
    print(f"Files: {len(files)}")
    print(f"{'='*60}")

    all_chunks  = []
    all_sources = []
    loaded      = []
    failed      = []

    for fname in files:
        fpath = os.path.join(docs_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found - skipping")
            failed.append(fname)
            continue

        try:
            result  = load_and_chunk(fpath)
            chunks  = result["chunks"]
            all_chunks.extend(chunks)
            all_sources.extend([fname] * len(chunks))
            loaded.append(fname)
            print(f"  OK  {fname:<30} "
                  f"{len(chunks):>5} chunks")
        except Exception as e:
            print(f"  FAIL {fname}: {e}")
            failed.append(fname)

    if not all_chunks:
        raise ValueError(
            f"No chunks loaded for domain {domain_name}"
        )

    print(f"\nTotal chunks for {domain_name}: {len(all_chunks)}")

    graph_name = f"graphrag_{domain_name}"
    summary = build_graphrag_index(
        all_chunks,
        all_sources,
        index_dir  = index_dir,
        graph_name = graph_name
    )

    summary["domain"]          = domain_name
    summary["files_loaded"]    = loaded
    summary["files_failed"]    = failed
    summary["graph_name"]      = graph_name

    return summary


def build_all_domain_graphs(
        docs_dir:  str = TESTED_DIR,
        index_dir: str = INDEX_DIR
) -> dict:
    """Build GraphRAG indexes for all domains."""
    results = {}
    for domain_name in DOMAIN_MAP:
        try:
            summary = build_domain_graph(
                domain_name, docs_dir, index_dir)
            results[domain_name] = summary
            print(f"\n✓ {domain_name}: "
                  f"{summary['graph_nodes']} nodes, "
                  f"{summary['graph_edges']} edges")
        except Exception as e:
            print(f"\n✗ {domain_name} FAILED: {e}")
            results[domain_name] = {"error": str(e)}

    return results

def route_query_to_domain(learning_objective: str) -> str:
    """
    Route a learning objective to the correct domain graph.

    Uses keyword matching - scores each domain by how many
    of its keywords appear in the objective.

    Returns domain name with highest score.
    Falls back to 'database' if no match (largest domain).
    """
    obj_lower = learning_objective.lower()
    scores    = {}

    for domain_name, domain in DOMAIN_MAP.items():
        score = sum(
            1 for kw in domain["keywords"]
            if kw in obj_lower
        )
        scores[domain_name] = score

    best_domain = max(scores, key=scores.get)
    best_score  = scores[best_domain]

    # If no keywords matched fall back to database
    if best_score == 0:
        return "database"

    return best_domain


def load_domain_graph(
        domain_name: str,
        index_dir:   str = INDEX_DIR
) -> tuple:
    """Load a domain-specific GraphRAG index."""
    graph_name = f"graphrag_{domain_name}"
    return load_graphrag_index(index_dir, graph_name)


# ============================================================
# TEST - python graphrag/graphrag_domain_indexer.py
# ============================================================
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=list(DOMAIN_MAP.keys()) + ["all"],
        default="deep_learning",
        help="Which domain to build"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DOMAIN-SPECIFIC GRAPHRAG INDEXER")
    print("=" * 60)

    if args.domain == "all":
        results = build_all_domain_graphs()
        print("\n" + "=" * 60)
        print("ALL DOMAINS COMPLETE")
        print("=" * 60)
        for domain, result in results.items():
            if "error" in result:
                print(f"  Wrong {domain}: {result['error']}")
            else:
                print(f"  Correct {domain}: "
                      f"{result['graph_nodes']} nodes, "
                      f"{result['graph_edges']} edges")
    else:
        summary = build_domain_graph(args.domain)
        print(f"\nDomain graph built: {args.domain}")
        print(f"Nodes: {summary['graph_nodes']}")
        print(f"Edges: {summary['graph_edges']}")

    # Test routing
    print("\n" + "=" * 60)
    print("QUERY ROUTING TEST")
    print("=" * 60)
    test_queries = [
        "Explain how pooling works in CNNs",
        "How does a hash table handle collisions?",
        "What is the difference between Agile and Waterfall?",
        "Explain virtual memory management in operating systems",
        "What is parameter sharing in convolutional networks?",
        "Describe the ACID properties of database transactions",
    ]
    for q in test_queries:
        domain = route_query_to_domain(q)
        print(f"  '{q[:55]}...'")
        print(f"    -> routed to: {domain}")
        print()