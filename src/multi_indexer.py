# ============================================================
# src/multi_indexer.py
# Builds a combined FAISS index from multiple documents
# Each chunk is tagged with its source document
# ============================================================

import os
import json
import numpy as np
import faiss
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.document_loader import load_and_chunk
from src.embedder import Embedder, DIMENSION

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
INDEX_NAME = "combined"


def build_combined_index(
        docs_dir: str   = TESTED_DIR,
        index_dir: str  = INDEX_DIR,
        index_name: str = INDEX_NAME
) -> dict:
    """
    Load all PDFs and PPTXs from docs_dir,
    chunk them, embed them, and build one
    combined FAISS index with source metadata.

    Each chunk stored with:
    - text content
    - source file name
    - chunk position within that file
    """
    os.makedirs(index_dir, exist_ok=True)

    # ── Step 1: Discover all documents ───────────────────
    supported = (".pdf", ".pptx")
    doc_files = sorted([
        f for f in os.listdir(docs_dir)
        if f.lower().endswith(supported)
    ])

    if not doc_files:
        raise FileNotFoundError(
            f"No PDF or PPTX files found in: {docs_dir}")

    print("=" * 60)
    print("MULTI-DOCUMENT INDEXER")
    print("=" * 60)
    print(f"Documents found : {len(doc_files)}")
    print(f"Source folder   : {docs_dir}")
    print(f"Index output    : {index_dir}/{index_name}")
    print("=" * 60)

    # ── Step 2: Load and chunk all documents ─────────────
    all_chunks = []   # raw text
    all_meta   = []   # metadata per chunk

    failed = []

    for fname in doc_files:
        fpath = os.path.join(docs_dir, fname)
        try:
            result = load_and_chunk(fpath)
            chunks = result["chunks"]

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({
                    "source":      fname,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })

            print(f"  OK  {fname:<40} {len(chunks)} chunks")

        except Exception as e:
            print(f"  FAIL {fname:<40} {str(e)[:50]}")
            failed.append(fname)

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("No chunks loaded — cannot build index.")
    
    
    # ── Step 3: Embed all chunks ──────────────────────────
    print("\nLoading embedding model...")
    embedder = Embedder()

    print(f"Embedding {len(all_chunks)} chunks...")
    print("This will take several minutes for large corpora.\n")

    # Embed in batches to show progress
    batch_size = 64
    all_embeddings = []

    for i in tqdm(range(0, len(all_chunks), batch_size),
                  desc="Embedding batches"):
        batch = all_chunks[i: i + batch_size]
        emb = embedder.model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"\nEmbeddings shape: {embeddings.shape}")


    # ── Step 4: Build FAISS index ─────────────────────────
    # Use IndexIVFFlat for large corpora (faster search)
    # Falls back to IndexFlatIP for small corpora
    n_chunks = len(all_chunks)

    if n_chunks > 1000:
        # IVF index — faster for large corpora
        # n_clusters = sqrt(n_chunks) is standard practice
        n_clusters = min(int(np.sqrt(n_chunks)), 256)
        print(f"Using IndexIVFFlat "
              f"(n_clusters={n_clusters}) for large corpus")

        quantizer = faiss.IndexFlatIP(DIMENSION)
        index = faiss.IndexIVFFlat(
            quantizer, DIMENSION, n_clusters,
            faiss.METRIC_INNER_PRODUCT
        )

        # IVF index must be trained before adding vectors
        print("Training IVF index...")
        index.train(embeddings)
        index.add(embeddings)

        # nprobe: how many clusters to search at query time
        # Higher = more accurate but slower
        index.nprobe = min(n_clusters, 32)

    else:
        # Flat index — exact search for small corpora
        print("Using IndexFlatIP (exact search) for small corpus")
        index = faiss.IndexFlatIP(DIMENSION)
        index.add(embeddings)

    print(f"FAISS index built — {index.ntotal} vectors stored")


    # ── Step 5: Save everything ───────────────────────────
    index_path = os.path.join(index_dir, f"{index_name}.faiss")
    meta_path  = os.path.join(index_dir, f"{index_name}_meta.json")

    faiss.write_index(index, index_path)
    print(f"Index saved : {index_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunks": all_chunks,
            "metadata": all_meta,
            "total_documents": len(doc_files),
            "total_chunks": len(all_chunks),
            "failed_documents": failed
        }, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved: {meta_path}")

    # ── Step 6: Summary ───────────────────────────────────
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"Documents indexed : {len(doc_files) - len(failed)}")
    print(f"Documents failed  : {len(failed)}")
    print(f"Total chunks      : {len(all_chunks)}")
    print(f"Embedding dim     : {DIMENSION}")
    print(f"Index type        : "
          f"{'IVFFlat' if n_chunks > 1000 else 'FlatIP'}")

    if failed:
        print(f"\nFailed documents:")
        for f in failed:
            print(f"  - {f}")

    return {
        "index_path":       index_path,
        "meta_path":        meta_path,
        "total_documents":  len(doc_files) - len(failed),
        "total_chunks":     len(all_chunks),
        "failed":           failed
    }


def load_combined_index(
        index_dir: str  = INDEX_DIR,
        index_name: str = INDEX_NAME
):
    """
    Load the combined FAISS index and metadata.

    Returns:
        index  — FAISS index object
        chunks — list of chunk text strings
        meta   — list of metadata dicts per chunk
    """
    index_path = os.path.join(index_dir, f"{index_name}.faiss")
    meta_path  = os.path.join(index_dir, f"{index_name}_meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Combined index not found at {index_path}. "
            f"Run build_combined_index() first."
        )

    index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Set nprobe if IVF index
    if hasattr(index, "nprobe"):
        index.nprobe = 32

    print(f"Combined index loaded — "
          f"{index.ntotal} vectors, "
          f"{data['total_documents']} documents")

    return index, data["chunks"], data["metadata"]


def search_combined(
        query: str,
        index,
        chunks: list,
        meta: list,
        embedder,
        k: int = 3
) -> list:
    """
    Search the combined index.
    Returns results with source document information.
    """
    query_emb = embedder.model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    distances, indices = index.search(query_emb, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx >= 0:
            results.append({
                "chunk":  chunks[idx],
                "score":  float(score),
                "source": meta[idx]["source"],
                "chunk_index": meta[idx]["chunk_index"]
            })

    return results


# ============================================================
# TEST
# python src/multi_indexer.py
# ============================================================
if __name__ == "__main__":

    # Build the combined index
    summary = build_combined_index()

    print("\nTesting search on combined index...")
    print("=" * 60)

    # Load it back
    from src.embedder import Embedder
    index, chunks, meta = load_combined_index()
    embedder = Embedder()

    # Test queries
    test_queries = [
        "What is parameter sharing in CNNs?",
        "How does gradient descent work?",
        "Explain binary search trees",
        "What is a hash table?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        results = search_combined(
            query, index, chunks, meta, embedder, k=3)

        for i, r in enumerate(results, 1):
            print(f"  Result {i} "
                  f"(score: {r['score']:.4f}) "
                  f"[{r['source']}]")
            print(f"  {r['chunk'][:120]}...")
            print()