# ============================================================
# src/embedder.py
# Embeds text chunks using BAAI/bge-large-en-v1.5
# and builds a FAISS index
# ============================================================

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # two dirnames
)
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DIMENSION  = 1024  # bge-large output dimension


class Embedder:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")

    def embed(self, texts: list) -> np.ndarray:
        """Convert a list of strings to normalized embeddings."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        return np.array(embeddings, dtype=np.float32)


def build_index(chunks: list, index_dir: str,
                index_name: str = "index") -> dict:
    """
    Embed chunks and build a FAISS index.
    Saves index + metadata to disk.

    Returns summary dict.
    """
    os.makedirs(index_dir, exist_ok=True)

    embedder = Embedder()
    print(f"\nEmbedding {len(chunks)} chunks...")
    embeddings = embedder.embed(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index - IndexFlatIP = inner product
    # (cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings)
    print(f"FAISS index built - {index.ntotal} vectors stored")

    # Save FAISS index
    index_path = os.path.join(index_dir, f"{index_name}.faiss")
    faiss.write_index(index, index_path)
    print(f"Index saved: {index_path}")

    # Save metadata (chunk text alongside index)
    meta_path = os.path.join(index_dir, f"{index_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved: {meta_path}")

    return {
        "index_path": index_path,
        "meta_path": meta_path,
        "total_chunks": len(chunks),
        "dimension": DIMENSION
    }


def load_index(index_dir: str, index_name: str = "index"):
    """Load a saved FAISS index and its metadata."""
    index_path = os.path.join(index_dir, f"{index_name}.faiss")
    meta_path  = os.path.join(index_dir, f"{index_name}_meta.json")

    index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return index, meta["chunks"]


def search(query: str, index, chunks: list,
           embedder: Embedder, k: int = 3) -> list:
    """
    Search the FAISS index for the top-k most relevant chunks.

    Returns list of dicts with chunk text and score.
    """
    query_embedding = embedder.embed([query])
    distances, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx >= 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(score),
                "index": int(idx)
            })

    return results


# ============================================================
# TEST - run directly to verify
# python src/embedder.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.document_loader import load_and_chunk

    print("=" * 50)
    print("EMBEDDER TEST")
    print("=" * 50)

    # Use the PDF we already tested
    pdf_path = os.path.join(BASE_DIR, "data", "tested", "source_1.pdf")
    index_dir = os.path.join(BASE_DIR, "indexes")

    # Step 1: Load and chunk
    result = load_and_chunk(pdf_path)
    chunks = result["chunks"]

    # Step 2: Build index
    summary = build_index(chunks, index_dir, index_name="cnn_lecture")
    print(f"\nIndex summary: {summary}")

    # Step 3: Load index back and test search
    print("\nLoading index and testing search...")
    index, chunks_loaded = load_index(index_dir, "cnn_lecture")
    embedder = Embedder()

    # Test queries
    test_queries = [
        "What is a convolutional neural network?",
        "How does pooling work?",
        "What is parameter sharing in CNNs?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        results = search(query, index, chunks_loaded, embedder, k=2)
        for i, r in enumerate(results, 1):
            print(f"Result {i} (score: {r['score']:.4f}):")
            print(r['chunk'][:200])
            print() 