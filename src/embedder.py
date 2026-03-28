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
 