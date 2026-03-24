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

  