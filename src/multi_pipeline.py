# ============================================================
# src/multi_pipeline.py
# Multi-document pipeline — retrieves across all indexed PDFs
# Uses combined FAISS index built by multi_indexer.py
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.multi_indexer import (
    build_combined_index,
    load_combined_index,
    search_combined
)
from src.embedder import Embedder
from src.generator import load_model, generate_question


BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_NAME = "combined"


class MultiPipeline:
    def __init__(self):
        self.index        = None
        self.chunks       = None
        self.meta         = None
        self.embedder     = None
        self.model        = None
        self.tokenizer    = None
        self._models_loaded = False
    
    # ----------------------------------------------------------
    # STAGE 1 — Load models
    # ----------------------------------------------------------
    def load_models(self):
        """Load embedding model and Qwen 2.5 into memory."""
        if self._models_loaded:
            print("Models already loaded.")
            return

        print("\n[MultiPipeline] Loading embedding model...")
        self.embedder = Embedder()

        print("\n[MultiPipeline] Loading Qwen 2.5 7B...")
        self.model, self.tokenizer = load_model()

        self._models_loaded = True
        print("\n[MultiPipeline] All models ready.")

    # ----------------------------------------------------------
    # STAGE 2 — Load combined index
    # ----------------------------------------------------------
    def load_index(self, index_dir: str = INDEX_DIR,
                   index_name: str = INDEX_NAME):
        """Load the pre-built combined FAISS index."""
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before load_index()")

        self.index, self.chunks, self.meta = load_combined_index(
            index_dir, index_name)
        
        import json
        meta_path = os.path.join(index_dir, f"{index_name}_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.sources = meta.get("sources", [])

        print(f"[MultiPipeline] Index loaded — "
              f"{len(self.chunks)} chunks across "
              f"{len(set(m['source'] for m in self.meta))} documents")

    # ----------------------------------------------------------
    # STAGE 3 — Rebuild index (if new docs added)
    # ----------------------------------------------------------
    def rebuild_index(self, docs_dir: str = TESTED_DIR):
        """Rebuild the combined index from scratch."""
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before rebuild_index()")

        print("\n[MultiPipeline] Rebuilding combined index...")
        build_combined_index(docs_dir, INDEX_DIR, INDEX_NAME)
        self.load_index()
        print("[MultiPipeline] Index rebuilt and loaded.")


    # ----------------------------------------------------------
    # STAGE 4 — Generate a question
    # ----------------------------------------------------------
    def generate(self, learning_objective: str,
                 difficulty: str = "medium",
                 question_type: str = "conceptual",
                 top_k: int = 3) -> dict:
        """
        Retrieve from combined index and generate question.
        Returns dict with question and source attribution.
        """
        if self.index is None:
            raise RuntimeError(
                "No index loaded. Call load_index() first.")

        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() first.")

        # Retrieve from combined index
        results = search_combined(
            learning_objective,
            self.index,
            self.chunks,
            self.meta,
            self.embedder,
            k=top_k
        )

        # Deduplicate — remove near-identical chunks
        seen = set()
        unique_results = []
        for r in results:
            # Use first 100 chars as dedup key
            key = r["chunk"][:100].strip()
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        results = unique_results if unique_results else results

        # Build context with source labels
        context_parts = []
        for r in results:
            context_parts.append(
                f"[From {r['source']}]\n{r['chunk']}"
            )
        context = "\n\n".join(context_parts)

        # Sources used
        sources = list(set(r["source"] for r in results))

        # Generate question
        question = generate_question(
            context,
            learning_objective,
            self.model,
            self.tokenizer,
            difficulty,
            question_type
        )

        return {
            "question": question,
            "metadata": {
                "learning_objective":  learning_objective,
                "difficulty":          difficulty,
                "question_type":       question_type,
                "retrieval_method":    "Standard RAG (multi doc)",
                "sources":             list(set(
                    r.get("source", "unknown")
                    for r in results
                )),
                "chunks_retrieved":    len(results),
                "avg_retrieval_score": round(float(
                    sum(r["score"] for r in results) /
                    len(results)
                ), 4) if results else 0.0
            }
        }



# ============================================================
# TEST
# python src/multi_pipeline.py
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("MULTI-PIPELINE TEST")
    print("=" * 60)

    pipeline = MultiPipeline()
    pipeline.load_models()
    pipeline.load_index()

    test_cases = [
        {
            "learning_objective": "Explain parameter sharing in CNNs",
            "difficulty":        "medium",
            "question_type":     "conceptual"
        },
        {
            "learning_objective": "How does a hash table handle collisions?",
            "difficulty":        "hard",
            "question_type":     "application"
        },
        {
            "learning_objective": "Define binary search trees",
            "difficulty":        "easy",
            "question_type":     "recall"
        },
    ]

    print("\n" + "=" * 60)
    print("GENERATED QUESTIONS")
    print("=" * 60)

    for i, tc in enumerate(test_cases, 1):
        print(f"\nQuestion {i}:")
        print(f"  Objective : {tc['learning_objective']}")
        print(f"  Difficulty: {tc['difficulty']}")
        print(f"  Type      : {tc['question_type']}")

        result = pipeline.generate(**tc)

        print(f"\n  Question  : {result['question']}")
        print(f"  Sources   : {result['sources_used']}")
        print(f"  Score     : {result['avg_retrieval_score']}")
        print("-" * 60)

    print("\nMulti-pipeline test complete.")