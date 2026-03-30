# ============================================================
# src/pipeline.py
# Connects document loader + embedder + generator
# into a single end-to-end pipeline
# for simple single-document RAG question generation.
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.document_loader import load_and_chunk
from src.embedder import build_index, load_index, search, Embedder
from src.generator import load_model, generate_question


INDEX_DIR = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen\indexes"
DATA_DIR  = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen\data"


class Pipeline:
    def __init__(self):
        self.index        = None
        self.chunks       = None
        self.embedder     = None
        self.model        = None
        self.tokenizer    = None
        self.index_name   = None
        self._models_loaded = False

    # ----------------------------------------------------------
    # STAGE 1 - Load models into memory (call once at startup)
    # ----------------------------------------------------------
    def load_models(self):
        """Load embedding model and Qwen 2.5 into memory."""
        if self._models_loaded:
            print("Models already loaded.")
            return

        print("\n[Pipeline] Loading embedding model...")
        self.embedder = Embedder()

        print("\n[Pipeline] Loading Qwen 2.5 7B...")
        self.model, self.tokenizer = load_model()

        self._models_loaded = True
        print("\n[Pipeline] All models ready.")

    # ----------------------------------------------------------
    # STAGE 2 - Ingest a document
    # ----------------------------------------------------------
    def ingest(self, file_path: str) -> dict:
        """
        Load a PDF or PPTX, chunk it, embed it,
        and save a FAISS index.

        Returns ingestion summary.
        """
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before ingest()")

        # Derive index name from file name
        file_name = os.path.basename(file_path)
        self.index_name = os.path.splitext(file_name)[0]

        print(f"\n[Pipeline] Ingesting: {file_name}")

        # Load and chunk
        doc_result = load_and_chunk(file_path)
        chunks = doc_result["chunks"]

        # Build FAISS index
        index_result = build_index(
            chunks, INDEX_DIR, self.index_name)

        # Load into memory for immediate use
        self.index, self.chunks = load_index(
            INDEX_DIR, self.index_name)

        print(f"[Pipeline] Ingestion complete — "
              f"{len(self.chunks)} chunks indexed.")

        return {
            "file_name":    file_name,
            "total_chunks": len(chunks),
            "index_name":   self.index_name,
            "index_path":   index_result["index_path"]
        }

    # ----------------------------------------------------------
    # STAGE 3 - Load an existing index
    # ----------------------------------------------------------
    def load_existing_index(self, index_name: str):
        """Load a previously built FAISS index."""
        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() before loading index.")

        self.index_name = index_name
        self.index, self.chunks = load_index(
            INDEX_DIR, index_name)

        print(f"[Pipeline] Loaded index: {index_name} "
              f"({len(self.chunks)} chunks)")


    # ----------------------------------------------------------
    # STAGE 4 - Generate a question
    # ----------------------------------------------------------
    def generate(self, learning_objective: str,
                 difficulty: str = "medium",
                 question_type: str = "conceptual",
                 top_k: int = 3) -> dict:
        """
        Full pipeline: objective -> retrieve -> generate.

        Returns dict with question and metadata.
        """
        if self.index is None:
            raise RuntimeError(
                "No index loaded. Call ingest() or "
                "load_existing_index() first.")

        if not self._models_loaded:
            raise RuntimeError(
                "Call load_models() first.")

        # Retrieve relevant chunks
        results = search(
            learning_objective,
            self.index,
            self.chunks,
            self.embedder,
            k=top_k
        )

	# Assemble context
        context = "\n\n".join([r["chunk"] for r in results])

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
                "retrieval_method":    "Standard RAG (single doc)",
                "index_name":          self.index_name,
                "chunks_retrieved":    len(results),
                "avg_retrieval_score": round(float(
                    sum(r["score"] for r in results) /
                    len(results)
                ), 4) if results else 0
            }
        }


# ============================================================
# TEST - run directly
# python src/pipeline.py
# ============================================================
if __name__ == "__main__":

    print("=" * 55)
    print("PIPELINE END-TO-END TEST")
    print("=" * 55)

    PDF_PATH = os.path.join(DATA_DIR, "source_1.pdf")

    # Create pipeline
    pipeline = Pipeline()

    # Load models once
    pipeline.load_models()

    # Ingest document
    pipeline.ingest(PDF_PATH)

    # Generate questions with different settings
    test_cases = [
        {
            "learning_objective": "Explain how CNNs use "
                                  "parameter sharing",
            "difficulty":   "medium",
            "question_type": "conceptual"
        },
        {
            "learning_objective": "Apply max pooling to "
                                  "reduce feature map size",
            "difficulty":   "hard",
            "question_type": "application"
        },
        {
            "learning_objective": "Define sparse connectivity "
                                  "in neural networks",
            "difficulty":   "easy",
            "question_type": "recall"
        },
    ]

    print("\n" + "=" * 55)
    print("GENERATED QUESTIONS")
    print("=" * 55)

    for i, tc in enumerate(test_cases, 1):
        print(f"\nQuestion {i}:")
        print(f"  Objective : {tc['learning_objective']}")
        print(f"  Difficulty: {tc['difficulty']}")
        print(f"  Type      : {tc['question_type']}")

        result = pipeline.generate(**tc)

        print(f"\n  Question  : {result['question']}")
        print(f"  Retrieval score: {result['avg_retrieval_score']}")
        print("-" * 55)

    print("\nPipeline test complete.")
    print("Ready for FastAPI wrapping.")







