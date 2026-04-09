# ============================================================
# src/document_loader.py
# Loads PDF and PPTX files and returns clean text chunks
# ============================================================

import os
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def load_pptx(file_path: str) -> str:
    """Extract raw text from a PPTX file."""
    prs = Presentation(file_path)
    text = ""
    for slide_num, slide in enumerate(prs.slides, 1):
        slide_text = ""
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text += shape.text.strip() + "\n"
        if slide_text:
            text += f"[Slide {slide_num}]\n{slide_text}\n"
    return text.strip()

def load_document(file_path: str) -> str:
    """Load a document based on its file extension."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if ext == ".pdf":
        print(f"  Loading PDF: {path.name}")
        return load_pdf(file_path)
    elif ext == ".pptx":
        print(f"  Loading PPTX: {path.name}")
        return load_pptx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or PPTX.")

def chunk_text(text: str, chunk_size: int = 500,
               chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    # Remove empty or very short chunks
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]
    return chunks

def load_and_chunk(file_path: str, chunk_size: int = 500,
                   chunk_overlap: int = 50) -> Dict:
    """
    Full pipeline: load a document and return chunks with metadata.

    Returns:
        {
            "file_name": str,
            "file_type": str,
            "total_chars": int,
            "total_chunks": int,
            "chunks": List[str]
        }
    """
    path = Path(file_path)
    print(f"\nProcessing: {path.name}")

    # Load raw text
    raw_text = load_document(file_path)
    print(f"  Extracted {len(raw_text):,} characters")

    # Chunk it
    chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
    print(f"  Created {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    return {
        "file_name": path.name,
        "file_type": path.suffix.lower(),
        "total_chars": len(raw_text),
        "total_chunks": len(chunks),
        "chunks": chunks
    }


# ============================================================
# TEST - run this file directly to verify it works
# python src/document_loader.py
# ============================================================
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("DOCUMENT LOADER TEST")
    print("=" * 50)

    # Check if a file path was passed as argument
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        result = load_and_chunk(test_file)
        print(f"\nResults:")
        print(f"  File      : {result['file_name']}")
        print(f"  Type      : {result['file_type']}")
        print(f"  Chars     : {result['total_chars']:,}")
        print(f"  Chunks    : {result['total_chunks']}")
        print(f"\nFirst chunk preview:")
        print("-" * 40)
        print(result['chunks'][0])
        print("-" * 40)
        print(f"\nLast chunk preview:")
        print("-" * 40)
        print(result['chunks'][-1])
        print("-" * 40)
    else:
        # Create a small test text file to verify chunking works
        print("No file provided - running chunking test on sample text")
        sample = """
        Machine learning is a subset of artificial intelligence.
        It enables systems to learn from data without being explicitly programmed.

        Supervised learning uses labelled training data.
        The model learns to map inputs to outputs based on example pairs.
        Common algorithms include linear regression, decision trees, and neural networks.

        Unsupervised learning finds patterns in unlabelled data.
        Clustering and dimensionality reduction are common techniques.
        K-means clustering groups data points into k clusters.

        Deep learning uses neural networks with many layers.
        Convolutional neural networks are used for image recognition.
        Recurrent neural networks handle sequential data like text.
        """ * 10  # repeat to create enough text for multiple chunks

        chunks = chunk_text(sample, chunk_size=200, chunk_overlap=20)
        print(f"Sample text chunked into {len(chunks)} chunks")
        print(f"\nChunk 1: {chunks[0][:100]}...")
        print(f"Chunk 2: {chunks[1][:100]}...")
        print("\nChunking works correctly.")

    print("\nDocument loader ready.")



