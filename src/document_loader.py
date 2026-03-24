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



