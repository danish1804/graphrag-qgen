# EdGenAI UI

Streamlit frontend for the EdGenAI FastAPI backend.

## Features

- Backend health dashboard
- Multi-document RAG status display
- GraphRAG status display
- PDF/PPTX upload for Standard RAG
- PDF/PPTX upload for GraphRAG
- Question generation for single-document Standard RAG
- Question generation for multi-document Standard RAG
- Question generation for GraphRAG
- GraphRAG domain selection with Auto route support
- Result display with generated question, sample answer, sources, key concepts, concept relationships, retrieved chunks, and raw API response
- Judge evaluation score display when `metadata.judge_scores` is returned
- Admin controls for multi-document and GraphRAG index rebuilds
- Admin document registry view using `/admin/new-documents`
- Admin upload-to-domain workflow using `/admin/upload-document`

## Install UI Dependencies

```bash
pip install -r ui/requirements-ui.txt
