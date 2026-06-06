# EdGenAI UI

Streamlit frontend for the EdGenAI FastAPI backend.

## Demo (Streamlit)

Use sidebar **Demo presets** on the Generate page, then follow controls documented below.

```bash
# Terminal 1
python main_gr.py --mode full

# Terminal 2
cd ui
streamlit run app.py
```

Default navigation opens **Generate Question** with GraphRAG selected. Use sidebar **Demo presets** to fill learning objectives.

## Generate page controls

### Difficulty (easy / medium / hard)

Controls **retrieval and routing** only. Values are read from `config.yml` and shown in the blue info box:

| Difficulty | top_k_nodes | neighbour_depth | max_chunks | Multi-domain |
|------------|-------------|-----------------|------------|--------------|
| easy | 2 | 1 | 3 | No |
| medium | 3 | 1 | 5 | No |
| hard | 5 | 2 | 7 | Yes, if domain = **Auto route** |

Tutors **cannot** change K in the UI — edit `config.yml` and restart the server to tune.

### Question type (recall / conceptual / application / case_study)

Controls **how the LLM writes** the question (Bloom level + prompt template). Does **not** change chunk counts or domain routing.

| Type | Bloom | Output style |
|------|-------|--------------|
| recall | 1 | Define, list, state |
| conceptual | 2 | Explain, compare |
| application | 3 | Design, analyse, apply |
| case_study | 4 | **SCENARIO:** … then **QUESTION:** … |

**Examples:**

- `medium + conceptual` → one domain, 5 chunks, explain-style question.
- `hard + case_study + Auto route` → multi-domain, 7 chunks per domain, scenario + question.
- `hard + case_study + database` → single domain, 7 chunks, scenario + question.

### What “K” means in the UI info box

The UI no longer has an editable “Top K” field. It displays config-driven retrieval parameters:

- **top_k_nodes** — how many graph **seed nodes** FAISS selects (graph entry points).
- **neighbour_depth** — how many **edge hops** expand related concepts.
- **max_chunks** — how many **text chunks** go into the LLM prompt.

For Standard RAG modes, the box shows **top_k_results** instead (flat chunk retrieval, no graph).

### Domain selection

- **Auto route** — embedding router picks domain(s). Hard enables multi-domain.
- **Specific domain** — forces that domain; multi-domain is disabled for hard.

## Features

- Backend health dashboard
- Multi-document RAG status display
- GraphRAG status display
- PDF/PPTX upload for GraphRAG via **Admin** page only (general Upload page removed)
- Question generation for single-document Standard RAG
- Question generation for multi-document Standard RAG
- Question generation for GraphRAG (default mode)
- GraphRAG domain selection with Auto route support
- Retrieval params (`top_k_nodes`, `neighbour_depth`, `max_chunks`) shown from `config.yml` per difficulty
- Hard + Auto route hint for multi-domain retrieval
- Routing banner: `retrieval_method`, domains used, domain scores
- Bloom level labels on question type (recall / conceptual / application / **case_study**)
- Demo preset learning objectives in sidebar
- Result display: question, sample answer, judge scores, chunks (seed/neighbour), raw API (collapsed)
- Judge evaluation score display when `metadata.judge_scores` is returned
- Admin controls for multi-document and GraphRAG index rebuilds
- Admin document registry view using `/admin/new-documents`
- Admin upload-to-domain workflow using `/admin/upload-document`

## Install UI Dependencies

```bash
pip install -r ui/requirements-ui.txt
