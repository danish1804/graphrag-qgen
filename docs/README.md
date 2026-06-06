# docs/ — Project documentation

Supplementary documentation beyond module READMEs. All paths are relative to the repo root.

---

## Committed guides

| Document | Purpose |
|----------|---------|
| [baseline.md](baseline.md) | Phase 0 baseline checklist — pre/post refactor validation, evidence log template |
| [deduplication.md](deduplication.md) | Admin upload/rebuild dedup design (byte hash, text hash, similarity ≥0.95) |

---

## Optional local assets (not required in repo)

You may keep presentation-only files **locally** without committing them:

| Path | Typical use |
|------|-------------|
| `demo_script.md` | Short click-by-click demo checklist |
| `demo_presenter_script.md` | Full spoken narration for industry demo |
| `images/` | Mermaid sources, PNG/SVG flowcharts for slides |

If present locally, do **not** `git add` them unless the team agrees to publish them. The root [README.md](../README.md) points to [ui/README.md](../ui/README.md) for Streamlit demo controls instead.

---

## Diagram assets (`images/`)

When included locally, `images/` holds architecture and flow diagrams (block diagram, upload flow, hard multi-domain, question generation). Source `.mmd` files can be rendered with `images/render_diagrams.py` or `render_diagrams.ps1`.

See [images/README.md](images/README.md) when that folder is present.

---

## Related module docs

| Topic | README |
|-------|--------|
| GraphRAG routing, multi-domain | [../graphrag/README.md](../graphrag/README.md) |
| Difficulty vs question type | [../README.md](../README.md) |
| UI controls | [../ui/README.md](../ui/README.md) |
| Services / admin | [../services/README.md](../services/README.md) |
| Data folders | [../data/README.md](../data/README.md) |
