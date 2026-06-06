# docs/images/ — Architecture and demo diagrams

Visual assets for presentations. **Optional** — keep locally or commit only if the team wants them on GitHub.

---

## Files

| Asset | Description |
|-------|-------------|
| `system_block_diagram.*` | End-to-end system: UI → API → GraphRAG → LLM → judge |
| `flowchart_document_upload.*` | Admin upload → dedup → registry → incremental rebuild |
| `flowchart_question_generation.*` | Single-domain generate path (retrieval → LLM → judge) |
| `flowchart_hard_multidomain.*` | Hard + auto route: multi-domain scoring and combined context |
| `demo_timeline_flow.*` | Demo session timeline (~40 min) |
| `*.mmd` | Mermaid source (edit these, then re-render) |
| `render_diagrams.py` / `.ps1` | Render `.mmd` → `.svg` / `.png` via Kroki or local tools |

---

## Regenerate PNG/SVG

```powershell
cd docs/images
python render_diagrams.py
# or
.\render_diagrams.ps1
```

---

## Related

- [../README.md](../README.md) — Docs index
- [../../graphrag/README.md](../../graphrag/README.md) — Multi-domain behaviour described in flowcharts
