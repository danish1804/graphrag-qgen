# postman/ — API collection (backup testing)

Postman assets for manual API testing when the Streamlit UI is unavailable or when demonstrating raw HTTP contracts.

---

## Files

```
postman/
├── P000233DS_GraphRAG.postman_collection.json   # Main collection
├── environments/
│   ├── Local Development.environment.yaml
│   └── Local_Development
└── globals/
    └── workspace.globals.yaml
```

---

## Setup

1. Import `P000233DS_GraphRAG.postman_collection.json` into Postman.
2. Select environment **Local Development** with base URL `http://localhost:8000`.
3. Start server: `python main_gr.py --mode full`.

---

## Key requests

| Request | Use case |
|---------|----------|
| `GET /health` | Server up, models loaded |
| `GET /graphrag/status` | Domain node/chunk counts |
| `POST /graphrag/generate` | Easy / medium / hard bodies (include `question_type`: recall, conceptual, application, **case_study**) |
| `POST /admin/upload-document` | Admin PDF upload (dedup runs server-side) |
| `POST /admin/rebuild-domain` | Incremental domain graph rebuild |

**Hard multi-domain:** omit `domain` or set routing to auto; difficulty must be `hard`.

---

## Related

- [../README.md](../README.md) — API overview and curl examples
- [../ui/README.md](../ui/README.md) — Primary demo surface (Streamlit)
