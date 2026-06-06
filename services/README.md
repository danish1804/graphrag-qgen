# services/ — Application service layer

Business logic extracted from `main_gr.py` FastAPI route handlers. Routes stay thin: validate input, call a service, return JSON.

**Why it exists:** Admin upload, deduplication, GraphRAG generation, and status reporting were growing inside `main_gr.py`. Services keep those workflows testable and modular.

---

## Files

```
services/
├── __init__.py
├── admin_domain_service.py      # Upload + incremental rebuild orchestration
├── admin_upload_handler.py      # Temp file → dedup → save/register
├── admin_rebuild_handler.py     # Pre-rebuild dedup filter + fingerprint storage
├── document_dedup_service.py    # Byte/text hash + similarity orchestration
├── document_dedup_models.py     # DedupResult dataclasses
├── domain_catalog_service.py    # Valid domains from config + registry sync
├── graphrag_generation_service.py  # Wraps GraphRAGPipeline.generate calls
├── index_repository.py          # Loads/saves FAISS + graph artifacts
└── status_service.py              # /health and GraphRAG status payloads
```

---

## Module roles

| Module | Use case |
|--------|----------|
| `AdminDomainService` | **Admin upload** and **incremental rebuild** — coordinates dedup, registry, graph pipeline, runtime state |
| `DocumentDedupService` | Reject byte-identical, text-identical, or near-duplicate PDFs (≥0.95 similarity) **within the target domain** |
| `admin_upload_handler` | Hash **before** writing to `data/tested/`; duplicate → no save, no registry entry |
| `admin_rebuild_handler` | Skip duplicates before chunking; store content fingerprints after successful index |
| `GraphRagGenerationService` | Single entry for `/graphrag/generate` — routing, multi-domain, judge |
| `DomainCatalogService` | Enabled domains from `config.yml`; sync registry domain lists |
| `IndexRepository` | Domain index paths under `indexes/` |
| `StatusService` | Health check fields (`models_loaded`, domain counts) |

---

## Admin upload flow (with dedup)

```
POST /admin/upload-document?domain=database
    ↓
save_temp_file()
    ↓
DocumentDedupService.assess()  — byte hash, text hash, similarity
    ↓
duplicate? → reject (nothing in tested/ or registry)
    ↓
move_temp_to_tested() → source_N.pdf
register_new_file_for_rebuild() → indexed: false
```

See [../docs/deduplication.md](../docs/deduplication.md) for thresholds and config.

---

## Incremental rebuild flow

```
POST /admin/rebuild-domain {"domain": "database"}
    ↓
AdminDomainService.rebuild_domain()
    ↓
filter_and_skip_duplicates() — skip near-dupes vs indexed files
    ↓
load_chunks_for_files() → graph merge via GraphRAGPipeline
    ↓
store_fingerprints_for_files() — persist hashes in registry metadata
    ↓
mark indexed: true in document_registry.json
```

---

## Related

- [../main_gr.py](../main_gr.py) — FastAPI routes that call these services
- [../utils/registry_content.py](../utils/registry_content.py) — Registry fingerprint fields
- [../docs/deduplication.md](../docs/deduplication.md) — Partner requirement spec
- [../data/README.md](../data/README.md) — `tested/`, `uploads/`, registry
- [../README.md](../README.md) — Main project README
