# tests/ — Unit tests

Automated tests for isolated logic (no GPU, no running server). Live API checks live in `scripts/test_new_features.py`.

---

## Files

```
tests/
├── __init__.py
└── test_document_dedup.py   # Hash + text similarity unit tests
```

---

## test_document_dedup.py

Covers deduplication building blocks used by `services/document_dedup_service.py`:

| Test area | What it verifies |
|-----------|------------------|
| `sha256_bytes` / `sha256_file` | Stable byte hashing for exact duplicate detection |
| `similarity_ratio` | Normalised text comparison (identical → ~1.0, minor edits → high score) |
| Near-duplicate threshold | Behaviour aligned with `config.yml` → `deduplication.similarity_threshold` (0.95) |

---

## Run tests

```bash
# All unit tests
python -m unittest discover -s tests -v

# Single module
python -m unittest tests.test_document_dedup -v
```

---

## Live / integration checks (not in tests/)

| Script | Purpose |
|--------|---------|
| [../scripts/test_new_features.py](../scripts/test_new_features.py) | `case_study` config, dedup integration, optional `--live --generate` against port 8000 |
| [../scripts/verify_refactor_changes.py](../scripts/verify_refactor_changes.py) | Post-refactor API smoke tests |

Formal pytest UAT with real PDFs is planned post-demo (see project backlog).

---

## Related

- [../services/README.md](../services/README.md) — Dedup orchestration
- [../scripts/README.md](../scripts/README.md) — Verification scripts
- [../docs/deduplication.md](../docs/deduplication.md) — Dedup design
