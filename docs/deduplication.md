# Document deduplication (admin pipeline)

## Problem (demo feedback)

Re-uploading a PDF whose content already exists in the graph produced **0 new nodes / 0 new edges** after expensive chunking and entity extraction — correct behaviour, but confusing and slow.

## Approach (Samar)

1. **Byte SHA-256** — identical files skipped immediately (compares on-disk hashes even when registry has no stored fingerprint yet).
2. **Normalised text SHA-256** — same extracted text, different file bytes.
3. **Text similarity** — `difflib.SequenceMatcher` on a bounded text sample; default threshold **0.95**.

**Upload order (partner feedback):**

1. Write upload to a **temporary file**
2. Run dedup assessment
3. If duplicate → **do not** save to `data/tested/`, **do not** register
4. If new → move to `data/tested/` as `source_N.pdf` and register with `indexed: false`

**Rebuild order:** same dedup filter before chunking/graph merge.

**Scope:** comparisons run only against **indexed files in the same domain**, capped by `max_similarity_comparisons` in `config.yml` (default 25).

## Configuration

See `config.yml` → `deduplication`.

## Code layout (modular)

| Module | Role |
|--------|------|
| `utils/content_hash.py` | File and text hashing |
| `utils/text_normalization.py` | Normalise text for hash/similarity |
| `utils/text_similarity.py` | Similarity ratio |
| `utils/registry_content.py` | Store hashes / duplicate flags in registry |
| `services/document_dedup_models.py` | Result dataclasses |
| `services/document_dedup_service.py` | Orchestration |
| `services/admin_upload_handler.py` | Post-upload dedup |
| `services/admin_rebuild_handler.py` | Pre-rebuild filter + fingerprint storage |

## API responses

Upload and rebuild may include:

- `dedup` — assessment details
- `skipped_duplicates` — files not sent to graph merge
- Registry fields: `duplicate_of`, `dedup_reason`, `similarity_score`, `content_hash_bytes`, `content_hash_text`

## Tests

```bash
python -m unittest tests.test_document_dedup
```
