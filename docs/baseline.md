# Phase 0 Baseline and Gap Register

This document freezes current behavior before refactor work starts.
Use it as the source of truth to compare pre/post changes.

## Scope

- Backend runtime and endpoint behavior (`main_gr.py`)
- GraphRAG generation path and metadata shape
- Admin ingestion/rebuild path
- Dataset generation path (small batch smoke)
- Script and documentation consistency

## Environment

- Project root: `D:/Semester 4/AI_Postr_Graduate_Project/graphrag_qgen`
- Canonical startup command:
  - `python main_gr.py --mode full`

## Baseline Validation Checklist

Run all checks in order and capture evidence in the "Evidence Log" section.

### 1) Startup and health

1. Start server in full mode:
   - `python main_gr.py --mode full`
2. Wait for startup logs:
   - `[Server] Judge model ready.`
   - `[Server] Server is live.`
3. Hit health endpoint:
   - `GET /health`

Pass criteria:
- HTTP 200
- `status == "ok"`
- `models_loaded == true`

### 2) GraphRAG status

1. Call:
   - `GET /graphrag/status`

Pass criteria:
- HTTP 200
- response includes `domains` map with availability per domain

### 3) Generate one question per difficulty

Run `POST /graphrag/generate` three times with same learning objective, varying:
- `difficulty: easy`
- `difficulty: medium`
- `difficulty: hard`

Recommended payload:

```json
{
  "learning_objective": "Explain how CNN parameter sharing improves learning efficiency",
  "difficulty": "medium",
  "question_type": "conceptual"
}
```

Pass criteria for each run:
- HTTP 200
- non-empty `question`
- response includes `metadata`
- metadata includes retrieval context and generation details (e.g. routed domain, concepts/chunks/sources)

### 4) Domain fallback behavior

1. Call `POST /graphrag/generate` with invalid domain:
   - `"domain": "invalid_domain_name"`

Pass criteria:
- request still succeeds (fallback to auto-routing)
- output includes normal generated result

### 5) Admin ingestion and incremental rebuild

1. Check unregistered files:
   - `GET /admin/new-documents`
2. If test file available, upload to valid domain:
   - `POST /admin/upload-document?domain=database` with `multipart/form-data`
3. Trigger incremental update:
   - `POST /admin/rebuild-domain` with body `{"domain":"database"}`

Pass criteria:
- all endpoints return HTTP 200
- rebuild response includes node/edge update counts

### 6) Dataset smoke run

1. Run small dataset generation batch:
   - `python dataset/generate_dataset.py --batch 3`

Pass criteria:
- run completes without crash
- output file updated in `dataset/outputs/`
- statuses updated in `dataset/inputs/input_questions.csv`

## Evidence Log (fill during execution)

- Date/time:
- Commit/branch:
- Runtime mode:
- Health response snapshot:
- GraphRAG status snapshot:
- Easy generate response ID/summary:
- Medium generate response ID/summary:
- Hard generate response ID/summary:
- Invalid domain fallback result:
- Admin upload result:
- Admin rebuild result:
- Dataset smoke result:
- Notes:

## Baseline API Contract Snapshot

Current canonical request model for GraphRAG generation (`main_gr.py`):
- `learning_objective: str`
- `difficulty: easy | medium | hard`
- `question_type: recall | conceptual | application`
- `top_k_nodes: int`
- `neighbour_depth: int`
- `domain: Optional[str]`

Current behavior notes:
- Invalid provided domain does not fail request; it falls back to auto-routing.
- Hard difficulty can activate multi-domain path if no domain specified.

## Gap Register (Phase 0 Findings)

Priority labels:
- P0 = correctness/safety risk
- P1 = maintainability/operational risk
- P2 = documentation/consistency risk

### P0

1. Runtime config mutation through admin API
- File: `main_gr.py`
- Endpoint: `/admin/add-domain`
- Issue: modifies `config.yml` during runtime.
- Risk: config corruption, drift across environments, concurrency hazards.
- Planned fix phase: Phase 3.

2. Path hardcoding in operational script
- File: `scripts/test_document_batch.py`
- Issue: `BASE_DATA_DIR` hardcoded to one machine path.
- Risk: script fails or behaves incorrectly in other environments.
- Planned fix phase: Phase 8.

### P1

3. Multiple overlapping entrypoints
- Files: `main.py`, `main_multi.py`, `main_gr.py`
- Issue: competing startup surfaces for similar responsibilities.
- Risk: behavior drift and operator confusion.
- Planned fix phase: Phase 1.

4. Thin automated test safety net
- Files: test scripts and ad hoc smoke files across repo
- Issue: no robust, centralized pytest-based contract/regression coverage.
- Risk: refactor regressions likely.
- Planned fix phase: Phase 7.

### P2

5. Endpoint method mismatch in startup log
- File: `main_gr.py`
- Issue: startup print lists `/admin/new-documents` as `POST`, actual route is `GET`.
- Risk: operator confusion.
- Planned fix phase: Phase 8.

6. Documentation references missing files
- File: `README.md`
- Issues:
  - references `.env.example` (not present in repository)
  - references `scripts/validate_docs.py` (not present)
- Risk: setup friction and failed onboarding.
- Planned fix phase: Phase 8.

7. Script README argument drift
- File: `scripts/README.md`
- Issue: documents flags not implemented in scripts:
  - `check_doc_topics.py --file`
  - `check_top_entities.py --domain --top`
  - `get_sample_chunks.py --file`
  - `test_document_batch.py --domain`
- Risk: operator commands fail despite docs.
- Planned fix phase: Phase 8.

## Phase 1 Ready Checklist

Phase 1 can start when:
- baseline checklist above is executed at least once and evidence captured
- one canonical startup command is accepted by team (`main_gr.py --mode full`)
- no blocking environment issue remains unresolved

## Phase 1 Execution Notes

Completed in this branch/session:
- `main.py` converted to compatibility wrapper re-exporting `main_gr.app`
- `main_multi.py` converted to compatibility wrapper re-exporting `main_gr.app`
- README updated to mark wrappers as deprecated and document canonical runtime
- startup endpoint log corrected for `/admin/new-documents` (`GET`)

## Phase 2 Execution Notes

Completed in this branch/session:
- Added stricter request validation in `main_gr.py` using typed constraints:
  - `difficulty` and `question_type` now `Literal` constrained
  - `learning_objective` has minimum length validation
  - `top_k` / `top_k_nodes` / `neighbour_depth` bounded with `Field`
- Added explicit response models on core endpoints while preserving payload shape:
  - `GET /health`
  - `POST /upload`
  - `POST /multi/rebuild`
  - `POST /graphrag/generate`
  - `POST /admin/rebuild-domain`
- Kept GraphRAG response compatibility by allowing extra top-level fields in the response model.

## Phase 3 Execution Notes

Completed in this branch/session:
- Stopped runtime `config.yml` mutation in `/admin/add-domain`.
- Added new mutable runtime state store:
  - `utils/runtime_state.py`
  - persisted at `data/runtime_state.json`
- Added registry helper `ensure_domain(...)` in `utils/registry.py` so runtime-added domains can accept uploads/index state safely.
- Added domain-catalog sync in `main_gr.py` (`_sync_runtime_domains`) to merge active domains from:
  - static config (`config.yml` enabled domains)
  - registry domains
  - runtime state domains
- Updated domain-aware endpoints to refresh active domains before validation/routing/rebuild calls.

## Phase 4 Execution Notes

Completed in this branch/session:
- Introduced application service layer modules:
  - `services/domain_catalog_service.py`
  - `services/graphrag_generation_service.py`
- Wired `main_gr.py` to use `DomainCatalogService` for domain synchronization instead of inline route-file logic.
- Extracted GraphRAG generation fallback orchestration (invalid explicit domain -> auto-route) into `GraphRAGGenerationService`.
- Kept endpoint contracts and payload shapes unchanged while thinning route handler responsibilities.
- Added `services/status_service.py` and moved health/status payload construction out of route handlers.
- Added `services/admin_domain_service.py` and moved admin domain operations out of route handlers:
  - upload-document workflow
  - rebuild-domain workflow
  - add-domain workflow
- `main_gr.py` endpoints now mainly validate input and delegate to service layer methods.

## Phase 5 Execution Notes

Completed in this branch/session:
- Added atomic JSON write utility in `utils/atomic_io.py`.
- Switched mutable state persistence to atomic writes:
  - `utils/registry.py::save_registry`
  - `utils/runtime_state.py::save_runtime_state`
- Added index repository abstraction in `services/index_repository.py` for domain incremental update/reload operations.
- Wired admin rebuild workflow to use `IndexRepository` through `AdminDomainService`.
- Added structured rebuild timing (`elapsed_seconds`) to admin rebuild response model and payload.

## Phase 6 Execution Notes

Completed in this branch/session:
- Added `utils/prompts.py` for generation prompt templates and `PROMPT_VERSION`.
- Refactored `utils/generator.py`:
  - removed erroneous `dataclasses_json` import shadowing config
  - centralized decoding via `config.yml` (`max_new_tokens_question`, `max_new_tokens_answer`, `temperature`, `do_sample`)
  - shared `_generate_from_messages()` helper for question/answer paths
  - added `get_generation_trace()` metadata helper
- Extended `config.yml` with explicit question/answer token limits and `prompts.version`.
- GraphRAG responses now include generation trace fields in `metadata` (`prompt_version`, model/temperature/token settings).
- Evaluation JSON records now include `prompt_version` and `rubric_version`.
