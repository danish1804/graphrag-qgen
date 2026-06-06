# data/ : Data Structure and Document Management

All PDF source documents, the document registry, and processing logs live here.

---

## Folder Structure

```
data/
├── raw/                      # Incoming PDFs : before processing
├── rejected/                 # PDFs that failed validation
├── tested/                   # Validated PDFs : permanent master library
├── uploads/                  # Admin API staging area : temporary
├── document_registry.json    # Registry of all PDFs and domain assignments
└── rename_log.txt            # History of all rename operations
```

---

## raw/

**Purpose:** Drop zone for new PDFs before they are renamed and validated.

**What goes here:**
- New textbooks downloaded from publishers
- Lecture notes from educators
- Any new IT material to add to the knowledge base

**What to do after dropping a file here:**
```bash
# Preview what would be renamed:
python scripts/rename_docs.py --dry-run

# Apply rename (moves file to tested/ after renaming):
python scripts/rename_docs.py
```

**After running rename_docs.py:**
- File is renamed to `source_N.pdf`
- File is moved to `data/tested/`
- `last_source_number` in registry is updated
- `data/raw/` is empty again

> Files in `raw/` are transient : they should not stay here. Run `rename_docs.py` promptly after adding new files.

---

## rejected/

**Purpose:** PDFs that failed the validation script are moved here automatically.

**Why a PDF gets rejected:**
- Corrupt or unreadable file
- Too few extractable characters (likely scanned image PDF without OCR)
- Empty after text extraction
- Password protected
- Less than minimum chunk threshold (5 chunks)

**What to do with rejected files:**
```
1. Check rejected/ for the filename
2. Try opening manually to see if it is readable
3. If scanned : run OCR tool first (e.g. Adobe Acrobat, tesseract)
4. If password protected : request unprotected version
5. If genuinely corrupt : discard and source a new copy
6. Once fixed : move back to raw/ and re-run rename_docs.py
```

> Do NOT manually move files from `rejected/` to `tested/` : always re-run through the validation pipeline.

---

## tested/

**Purpose:** The permanent master library of all validated, renamed PDFs.

**Key properties:**
- Every PDF here has passed the validation script
- Every PDF is named in `source_N.pdf` format
- Files NEVER leave this folder : they are only added
- This is the source of truth for all graph building

**Current contents:**
```
63 PDFs across 5 domains:
  source_1.pdf  -> source_22.pdf   (software_engineering)
  source_5.pdf  -> source_43.pdf   (operating_systems)
  source_6.pdf  -> source_44.pdf   (computer_architecture)
  source_7.pdf  -> source_49.pdf   (database)
  source_50.pdf -> source_64.pdf   (deep_learning)
```

> Full domain mapping is in `document_registry.json`.

**What uses tested/:**
```
graphrag_domain_indexer.py  -> reads PDFs to build graphs
standard_rag/multi_indexer.py -> reads PDFs for Standard RAG
scripts/test_document_batch.py -> validates loading
admin rebuild endpoint -> reads new PDFs after upload
```

**What NOT to do:**
```
 Do not delete files from tested/
 Do not rename files manually in tested/
 Do not move files out of tested/
   -> indexes/ graphs reference source filenames
   -> deleting breaks graph metadata
```

---

## uploads/

**Purpose:** Temporary staging area for admin API document uploads.

**Lifecycle of a file in uploads/:**
```
Admin calls POST /admin/upload-document
    ↓
File written to temp path; dedup assessed (byte + text hash + similarity)
    ↓
Duplicate? → rejected — NOT saved to tested/, NOT registered
    ↓
New file → saved as source_N.pdf in data/tested/
Registered in document_registry.json (indexed: false)
    ↓
Admin calls POST /admin/rebuild-domain
    ↓
Incremental graph merge; fingerprints stored; indexed: true
    ↓
uploads/ is empty again
```

**Deduplication (Samar requirement):** See [../docs/deduplication.md](../docs/deduplication.md). Comparisons are scoped to **indexed files in the same domain** only. Near-duplicates at similarity ≥ `0.95` are skipped before any graph work.

**Legacy note:** Older docs described saving directly to `uploads/`; the current path uses **temp → dedup → `tested/`** via `services/admin_upload_handler.py`.

**When uploads/ has files but rebuild has not run:**
```bash
# Check what is pending:
curl http://localhost:8000/admin/new-documents

# Trigger rebuild:
curl -X POST http://localhost:8000/admin/rebuild-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "database"}'
```

**If server crashed between upload and rebuild:**
```bash
# File is stuck in uploads/ : manually trigger rebuild:
curl -X POST http://localhost:8000/admin/rebuild-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "database"}'
# Rebuild will find the file and process it
```

> `uploads/` should normally be empty. A file here means a rebuild is pending.

---

## document_registry.json

**Purpose:** Single source of truth for all PDF tracking. Tracks domain assignments, indexing status, and naming counter.

**Structure:**
```json
{
  "last_source_number": 64,
  "domains": {
    "deep_learning": {
      "description": "Deep Learning - CNNs, pooling, backprop",
      "files": {
        "source_50.pdf": {"indexed": true, "added": "2026-03-01"},
        "source_63.pdf": {"indexed": true, "added": "2026-05-02"},
        "source_64.pdf": {"indexed": true, "added": "2026-05-03"}
      }
    },
    "database": {
      "description": "Database Systems - SQL, transactions, indexing",
      "files": {
        "source_7.pdf":  {"indexed": true, "added": "2026-03-01"},
        "source_8.pdf":  {"indexed": true, "added": "2026-03-01"}
      }
    },
    "software_engineering": { ... },
    "operating_systems":    { ... },
    "computer_architecture":{ ... }
  }
}
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `last_source_number` | int | Highest source number assigned. Next file = this + 1 |
| `domains` | dict | Domain-grouped file registry |
| `files[filename].indexed` | bool | True if file is in the knowledge graph |
| `files[filename].added` | str | Date file was registered (YYYY-MM-DD) |

**Who reads it:**
```
rename_docs.py              -> reads last_source_number
graphrag_domain_indexer.py  -> reads file lists per domain
admin upload endpoint       -> registers new files
admin rebuild endpoint      -> reads unindexed files, marks as indexed
GET /admin/new-documents    -> reads all files to find unregistered ones
utils/registry.py           -> all registry operations
```

**Who writes it:**
```
scripts/create_registry.py  -> initialises (run once)
rename_docs.py              -> updates last_source_number
admin upload endpoint       -> adds new file entry
admin rebuild endpoint      -> marks files as indexed
```

**Initialise registry (first time setup):**
```bash
python scripts/create_registry.py
```

**View registry summary:**
```bash
python -c "
from utils.registry import get_registry_summary
import json
print(json.dumps(get_registry_summary(), indent=2))
"
```

---

## rename_log.txt

**Purpose:** Human-readable audit trail of all rename operations.

**Format:**
```
============================================================
Rename session: 2026-05-03 01:34:22
============================================================
  [ 63] DeepLearning_Lecture8.pdf -> source_63.pdf [RENAMED] [MOVED TO TESTED] [REGISTRY UPDATED]
  [ 64] DeepLearning_Lecture9.pdf -> source_64.pdf [RENAMED] [MOVED TO TESTED] [REGISTRY UPDATED]
```

Useful for:
- Auditing what was renamed when
- Recovering original filenames if needed
- Debugging naming conflicts

---

## Complete Document Flow

```
Publisher / Educator
        ↓
  data/raw/new_textbook.pdf
        ↓
  python scripts/rename_docs.py
        ↓
  [validation check]
  Pass -> data/tested/source_N.pdf
  Fail -> data/rejected/new_textbook.pdf
        ↓
  GET /admin/new-documents
  (shows source_N.pdf as unregistered)
        ↓
  POST /admin/upload-document?domain=database
  (dedup check → register in document_registry.json if new)
  ↓
  POST /admin/rebuild-domain {"domain": "database"}
  (incremental graph update ~2-5 mins; pre-rebuild dedup filter)
  (marks as indexed in registry)
        ↓
  Knowledge graph updated
  New content searchable immediately
```

---

## Domain to Source File Mapping

| Domain | Source Files |
|--------|-------------|
| deep_learning | source_50 to source_64 (15 files) |
| database | source_7, 8, 13, 14, 17, 18, 31, 36, 49 (9 files) |
| software_engineering | source_1, 2, 3, 4, 11, 15, 16, 21, 22, 37, 45, 46, 47, 48 (14 files) |
| operating_systems | source_5, 10, 19, 20, 33, 35, 39, 40, 41, 42, 43 (11 files) |
| computer_architecture | source_6, 9, 12, 23, 24, 25, 26, 27, 28, 29, 30, 32, 38, 44 (14 files) |

> Full mapping with indexed status always available in `document_registry.json`.

---

## Gitignore Note

The following data/ contents should NOT be committed to GitHub:

```gitignore
# Add to .gitignore:
data/raw/*
data/rejected/*
data/tested/*
data/uploads/*
!data/raw/.gitkeep
!data/rejected/.gitkeep
!data/tested/.gitkeep
!data/uploads/.gitkeep
```

Keep `document_registry.json` and `rename_log.txt` committed : they are small and essential for reproducibility.

---

## Related

- [../scripts/README.md](../scripts/README.md) : rename_docs.py, validate_docs.py
- [../services/README.md](../services/README.md) : admin upload/rebuild handlers
- [../docs/deduplication.md](../docs/deduplication.md) : dedup rules and config
- [../utils/README.md](../utils/README.md) : registry.py, document_loader.py
- [../graphrag/README.md](../graphrag/README.md) : how PDFs become knowledge graphs
- [../README.md](../README.md) : Main project README
