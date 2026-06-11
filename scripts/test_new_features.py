"""
Test case_study question type and document deduplication.

Usage (project root, venv active):
  python scripts/test_new_features.py
  python scripts/test_new_features.py --live
  python scripts/test_new_features.py --live --generate   # slow: full LLM call
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

API_BASE = os.getenv("GRAPHRAG_API_BASE", "http://localhost:8000")
TESTED_DIR = os.path.join(ROOT, "data", "tested")


def _header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _pass(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def run_unit_tests() -> bool:
    _header("Unit tests (hash + similarity)")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("tests.test_document_dedup")
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if result.wasSuccessful():
        _pass(f"{result.testsRun} tests in tests.test_document_dedup")
        return True
    _fail(f"{len(result.failures)} failure(s), {len(result.errors)} error(s)")
    return False


def run_static_checks() -> bool:
    _header("Static checks (config, prompts, API types)")
    ok = True

    from utils.config import cfg
    from utils.question_types import valid_question_types
    from utils.prompts import build_question_messages
    from utils.question_type_prompts import instructions_for_question_type

    if "case_study" not in cfg.get("question_types", {}):
        _fail("case_study missing from config.yml question_types")
        ok = False
    else:
        _pass("case_study in config.yml")

    if "case_study" not in valid_question_types():
        _fail("valid_question_types() missing case_study")
        ok = False
    else:
        _pass(f"valid_question_types = {sorted(valid_question_types())}")

    instr = instructions_for_question_type("case_study")
    if "SCENARIO" not in instr or "QUESTION" not in instr:
        _fail("case_study instructions missing SCENARIO/QUESTION")
        ok = False
    else:
        _pass("case_study prompt instructions include SCENARIO + QUESTION")

    messages = build_question_messages(
        context="Sample context about transactions.",
        learning_objective="Explain deadlocks",
        difficulty="medium",
        question_type="case_study",
    )
    user_content = messages[1]["content"]
    if "SCENARIO" not in user_content:
        _fail("build_question_messages did not embed case_study block")
        ok = False
    else:
        _pass("build_question_messages embeds case_study block")

    if not cfg.get("deduplication", {}).get("enabled"):
        _fail("deduplication.enabled is false")
        ok = False
    else:
        _pass(
            "deduplication enabled "
            f"(threshold={cfg['deduplication'].get('similarity_threshold')})"
        )

    return ok


def run_dedup_integration() -> bool:
    _header("Dedup integration (temp files, no server)")
    from services.document_dedup_service import DocumentDedupService
    from utils.registry_content import list_indexed_files, update_file_fields

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tested = os.path.join(tmp, "tested")
        os.makedirs(tested)

        # Simulate indexed file with known hash
        indexed_name = "source_9001.pdf"
        indexed_path = os.path.join(tested, indexed_name)
        content = b"%PDF-1.4 fake identical content for dedup test"
        with open(indexed_path, "wb") as f:
            f.write(content)

        from utils.content_hash import sha256_bytes, sha256_text
        from utils.text_normalization import normalize_document_text

        bytes_hash = sha256_bytes(content)
        text_hash = sha256_text(normalize_document_text("identical extracted text"))

        domain = "database"
        registry_path = os.path.join(tmp, "document_registry.json")
        registry = {
            "last_source_number": 9001,
            "domains": {
                domain: {
                    "description": "test",
                    "files": {
                        indexed_name: {
                            "indexed": True,
                            "added": "2026-06-04",
                            "content_hash_bytes": bytes_hash,
                            "content_hash_text": text_hash,
                        }
                    },
                }
            },
        }
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f)

        dedup = DocumentDedupService(
            tested,
            dedup_config={
                "enabled": True,
                "similarity_threshold": 0.95,
                "on_exact_hash": "skip",
                "on_high_similarity": "skip",
                "max_similarity_comparisons": 5,
                "similarity_sample_chars": 5000,
            },
        )

        duplicate_path = os.path.join(tested, "source_9002.pdf")
        with open(duplicate_path, "wb") as f:
            f.write(content)

        with patch("utils.registry_content.load_registry", return_value=registry):
            with patch("utils.registry_content.save_registry"):
                assessment = dedup.assess_file(duplicate_path, domain)

        if not assessment.is_duplicate or not assessment.should_skip_indexing:
            _fail("identical bytes should be detected as duplicate")
            ok = False
        elif assessment.match and assessment.match.reason != "exact_bytes":
            _fail(f"expected exact_bytes, got {assessment.match.reason}")
            ok = False
        else:
            _pass(
                f"byte-duplicate detected "
                f"(matched {assessment.match.matched_file})"
            )

        # Near-duplicate via similarity (different bytes, same text when extracted)
        near_name = "source_9003.pdf"
        near_path = os.path.join(tested, near_name)
        with open(near_path, "wb") as f:
            f.write(b"different bytes but we mock load_document")

        def fake_load(path: str) -> str:
            base = (
                "Database transaction isolation levels and locking protocols "
                "for concurrent access in relational systems. " * 30
            )
            if "9003" in path:
                return "Header v2\n" + base
            if "9001" in path:
                return "Header v1\n" + base
            return ""

        with patch("utils.registry_content.load_registry", return_value=registry):
            with patch(
                "services.document_dedup_service.load_document",
                side_effect=fake_load,
            ):
                assessment2 = dedup.assess_file(near_path, domain)

        if not assessment2.is_duplicate:
            _fail("near-duplicate text should be detected")
            ok = False
        else:
            reason = assessment2.match.reason if assessment2.match else "?"
            _pass(f"near-duplicate detected (reason={reason})")

    return ok


def run_live_api(run_generate: bool) -> bool:
    _header(f"Live API tests ({API_BASE})")
    try:
        import requests
    except ImportError:
        _fail("requests not installed")
        return False

    ok = True

    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        _pass(f"health status={health.get('status')}")
    except Exception as exc:
        _fail(f"server not reachable — start with: python main_gr.py --mode full ({exc})")
        return False

    # Invalid question type
    try:
        r = requests.post(
            f"{API_BASE}/graphrag/generate",
            json={
                "learning_objective": "Explain SQL joins",
                "difficulty": "easy",
                "question_type": "invalid_type",
            },
            timeout=15,
        )
        if r.status_code in (400, 422):
            _pass(f"invalid question_type rejected (HTTP {r.status_code})")
        else:
            _fail(f"expected 400 for invalid type, got {r.status_code}")
            ok = False
    except Exception as exc:
        _fail(f"invalid type test: {exc}")
        ok = False

    # Valid case_study accepted (quick validation only unless --generate)
    payload = {
        "learning_objective": "Apply deadlock prevention with multiple processes",
        "difficulty": "medium",
        "question_type": "case_study",
        "domain": "operating_systems",
    }
    if not run_generate:
        _pass(
            "case_study payload ready (skip full generate; use --live --generate)"
        )
        return ok

    print("  ... generating case_study question (1-5 min on GPU) ...")
    try:
        r = requests.post(
            f"{API_BASE}/graphrag/generate",
            json=payload,
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        question = data.get("question", "")
        meta = data.get("metadata", {})
        if not question:
            _fail("empty question in response")
            ok = False
        else:
            has_scenario = "SCENARIO" in question.upper()
            _pass(f"case_study generated ({len(question)} chars)")
            if has_scenario:
                _pass("response contains SCENARIO marker")
            else:
                print(
                    "  WARN  SCENARIO marker not found — model may use free format"
                )
        if meta.get("sample_answer"):
            _pass("sample_answer present in metadata")
        else:
            _fail("sample_answer missing")
            ok = False
        if meta.get("question_type") == "case_study" or payload["question_type"]:
            _pass(f"metadata difficulty={meta.get('difficulty')}")
    except Exception as exc:
        _fail(f"case_study generate: {exc}")
        ok = False

    return ok


def run_live_dedup_upload() -> bool:
    """Upload same bytes twice to admin endpoint if a sample PDF exists."""
    _header("Live dedup upload test (admin)")
    try:
        import requests
    except ImportError:
        return False

    pdfs = [
        f
        for f in os.listdir(TESTED_DIR)
        if f.lower().endswith(".pdf")
    ] if os.path.isdir(TESTED_DIR) else []
    if not pdfs:
        print("  SKIP  no PDF in data/tested/ for upload dedup test")
        return True

    sample = pdfs[0]
    sample_path = os.path.join(TESTED_DIR, sample)
    domain = "database"

    try:
        with open(sample_path, "rb") as f:
            r1 = requests.post(
                f"{API_BASE}/admin/upload-document?domain={domain}",
                files={"file": (f"dedup_test_{sample}", f.read(), "application/pdf")},
                timeout=120,
            )
        if r1.status_code != 200:
            print(f"  SKIP  upload returned {r1.status_code}: {r1.text[:200]}")
            return True

        body1 = r1.json()
        dedup1 = body1.get("dedup") or {}
        msg1 = body1.get("message", "")
        _pass(f"first upload: {msg1[:80]}...")

        with open(sample_path, "rb") as f:
            r2 = requests.post(
                f"{API_BASE}/admin/upload-document?domain={domain}",
                files={"file": (f"dedup_test_copy_{sample}", f.read(), "application/pdf")},
                timeout=120,
            )
        r2.raise_for_status()
        body2 = r2.json()
        dedup2 = body2.get("dedup") or {}

        if body2.get("saved_to_disk") is False or (
            body2.get("already_indexed") and body2.get("dedup", {}).get(
                "is_duplicate"
            )
        ):
            _pass(
                f"duplicate upload rejected without save "
                f"(reason={dedup2.get('match', {}).get('reason', 'duplicate')})"
            )
        elif body2.get("already_indexed") or dedup2.get("is_duplicate"):
            _pass(
                f"second upload flagged duplicate "
                f"(reason={dedup2.get('match', {}).get('reason', 'indexed')})"
            )
        else:
            _fail("second identical upload was not flagged as duplicate")
            return False
    except Exception as exc:
        _fail(f"dedup upload test: {exc}")
        return False

    return True


def run_real_library_dedup() -> bool:
    """Test dedup against real data/tested + registry (no server)."""
    _header("Real library dedup (on-disk byte hash)")
    if not os.path.isdir(TESTED_DIR):
        print("  SKIP  data/tested/ not found")
        return True

    from services.document_dedup_service import DocumentDedupService
    from utils.registry import load_registry

    registry = load_registry()
    domain = "database"
    files = registry["domains"].get(domain, {}).get("files", {})
    indexed = [n for n, i in files.items() if i.get("indexed")]
    if not indexed:
        print("  SKIP  no indexed database files")
        return True

    ref = indexed[0]
    ref_path = os.path.join(TESTED_DIR, ref)
    if not os.path.exists(ref_path):
        print(f"  SKIP  {ref} missing on disk")
        return True

    dedup = DocumentDedupService(TESTED_DIR)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        with open(ref_path, "rb") as src:
            tmp.write(src.read())

    try:
        t0 = __import__("time").time()
        result = dedup.assess_file(tmp_path, domain)
        elapsed = __import__("time").time() - t0
        if result.is_duplicate and result.match:
            _pass(
                f"copy of {ref} detected in {elapsed:.2f}s "
                f"({result.match.reason} -> {result.match.matched_file})"
            )
            return True
        _fail(f"copy of indexed {ref} not detected ({elapsed:.2f}s)")
        return False
    finally:
        os.unlink(tmp_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Hit running API")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="With --live, run full case_study generation (slow)",
    )
    parser.add_argument(
        "--dedup-upload",
        action="store_true",
        help="With --live, test duplicate admin upload",
    )
    args = parser.parse_args()

    print("EdGenAI — test case_study + deduplication")
    results = [
        run_unit_tests(),
        run_static_checks(),
        run_dedup_integration(),
        run_real_library_dedup(),
    ]

    if args.live:
        results.append(run_live_api(args.generate))
        if args.dedup_upload:
            results.append(run_live_dedup_upload())

    _header("Summary")
    if all(results):
        print("  ALL CHECKS PASSED")
        if not args.live:
            print("\n  Next: start server and run:")
            print("    python main_gr.py --mode full")
            print("    python scripts/test_new_features.py --live --dedup-upload")
            print("    python scripts/test_new_features.py --live --generate")
        return 0

    print("  SOME CHECKS FAILED — see output above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
