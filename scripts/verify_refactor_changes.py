"""
Verify Phases 1-5 refactor changes.

Usage (from project root, with venv active):
  python scripts/verify_refactor_changes.py
  python scripts/verify_refactor_changes.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

API_BASE = os.getenv("GRAPHRAG_API_BASE", "http://localhost:8000")


def _ok(name: str, detail: str = "") -> bool:
    suffix = f" — {detail}" if detail else ""
    print(f"  PASS  {name}{suffix}")
    return True


def _fail(name: str, detail: str = "") -> bool:
    suffix = f" — {detail}" if detail else ""
    print(f"  FAIL  {name}{suffix}")
    return False


def check_static() -> bool:
    print("\n=== Static checks ===")
    passed = True

    # Compatibility wrappers
    for path in ("main.py", "main_multi.py"):
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            passed = _fail(f"{path} exists") and passed
            continue
        text = open(full, encoding="utf-8").read()
        if "from main_gr import app" in text and "Deprecated" in text:
            passed = _ok(f"{path} compatibility wrapper") and passed
        else:
            passed = _fail(f"{path} compatibility wrapper") and passed

    # No runtime config.yml writes in main_gr admin path
    main_gr = open(os.path.join(ROOT, "main_gr.py"), encoding="utf-8").read()
    if "yaml.dump" in main_gr:
        passed = _fail("main_gr.py avoids config.yml runtime writes", "yaml.dump still present") and passed
    else:
        passed = _ok("main_gr.py avoids config.yml runtime writes") and passed

    # Service modules exist
    for rel in (
        "services/domain_catalog_service.py",
        "services/graphrag_generation_service.py",
        "services/status_service.py",
        "services/admin_domain_service.py",
        "services/index_repository.py",
        "utils/atomic_io.py",
    ):
        exists = os.path.exists(os.path.join(ROOT, rel))
        passed = (_ok(rel) if exists else _fail(rel, "missing")) and passed

    # Atomic JSON write
    try:
        from utils.atomic_io import atomic_write_json

        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(tmp)
        atomic_write_json(tmp, {"verify": True})
        with open(tmp, encoding="utf-8") as f:
            data = json.load(f)
        os.remove(tmp)
        if data.get("verify") is True:
            passed = _ok("atomic_write_json round-trip") and passed
        else:
            passed = _fail("atomic_write_json round-trip") and passed
    except Exception as e:
        passed = _fail("atomic_write_json round-trip", str(e)) and passed

    return passed


def check_imports() -> bool:
    print("\n=== Import / wiring checks ===")
    passed = True
    try:
        import main_gr

        domains = main_gr._sync_runtime_domains()
        if not domains:
            passed = _fail("domain sync", "no domains returned") and passed
        else:
            passed = _ok("domain sync", f"{len(domains)} domains") and passed

        for attr in (
            "domain_catalog_service",
            "graphrag_generation_service",
            "status_service",
            "admin_domain_service",
            "index_repository",
            "app",
        ):
            if hasattr(main_gr, attr):
                passed = _ok(f"main_gr.{attr} wired") and passed
            else:
                passed = _fail(f"main_gr.{attr} wired") and passed

        health = main_gr.status_service.build_health(domains)
        if health.get("status") == "ok" and "graphrag" in health:
            passed = _ok("StatusService.build_health") and passed
        else:
            passed = _fail("StatusService.build_health") and passed

    except Exception as e:
        passed = _fail("import main_gr", str(e)) and passed
    return passed


def check_live() -> bool:
    print("\n=== Live API checks ===")
    passed = True
    try:
        import requests
    except ImportError:
        return _fail("requests installed", "pip install requests")

    def get(path: str):
        r = requests.get(f"{API_BASE}{path}", timeout=15)
        r.raise_for_status()
        return r.json()

    def post(path: str, payload: dict):
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    try:
        health = get("/health")
        if health.get("status") == "ok":
            passed = _ok("GET /health") and passed
        else:
            passed = _fail("GET /health", str(health)) and passed

        status = get("/graphrag/status")
        if "domains" in status:
            passed = _ok("GET /graphrag/status", f"{len(status['domains'])} domains") and passed
        else:
            passed = _fail("GET /graphrag/status") and passed

        gen = post(
            "/graphrag/generate",
            {
                "learning_objective": "Explain SQL joins and their use in relational databases",
                "difficulty": "medium",
                "question_type": "conceptual",
                "domain": "database",
            },
        )
        if gen.get("question") and gen.get("metadata"):
            passed = _ok("POST /graphrag/generate (valid domain)") and passed
        else:
            passed = _fail("POST /graphrag/generate (valid domain)") and passed

        gen_fallback = post(
            "/graphrag/generate",
            {
                "learning_objective": "Explain SQL joins and their use in relational databases",
                "difficulty": "medium",
                "question_type": "conceptual",
                "domain": "invalid_domain_xyz",
            },
        )
        if gen_fallback.get("question"):
            passed = _ok("POST /graphrag/generate (invalid domain fallback)") and passed
        else:
            passed = _fail("POST /graphrag/generate (invalid domain fallback)") and passed

        rebuild = post("/admin/rebuild-domain", {"domain": "database"})
        if rebuild.get("status") == "ok" and "elapsed_seconds" in rebuild:
            passed = _ok("POST /admin/rebuild-domain", f"elapsed={rebuild['elapsed_seconds']}s") and passed
        else:
            passed = _fail("POST /admin/rebuild-domain", "missing elapsed_seconds or status") and passed

    except Exception as e:
        passed = _fail("live API", str(e)) and passed
        print("  Hint: start server first -> python main_gr.py --mode full")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify refactor phases 1-5")
    parser.add_argument("--live", action="store_true", help="Also hit running API endpoints")
    args = parser.parse_args()

    print("P000233DS refactor verification")
    print(f"Project root: {ROOT}")

    all_pass = check_static()
    all_pass = check_imports() and all_pass
    if args.live:
        all_pass = check_live() and all_pass

    print("\n=== Summary ===")
    if all_pass:
        print("All requested checks passed.")
        return 0
    print("Some checks failed. Review output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
