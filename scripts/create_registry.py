# scripts/create_registry.py
import json
import os
import sys
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from graphrag.graphrag_domain_indexer import DOMAIN_MAP
from datetime import date

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(
    BASE_DIR, "data", "document_registry.json")

def create_registry():
    domains = {}
    last_number = 0

    for domain, info in DOMAIN_MAP.items():
        files = {}
        for filename in info["files"]:
            files[filename] = {
                "indexed": True,
                "added":   str(date.today())
            }
            try:
                num = int(
                    filename.replace("source_", "")
                    .replace(".pdf", "")
                    .replace(".pptx", "")
                )
                last_number = max(last_number, num)
            except ValueError:
                pass

        domains[domain] = {
            "description": info.get(
                "description", domain),
            "files": files
        }

    registry = {
        "last_source_number": last_number,
        "domains": domains
    }

    with open(REGISTRY_PATH, "w",
              encoding="utf-8") as f:
        json.dump(registry, f, indent=2,
                  ensure_ascii=False)

    total = sum(
        len(d["files"]) for d in domains.values())
    print(f"Registry created: {REGISTRY_PATH}")
    print(f"Total documents:  {total}")
    print(f"Last number:      {last_number}")
    for domain, info in domains.items():
        print(f"  {domain}: "
              f"{len(info['files'])} files")

if __name__ == "__main__":
    create_registry()