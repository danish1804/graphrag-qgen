# ============================================================
# dataset/inputs/generate_input_csv.py
#
# Generates input_questions.csv — the master list of all
# question generation jobs to run.
#
# Usage:
#   python dataset/inputs/generate_input_csv.py
#
# Output:
#   dataset/inputs/input_questions.csv
#
# The CSV is then consumed by dataset/generate_dataset.py
# which calls the GraphRAG API for each row.
#
# Author: Mohammed Danish Alam
# Project: P000233DS — EdGenAI Technologies Capstone
# ============================================================

import csv
import os
import sys
import itertools
from datetime import datetime

# ── Add project root to path ──────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from topics import (
    DOMAIN_TOPICS,
    CROSS_DOMAIN_TOPICS,
    HARD_MULTI_DOMAIN_TOPICS,
    CASE_STUDY_TOPICS,
)

# ── Output path ───────────────────────────────────────────
OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "input_questions.csv"
)

# ── CSV columns ───────────────────────────────────────────
COLUMNS = [
    "q_id",
    "category",
    "domain",
    "difficulty",
    "question_type",
    "learning_objective",
    "routing_method",
    "top_k",
    "seed_ratio",
    "status",           # pending / done / failed
    "generated_at",     # filled after generation
]

# ── RAG params by difficulty ──────────────────────────────
RAG_PARAMS = {
    "easy":       {"top_k": 3, "seed_ratio": 0.8},
    "medium":     {"top_k": 5, "seed_ratio": 0.6},
    "hard":       {"top_k": 7, "seed_ratio": 0.4},
    "case_study": {"top_k": 5, "seed_ratio": 0.6},
}


def make_id(prefix: str, counter: int) -> str:
    return f"{prefix}-{counter:03d}"


def build_rows() -> list:
    rows = []
    counter = 1

    # ── 1. EASY — single domain, all topics ───────────────
    for domain, topics in DOMAIN_TOPICS.items():
        domain_code = domain[:2].upper()
        for topic in topics:
            rows.append({
                "q_id":               make_id(f"E-{domain_code}", counter),
                "category":           f"easy_{domain}",
                "domain":             domain,
                "difficulty":         "easy",
                "question_type":      "recall",
                "learning_objective": topic,
                "routing_method":     "explicit",
                "top_k":              RAG_PARAMS["easy"]["top_k"],
                "seed_ratio":         RAG_PARAMS["easy"]["seed_ratio"],
                "status":             "pending",
                "generated_at":       "",
            })
            counter += 1

    # ── 2. MEDIUM — 2-domain combos, cross-domain topics ──
    domain_names = list(DOMAIN_TOPICS.keys())
    for d1, d2 in itertools.combinations(domain_names, 2):
        key   = (d1, d2)
        topics = CROSS_DOMAIN_TOPICS.get(key, [])

        # If no specific cross-domain topics defined
        # use first 3 topics from each domain
        if not topics:
            topics = (
                DOMAIN_TOPICS[d1][:2] +
                DOMAIN_TOPICS[d2][:2]
            )

        # Take up to 15 topics per combo
        topics = topics[:15]

        combo_code = f"{d1[:2].upper()}{d2[:2].upper()}"
        for topic in topics:
            rows.append({
                "q_id":               make_id(f"M-{combo_code}", counter),
                "category":           f"medium_{d1}_{d2}",
                "domain":             f"{d1}+{d2}",
                "difficulty":         "medium",
                "question_type":      "conceptual",
                "learning_objective": topic,
                "routing_method":     "auto",
                "top_k":              RAG_PARAMS["medium"]["top_k"],
                "seed_ratio":         RAG_PARAMS["medium"]["seed_ratio"],
                "status":             "pending",
                "generated_at":       "",
            })
            counter += 1

    # ── 3. HARD — multi-domain auto route ─────────────────
    for topic in HARD_MULTI_DOMAIN_TOPICS:
        rows.append({
            "q_id":               make_id("H-MD", counter),
            "category":           "hard_multi_domain",
            "domain":             "auto",
            "difficulty":         "hard",
            "question_type":      "application",
            "learning_objective": topic,
            "routing_method":     "auto",
            "top_k":              RAG_PARAMS["hard"]["top_k"],
            "seed_ratio":         RAG_PARAMS["hard"]["seed_ratio"],
            "status":             "pending",
            "generated_at":       "",
        })
        counter += 1

    # ── 4. CASE STUDY — all domains ───────────────────────
    for topic in CASE_STUDY_TOPICS:
        rows.append({
            "q_id":               make_id("CS", counter),
            "category":           "case_study",
            "domain":             "auto",
            "difficulty":         "case_study",
            "question_type":      "case_study",
            "learning_objective": topic,
            "routing_method":     "auto",
            "top_k":              RAG_PARAMS["case_study"]["top_k"],
            "seed_ratio":         RAG_PARAMS["case_study"]["seed_ratio"],
            "status":             "pending",
            "generated_at":       "",
        })
        counter += 1

    return rows


def check_duplicates(rows: list) -> list:
    """Flag duplicate learning objectives."""
    seen = {}
    dupes = []
    for row in rows:
        obj = row["learning_objective"].strip().lower()
        if obj in seen:
            dupes.append((row["q_id"], seen[obj], obj))
        else:
            seen[obj] = row["q_id"]
    return dupes


def save_csv(rows: list):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] Saved: {OUTPUT_CSV}")


def print_summary(rows: list):
    from collections import Counter
    cats = Counter(r["category"].split("_")[0] for r in rows)
    domains = Counter(r["domain"] for r in rows)

    print("\n" + "=" * 55)
    print("INPUT CSV SUMMARY")
    print("=" * 55)
    print(f"  Total rows:    {len(rows)}")
    print(f"\n  By difficulty:")
    for cat, count in sorted(cats.items()):
        print(f"    {cat:<15}: {count}")
    print(f"\n  Unique domains: {len(domains)}")
    print("=" * 55)


if __name__ == "__main__":
    print("[CSV Generator] Building input question list...")

    rows = build_rows()

    # Check for duplicates
    dupes = check_duplicates(rows)
    if dupes:
        print(f"\n[Warning] {len(dupes)} duplicate objectives found:")
        for q_id, orig_id, obj in dupes:
            print(f"  {q_id} duplicates {orig_id}: '{obj[:60]}'")
    else:
        print("[CSV Generator] No duplicate objectives found ✅")

    # Save
    save_csv(rows)
    print_summary(rows)

    print(f"\n[CSV Generator] Done. Edit input_questions.csv")
    print(f"                to adjust objectives before running")
    print(f"                generate_dataset.py")