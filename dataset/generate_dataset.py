# ============================================================
# dataset/generate_dataset.py
#
# Reads input_questions.csv and calls the GraphRAG API
# for each row. Saves results to Excel progressively
# (saves after every batch so no data is lost on crash).
#
# Usage:
#   # Full run — all pending rows:
#   python dataset/generate_dataset.py
#
#   # Run specific batch size:
#   python dataset/generate_dataset.py --batch 15
#
#   # Run specific category only:
#   python dataset/generate_dataset.py --category easy_deep_learning
#
#   # Dry run — show what would be generated:
#   python dataset/generate_dataset.py --dry-run
#
# Author: Mohammed Danish Alam
# Project: P000233DS — EdGenAI Technologies Capstone
# ============================================================

import os
import sys
import csv
import json
import time
import argparse
import requests
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter

# ── Paths ─────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(_DIR, "inputs", "input_questions.csv")
OUTPUT_XLSX = os.path.join(_DIR, "outputs",
                           "P000233DS_Question_Dataset.xlsx")
TEMPLATE    = os.path.join(_DIR, "outputs",
                           "P000233DS_Question_Dataset_Template.xlsx")

# ── API config ────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
TIMEOUT     = 300       # seconds per request
RETRY_LIMIT = 2         # retries on failure

# ── Styling ───────────────────────────────────────────────
HDR_FILL  = PatternFill("solid", start_color="1F4E79")
ALT_FILL  = PatternFill("solid", start_color="EBF3FB")
PASS_FILL = PatternFill("solid", start_color="E2EFDA")
FAIL_FILL = PatternFill("solid", start_color="FCE4D6")
BORD_FILL = PatternFill("solid", start_color="FFF2CC")
WHT_FILL  = PatternFill("solid", start_color="FFFFFF")
HDR_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
BODY_FONT = Font(name="Arial", size=10)
WRAP      = Alignment(wrap_text=True, vertical="top")
CENTER    = Alignment(horizontal="center", vertical="center",
                      wrap_text=True)
thin      = Side(style="thin", color="CCCCCC")
BORDER    = Border(left=thin, right=thin, top=thin, bottom=thin)

# ── Excel columns ─────────────────────────────────────────
EXCEL_HEADERS = [
    # Input parameters
    ("Q_ID",                6),
    ("Domain",              18),
    ("Difficulty",          12),
    ("Question Type",       16),
    # RAG config
    ("Learning Objective",  35),
    ("Routing Method",      16),
    ("Top K",               8),
    ("Seed Ratio",          10),
    # Generated output
    ("Generated Question",  50),
    ("Sample Answer",       50),
    ("Sources Used",        25),
    # Judge scores
    ("Q Score (/5)",        12),
    ("A Score (/5)",        12),
    ("R Score (/5)",        12),
    ("Attempts",            10),
    # Human review
    ("Human Rating",        14),
    ("Reviewed By",         14),
    # Remarks
    ("Remarks",             35),
    # Meta
    ("Generated At",        18),
    ("Status",              12),
]


# ════════════════════════════════════════════════════════════
# API
# ════════════════════════════════════════════════════════════

def call_graphrag_api(row: dict) -> dict:
    """Call POST /graphrag/generate and return parsed result."""

    domain  = row["domain"]
    routing = row["routing_method"].lower()
    diff    = row["difficulty"]

    # Build payload
    payload = {
        "learning_objective": row["learning_objective"],
        "difficulty":         diff if diff != "case_study" else "medium",
        "question_type":      row["question_type"],
        "top_k_nodes":        int(row["top_k"]),
        "neighbour_depth":    1,
    }

    # Add domain only if explicit routing and valid domain
    if routing == "explicit" and "+" not in domain and \
            domain not in ("auto", ""):
        payload["domain"] = domain

    for attempt in range(RETRY_LIMIT + 1):
        try:
            resp = requests.post(
                f"{API_BASE}/graphrag/generate",
                json=payload,
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            print(f"    [API] Timeout (attempt {attempt+1})")
            if attempt == RETRY_LIMIT:
                return {"error": "timeout"}
            time.sleep(5)

        except requests.exceptions.HTTPError as e:
            print(f"    [API] HTTP error: {e}")
            return {"error": str(e)}

        except Exception as e:
            print(f"    [API] Error: {e}")
            return {"error": str(e)}


def parse_response(response: dict) -> dict:
    """Extract fields from API response."""

    if "error" in response:
        return {
            "question":    "",
            "answer":      "",
            "sources":     "",
            "q_score":     "",
            "a_score":     "",
            "r_score":     "",
            "attempts":    "",
            "status":      f"FAILED: {response['error']}",
        }

    meta    = response.get("metadata", {})
    judge   = meta.get("judge_scores", {}) or {}
    sources = meta.get("sources", [])

    return {
        "question": response.get("question", ""),
        "answer":   meta.get("sample_answer", ""),
        "sources":  ", ".join(sources) if sources else "",
        "q_score":  judge.get("question_overall", ""),
        "a_score":  judge.get("answer_overall", ""),
        "r_score":  judge.get("retrieval_overall", ""),
        "attempts": judge.get("attempts", 1),
        "status":   "done",
    }


# ════════════════════════════════════════════════════════════
# Excel
# ════════════════════════════════════════════════════════════

def init_excel() -> openpyxl.Workbook:
    """Create or load the output Excel workbook."""

    os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)

    if os.path.exists(OUTPUT_XLSX):
        print(f"[Excel] Loading existing: {OUTPUT_XLSX}")
        return openpyxl.load_workbook(OUTPUT_XLSX)

    print(f"[Excel] Creating new workbook: {OUTPUT_XLSX}")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Question Dataset"
    ws.freeze_panes = "A3"

    # Row 1 — group headers
    groups = [
        (1,  4,  "INPUT PARAMETERS"),
        (5,  8,  "RAG CONFIGURATION"),
        (9,  11, "GENERATED OUTPUT"),
        (12, 15, "LLM JUDGE SCORES"),
        (16, 17, "HUMAN REVIEW"),
        (18, 18, "REMARKS"),
        (19, 20, "META"),
    ]
    for start, end, label in groups:
        ws.merge_cells(
            start_row=1, start_column=start,
            end_row=1,   end_column=end)
        c = ws.cell(row=1, column=start, value=label)
        c.font      = HDR_FONT
        c.fill      = HDR_FILL
        c.alignment = CENTER
        c.border    = BORDER

    # Row 2 — column headers
    for col, (name, width) in enumerate(EXCEL_HEADERS, 1):
        c = ws.cell(row=2, column=col, value=name)
        c.font      = HDR_FONT
        c.fill      = HDR_FILL
        c.alignment = CENTER
        c.border    = BORDER
        ws.column_dimensions[
            get_column_letter(col)].width = width

    ws.row_dimensions[1].height = 20
    ws.row_dimensions[2].height = 35
    wb.save(OUTPUT_XLSX)
    return wb


def get_existing_ids(wb: openpyxl.Workbook) -> set:
    """Return set of Q_IDs already in the workbook."""
    ws   = wb.active
    ids  = set()
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[0]:
            ids.add(str(row[0]))
    return ids


def append_row(wb: openpyxl.Workbook,
               input_row: dict,
               result: dict):
    """Append one result row to the Excel workbook."""
    ws        = wb.active
    next_row  = ws.max_row + 1
    fill      = ALT_FILL if next_row % 2 == 0 else WHT_FILL

    values = [
        input_row["q_id"],
        input_row["domain"],
        input_row["difficulty"],
        input_row["question_type"],
        input_row["learning_objective"],
        input_row["routing_method"],
        input_row["top_k"],
        input_row["seed_ratio"],
        result["question"],
        result["answer"],
        result["sources"],
        result["q_score"],
        result["a_score"],
        result["r_score"],
        result["attempts"],
        "",                             # Human Rating
        "",                             # Reviewed By
        "",                             # Remarks
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        result["status"],
    ]

    for col, val in enumerate(values, 1):
        c           = ws.cell(row=next_row, column=col, value=val)
        c.font      = BODY_FONT
        c.fill      = fill
        c.alignment = WRAP
        c.border    = BORDER

    ws.row_dimensions[next_row].height = 80


def save_excel(wb: openpyxl.Workbook):
    wb.save(OUTPUT_XLSX)
    print(f"    [Excel] Saved: {OUTPUT_XLSX}")


# ════════════════════════════════════════════════════════════
# CSV
# ════════════════════════════════════════════════════════════

def load_csv(category: str = None) -> list:
    """Load input CSV, optionally filtered by category."""
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] Input CSV not found: {INPUT_CSV}")
        print("        Run generate_input_csv.py first.")
        sys.exit(1)

    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "pending":
                if category is None or \
                        row["category"] == category:
                    rows.append(row)

    return rows


def mark_done_in_csv(q_id: str, status: str = "done"):
    """Update status of a row in the input CSV."""
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["q_id"] == q_id:
                row["status"]       = status
                row["generated_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M")
            rows.append(row)

    with open(INPUT_CSV, "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def run(batch_size: int = 15,
        category: str = None,
        dry_run: bool = False):

    print("\n" + "=" * 55)
    print("P000233DS — Question Dataset Generator")
    print("=" * 55)

    # Load pending rows
    pending = load_csv(category)
    print(f"[Generator] Pending rows: {len(pending)}")
    if category:
        print(f"[Generator] Category filter: {category}")

    if not pending:
        print("[Generator] Nothing to generate. All done!")
        return

    if dry_run:
        print("\n[DRY RUN] Would generate:")
        for row in pending[:batch_size]:
            print(f"  {row['q_id']:15} | {row['domain']:30} | "
                  f"{row['difficulty']:10} | "
                  f"{row['learning_objective'][:40]}")
        print(f"\n  ... and {max(0, len(pending)-batch_size)} "
              f"more rows")
        return

    # Check API is running
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=10)
        resp.raise_for_status()
        print("[Generator] API is running Pass")
    except Exception:
        print("[Generator] Failed to connect to API.")
        print("            Start: python main_gr.py --mode full")
        sys.exit(1)

    # Init Excel
    wb          = init_excel()
    existing_ids = get_existing_ids(wb)
    print(f"[Generator] Already in Excel: {len(existing_ids)}")

    # Generate
    to_process = pending[:batch_size]
    success    = 0
    failed     = 0

    print(f"\n[Generator] Generating {len(to_process)} questions...")
    print("-" * 55)

    for idx, row in enumerate(to_process, 1):
        q_id = row["q_id"]

        # Skip already in Excel
        if q_id in existing_ids:
            print(f"  [{idx:3}/{len(to_process)}] "
                  f"SKIP (already exists): {q_id}")
            mark_done_in_csv(q_id, "done")
            continue

        print(f"  [{idx:3}/{len(to_process)}] {q_id} — "
              f"{row['learning_objective'][:45]}...")

        start_time = time.time()
        response   = call_graphrag_api(row)
        elapsed    = time.time() - start_time
        result     = parse_response(response)

        status = "Passed" if result["status"] == "done" else "Failed"
        print(f"    {status} {elapsed:.0f}s | "
              f"Q={result['q_score']} "
              f"A={result['a_score']} "
              f"R={result['r_score']}")

        # Append to Excel
        append_row(wb, row, result)

        # Mark in CSV
        mark_done_in_csv(
            q_id,
            "done" if result["status"] == "done" else "failed"
        )

        if result["status"] == "done":
            success += 1
        else:
            failed += 1

        # Save every 5 questions
        if idx % 5 == 0:
            save_excel(wb)

    # Final save
    save_excel(wb)

    # Summary
    print("\n" + "=" * 55)
    print("BATCH COMPLETE")
    print("=" * 55)
    print(f"  Generated: {success}")
    print(f"  Failed:    {failed}")
    print(f"  Remaining: {len(pending) - len(to_process)}")
    print(f"  Output:    {OUTPUT_XLSX}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate question dataset via GraphRAG API"
    )
    parser.add_argument(
        "--batch", type=int, default=15,
        help="Number of questions per batch (default: 15)"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter by category e.g. easy_deep_learning"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be generated without calling API"
    )

    parser.add_argument(
    "--reset-failed", action="store_true",
    help="Reset failed rows to pending before running"
    )
    args = parser.parse_args()

    run(
        batch_size = args.batch,
        category   = args.category,
        dry_run    = args.dry_run,
    )