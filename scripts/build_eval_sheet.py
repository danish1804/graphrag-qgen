# ============================================================
# scripts/build_eval_sheet.py
# Populate the human-evaluation workbook from saved evaluation
# records, so the team can label a golden corpus.
#
# Fills:
#   Questions   — id, domain, type, target difficulty, LO, question text
#   System_Key  — system difficulty + LLM judge scores (BLIND: raters
#                 must not open this tab until labelling is finished)
#
# The rater tabs and the Agreement/kappa formulas are left untouched.
#
# IMPORTANT — run this AFTER re-indexing. Questions generated from a
# stale index produce a golden corpus that no longer describes the
# system, wasting the raters' effort.
#
# Usage:
#   python scripts/build_eval_sheet.py
#   python scripts/build_eval_sheet.py --limit 60 --balanced
#   python scripts/build_eval_sheet.py --evals outputs/evaluations \
#       --template docs/human_evaluation_template.xlsx \
#       --out docs/human_evaluation_ROUND1.xlsx
# ============================================================
import os
import sys
import json
import glob
import random
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EVALS = os.path.join(ROOT, "outputs", "evaluations")
DEFAULT_TEMPLATE = os.path.join(ROOT, "docs", "human_evaluation_template.xlsx")
DEFAULT_OUT = os.path.join(ROOT, "docs", "human_evaluation_ROUND1.xlsx")

SEED = 42
MAX_ROWS = 60          # capacity built into the template


def load_corpus(path):
    """Read a corpus produced by scripts/build_question_corpus.py."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("judge_enabled"):
        print("  [!] This corpus was generated with the judge ENABLED, so it "
              "is quality-filtered. Human labels on it cannot properly "
              "calibrate the judge (range restriction).")
    out = []
    for q in payload.get("questions", []):
        # Prefer scores from the INDEPENDENT judge (scripts/judge_corpus.py)
        # over anything the generation pipeline produced, which would be
        # the generator grading itself.
        ind = q.get("independent_judge") or {}
        js = ind or (q.get("judge_scores") or {})
        out.append({
            "eval_id":    q.get("question_id", ""),
            "domain":     q.get("domain") or "unknown",
            "qtype":      q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            "lo":         q.get("learning_outcome", ""),
            "question":   (q.get("question") or "").strip(),
            "j_question": js.get("question_overall"),
            "j_answer":   js.get("answer_overall"),
            "j_retrieval": js.get("retrieval_overall"),
            "judge":      js.get("judge_model", "" if not js else "pipeline"),
            "backend":    js.get("judge_backend",
                                 "" if not q.get("judge_enabled") else "enabled"),
        })
    return out


def load_records(eval_dir):
    """Read every evaluation JSON and keep the ones with a question."""
    records = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue

        question = (rec.get("question") or {}).get("generated")
        if not question or not str(question).strip():
            continue

        meta = rec.get("metadata") or {}
        inp = rec.get("input") or {}
        domain = (meta.get("domain_routed_to")
                  or meta.get("domain")
                  or (meta.get("domains_used") or ["unknown"])[0]
                  if meta.get("domains_used") else
                  meta.get("domain_routed_to") or "unknown")

        records.append({
            "eval_id":    rec.get("eval_id", os.path.basename(path)[:-5]),
            "domain":     domain or "unknown",
            "qtype":      inp.get("question_type", ""),
            "difficulty": inp.get("difficulty", ""),
            "lo":         inp.get("learning_objective", ""),
            "question":   str(question).strip(),
            "j_question": (rec.get("question") or {}).get("overall"),
            "j_answer":   (rec.get("answer") or {}).get("overall"),
            "j_retrieval": (rec.get("retrieval") or {}).get("overall"),
            "judge":      rec.get("judge_model", ""),
            "backend":    rec.get("judge_backend", ""),
        })
    return records


def deduplicate(records):
    """Drop repeated question text — near-identical items waste rater time."""
    seen, out = set(), []
    for rec in records:
        key = " ".join(rec["question"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def sample(records, limit, balanced):
    """Take `limit` records, optionally balanced across domain x difficulty."""
    random.seed(SEED)
    if len(records) <= limit:
        return records
    if not balanced:
        return random.sample(records, limit)

    buckets = defaultdict(list)
    for rec in records:
        buckets[(rec["domain"], rec["difficulty"])].append(rec)
    for items in buckets.values():
        random.shuffle(items)

    chosen, keys = [], sorted(buckets.keys())
    while len(chosen) < limit and any(buckets[k] for k in keys):
        for key in keys:                      # round-robin across buckets
            if buckets[key] and len(chosen) < limit:
                chosen.append(buckets[key].pop())
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None,
                    help="corpus JSON from build_question_corpus.py "
                         "(preferred). Falls back to --evals if omitted.")
    ap.add_argument("--evals", default=DEFAULT_EVALS)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=MAX_ROWS)
    ap.add_argument("--balanced", action="store_true",
                    help="spread the sample across domain x difficulty")
    ap.add_argument("--shuffle", action="store_true", default=True,
                    help="randomise row order so raters cannot infer "
                         "grouping (on by default)")
    args = ap.parse_args()

    if args.limit > MAX_ROWS:
        print(f"[!] --limit capped at {MAX_ROWS} (template capacity).")
        args.limit = MAX_ROWS

    if not os.path.isfile(args.template):
        sys.exit(f"[!] Template not found: {args.template}")

    if args.corpus:
        if not os.path.isfile(args.corpus):
            sys.exit(f"[!] Corpus not found: {args.corpus}")
        print(f"Reading question corpus from {args.corpus} ...")
        records = load_corpus(args.corpus)
    else:
        if not os.path.isdir(args.evals):
            sys.exit(f"[!] Evaluation folder not found: {args.evals}")
        print(f"Reading evaluation records from {args.evals} ...")
        records = load_records(args.evals)
    print(f"  usable records      : {len(records)}")
    records = deduplicate(records)
    print(f"  after de-duplication: {len(records)}")
    if not records:
        sys.exit("[!] No usable questions found. Generate some first.")

    records = sample(records, args.limit, args.balanced)

    # A corpus from build_question_corpus.py is ALREADY shuffled and
    # already carries stable question ids. Do not reshuffle or renumber
    # it: the sheet's ids must stay joinable to the corpus, or the
    # raters' scores cannot be matched back to the questions (and hence
    # to the judge scores) once labelling is finished.
    if args.corpus:
        print("  keeping corpus ids and order (joinable to "
              "question_corpus.json)")
    elif args.shuffle:
        random.seed(SEED)
        random.shuffle(records)
    print(f"  selected for rating : {len(records)}")

    try:
        import openpyxl
    except ImportError:
        sys.exit("[!] pip install openpyxl")

    wb = openpyxl.load_workbook(args.template)
    qs, sk = wb["Questions"], wb["System_Key"]

    # Clear the template's demonstration row from the rater tabs. The
    # template ships one worked example so the expected format is clear;
    # leaving it in a live sheet means the first rater opens the file to
    # find Q001 already scored, which both confuses them and anchors
    # their judgement.
    for tab in ("Rater_1", "Rater_2", "Rater_3"):
        if tab not in wb.sheetnames:
            continue
        ws = wb[tab]
        for row in range(3, MAX_ROWS + 3):
            for col in range(2, 8):          # B..G, leaves the id formula
                ws.cell(row=row, column=col).value = None

    # clear the example row and any previous content
    for row in range(2, MAX_ROWS + 2):
        for col in range(1, 8):
            qs.cell(row=row, column=col).value = None
        qs.cell(row=row, column=1).font = openpyxl.styles.Font(
            name="Arial", size=10)
    qs.cell(row=3, column=1).value = None

    for i, rec in enumerate(records):
        row = 2 + i
        # Preserve the corpus id so the sheet joins back to
        # question_corpus.json; only synthesise one if there isn't one.
        qid = rec.get("eval_id") or f"Q{i + 1:03d}"
        qs.cell(row=row, column=1, value=qid)
        qs.cell(row=row, column=2, value=rec["domain"])
        qs.cell(row=row, column=3, value=rec["qtype"])
        # column 4 is the learning outcome, NOT the difficulty. The
        # intended difficulty is written only into System_Key (below),
        # so raters cannot see it while judging difficulty themselves.
        qs.cell(row=row, column=4, value=rec["lo"])
        qs.cell(row=row, column=5, value=rec["question"])
        for col in range(1, 6):
            qs.cell(row=row, column=col).alignment = \
                openpyxl.styles.Alignment(vertical="top", wrap_text=True)
            qs.cell(row=row, column=col).font = \
                openpyxl.styles.Font(name="Arial", size=10)

        # System_Key rows start at 4; columns A/G/H are formulas already.
        # Column B (system_difficulty) is written here as a literal
        # because it is deliberately absent from the Questions tab.
        krow = 4 + i
        sk.cell(row=krow, column=2, value=rec["difficulty"])
        sk.cell(row=krow, column=3, value=rec["j_question"])
        sk.cell(row=krow, column=4, value=rec["j_answer"])
        sk.cell(row=krow, column=5, value=rec["j_retrieval"])
        sk.cell(row=krow, column=6,
                value=f"{rec['judge']} ({rec['backend']})".strip())

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    wb.save(args.out)

    backends = {r["backend"] for r in records if r["backend"]}
    print(f"\nWrote {args.out}")
    print(f"  questions          : {len(records)}")
    print(f"  domains            : {sorted({r['domain'] for r in records})}")
    print(f"  judge backend(s)   : {sorted(backends) or ['unknown']}")
    if "reuse_generator" in backends:
        print("  [!] Some judge scores came from the SELF-BIASED backend "
              "(generator grading itself). Note this when comparing the "
              "judge against the human labels.")
    print("\nNext:")
    print("  1. Send this file to the three raters.")
    print("  2. Each rater fills ONLY their own tab (Rater_1/2/3).")
    print("  3. Nobody opens System_Key until all three are finished.")
    print("  4. Open the Agreement tab for kappa and the match rate.")


if __name__ == "__main__":
    main()
