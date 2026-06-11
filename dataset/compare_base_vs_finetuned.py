# ============================================================
# dataset/compare_base_vs_finetuned.py
#
# Paired comparison of LLM-judge scores between the BASE model
# dataset and the fine-tuned (question LoRA) dataset.
#
# Rows are matched on Q_ID, so this is a PAIRED design:
# the same learning objective / difficulty / question type was
# run through both models. Significance is tested with the
# Wilcoxon signed-rank test (appropriate for paired ordinal
# judge scores; does not assume normality).
#
# Usage:
#   python dataset/compare_base_vs_finetuned.py
#
#   # Custom paths:
#   python dataset/compare_base_vs_finetuned.py \
#       --base dataset/outputs/P000233DS_Question_Dataset_BASE_MODEL.xlsx \
#       --finetuned dataset/outputs/P000233DS_Question_Dataset.xlsx
#
# Outputs:
#   dataset/outputs/comparison_report.md   (tables for the report)
#   dataset/outputs/comparison_per_row.csv (per-question deltas)
#
# Repro: Python 3.10+, openpyxl, scipy (optional — falls back to
#        a normal-approximation Wilcoxon if scipy is missing).
#
# Author: Mohammed Danish Alam
# Project: P000233DS — EdGenAI Technologies Capstone
# ============================================================

import argparse
import csv
import math
import os
from collections import defaultdict

import openpyxl

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.join(_DIR, "outputs", "P000233DS_Question_Dataset_BASE_MODEL.xlsx")
DEFAULT_FT   = os.path.join(_DIR, "outputs", "P000233DS_Question_Dataset.xlsx")
REPORT_MD    = os.path.join(_DIR, "outputs", "comparison_report.md")
PER_ROW_CSV  = os.path.join(_DIR, "outputs", "comparison_per_row.csv")

HEADER_ROW = 2          # row 1 = merged group headers, row 2 = column names
SCORE_COLS = {          # column name -> short key
    "Q Score (/5)": "q_score",
    "A Score (/5)": "a_score",
    "R Score (/5)": "r_score",
}
PRIMARY = "q_score"     # headline metric (question_overall from the judge)


# ────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────

def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None    # "N/A", blanks, etc.


def load_dataset(path: str) -> dict:
    """Return {Q_ID: row_dict} for rows with Status == done."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    rows = ws.iter_rows(min_row=HEADER_ROW, values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    idx = {name: i for i, name in enumerate(header)}

    required = ["Q_ID", "Domain", "Difficulty", "Question Type", "Status"] + list(SCORE_COLS)
    missing = [c for c in required if c not in idx]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns: {missing}")

    out = {}
    for r in rows:
        if r is None or r[idx["Q_ID"]] is None:
            continue
        status = str(r[idx["Status"]] or "").strip().lower()
        if status != "done":
            continue
        rec = {
            "domain":     str(r[idx["Domain"]] or "").strip(),
            "difficulty": str(r[idx["Difficulty"]] or "").strip(),
            "qtype":      str(r[idx["Question Type"]] or "").strip(),
            "attempts":   _to_float(r[idx["Attempts"]]) if "Attempts" in idx else None,
        }
        for col, key in SCORE_COLS.items():
            rec[key] = _to_float(r[idx[col]])
        out[str(r[idx["Q_ID"]]).strip()] = rec
    wb.close()
    return out


# ────────────────────────────────────────────────────────────
# Statistics (pure-python fallbacks; scipy used if available)
# ────────────────────────────────────────────────────────────

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Uses scipy if installed; otherwise a normal approximation with
    tie correction (valid for n >= ~15, which holds here).
    Zero differences are dropped (standard 'wilcox' handling).
    Returns (statistic W, p_value, n_effective) or (None, None, n) if degenerate.
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 5:
        return None, None, n   # too few non-zero pairs for a meaningful test

    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(d, alternative="two-sided")
        return float(res.statistic), float(res.pvalue), n
    except ImportError:
        pass

    # Normal approximation with tie correction
    abs_d = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1][0] == abs_d[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[abs_d[k][1]] = avg_rank
        i = j + 1

    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    # tie correction term
    tie_counts = defaultdict(int)
    for x in d:
        tie_counts[abs(x)] += 1
    tie_term = sum(t ** 3 - t for t in tie_counts.values())
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48)
    if sigma == 0:
        return w_plus, None, n
    z = (w_plus - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return w_plus, p, n


def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation (effect size, -1..1).
    Positive = fine-tuned scores higher than base."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return float("nan")
    abs_d = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1][0] == abs_d[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[abs_d[k][1]] = avg_rank
        i = j + 1
    w_plus  = sum(r for r, x in zip(ranks, d) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, d) if x < 0)
    total = w_plus + w_minus
    return (w_plus - w_minus) / total if total else float("nan")


# ────────────────────────────────────────────────────────────
# Comparison
# ────────────────────────────────────────────────────────────

def compare_metric(pairs, key):
    """pairs: list of (base_rec, ft_rec). Returns summary dict for one metric."""
    valid = [(b[key], f[key]) for b, f in pairs
             if b[key] is not None and f[key] is not None]
    if not valid:
        return None
    base_v = [v[0] for v in valid]
    ft_v   = [v[1] for v in valid]
    diffs  = [f - b for b, f in valid]
    w, p, n_eff = wilcoxon_signed_rank(diffs)
    return {
        "n":          len(valid),
        "base_mean":  mean(base_v),  "base_std": std(base_v),  "base_med": median(base_v),
        "ft_mean":    mean(ft_v),    "ft_std":   std(ft_v),    "ft_med":   median(ft_v),
        "delta":      mean(diffs),
        "wins":       sum(1 for d in diffs if d > 0),
        "ties":       sum(1 for d in diffs if d == 0),
        "losses":     sum(1 for d in diffs if d < 0),
        "wilcoxon_w": w, "p_value": p, "n_nonzero": n_eff,
        "effect_rb":  rank_biserial(diffs),
    }


def fmt_p(p):
    if p is None:
        return "n/a"
    return f"{p:.4f}" + (" *" if p < 0.05 else "")


def fmt_effect(r):
    return "n/a" if (r is None or math.isnan(r)) else f"{r:+.2f}"


def metric_table(rows, title):
    """rows: list of (label, summary). Returns a markdown table string."""
    lines = [f"### {title}", "",
             "| Group | n | Base mean±sd | FT mean±sd | Δ mean | W/T/L | p (Wilcoxon) | Effect (r) |",
             "|---|---|---|---|---|---|---|---|"]
    for label, s in rows:
        if s is None:
            lines.append(f"| {label} | 0 | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {label} | {s['n']} "
            f"| {s['base_mean']:.2f}±{s['base_std']:.2f} "
            f"| {s['ft_mean']:.2f}±{s['ft_std']:.2f} "
            f"| {s['delta']:+.2f} "
            f"| {s['wins']}/{s['ties']}/{s['losses']} "
            f"| {fmt_p(s['p_value'])} "
            f"| {fmt_effect(s['effect_rb'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Paired base vs fine-tuned judge-score comparison")
    ap.add_argument("--base",      default=DEFAULT_BASE, help="Base-model dataset xlsx")
    ap.add_argument("--finetuned", default=DEFAULT_FT,   help="Fine-tuned dataset xlsx")
    args = ap.parse_args()

    base = load_dataset(args.base)
    ft   = load_dataset(args.finetuned)

    common = sorted(set(base) & set(ft))
    only_base = len(base) - len(common)
    only_ft   = len(ft) - len(common)
    print(f"Base rows (done): {len(base)}   Fine-tuned rows (done): {len(ft)}")
    print(f"Matched on Q_ID : {len(common)}   (unmatched: {only_base} base-only, {only_ft} ft-only)")
    if len(common) < 30:
        print("[WARN] Fewer than 30 matched pairs — treat p-values with caution.")

    pairs = [(base[q], ft[q]) for q in common]

    md = ["# Base vs Fine-tuned (Question LoRA) — Paired Comparison",
          "",
          f"- Base dataset: `{os.path.basename(args.base)}` ({len(base)} done rows)",
          f"- Fine-tuned dataset: `{os.path.basename(args.finetuned)}` ({len(ft)} done rows)",
          f"- Matched pairs (same Q_ID): **{len(common)}**",
          "- Test: two-sided Wilcoxon signed-rank on paired differences "
          "(zero-difference pairs excluded). `*` marks p < 0.05.",
          "- Effect size: matched-pairs rank-biserial correlation "
          "(positive favours the fine-tuned model).",
          ""]

    # ── Overall, all three judge metrics ──
    overall_rows = []
    for col, key in SCORE_COLS.items():
        overall_rows.append((col, compare_metric(pairs, key)))
    md.append(metric_table(overall_rows, "Overall (all matched pairs)"))

    # ── Primary metric breakdowns ──
    for group_key, title in [("difficulty", "Q Score by difficulty"),
                             ("domain",     "Q Score by domain"),
                             ("qtype",      "Q Score by question type")]:
        groups = defaultdict(list)
        for b, f in pairs:
            groups[b[group_key] or "unknown"].append((b, f))
        rows = [(g, compare_metric(groups[g], PRIMARY)) for g in sorted(groups)]
        md.append(metric_table(rows, title))

    # ── Attempts (regeneration efficiency) ──
    att = compare_metric(pairs, "attempts")
    if att:
        md.append(metric_table([("Attempts (lower=better)", att)],
                               "Judge-triggered regeneration attempts"))
        md.append("_Note: for Attempts a negative Δ is an improvement._\n")

    report = "\n".join(md)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)

    # ── Per-row CSV ──
    with open(PER_ROW_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Q_ID", "Domain", "Difficulty", "Question Type",
                    "Base Q Score", "FT Q Score", "Delta Q Score",
                    "Base A Score", "FT A Score",
                    "Base R Score", "FT R Score"])
        for q in common:
            b, t = base[q], ft[q]
            dq = (t["q_score"] - b["q_score"]
                  if b["q_score"] is not None and t["q_score"] is not None else "")
            w.writerow([q, b["domain"], b["difficulty"], b["qtype"],
                        b["q_score"], t["q_score"], dq,
                        b["a_score"], t["a_score"],
                        b["r_score"], t["r_score"]])

    print("\n" + report)
    print(f"\nSaved: {REPORT_MD}")
    print(f"Saved: {PER_ROW_CSV}")


if __name__ == "__main__":
    main()
