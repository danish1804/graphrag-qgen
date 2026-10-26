# ============================================================
# scripts/graph_quality_report.py
# Graph-quality measurement that does not depend on an LLM judge.
#
# Three reference-free and one reference-based family of measures:
#
#   1. STRUCTURE   isolated nodes, components, largest-component share,
#                  degree distribution, hub concentration.
#   2. CONCEPT COVERAGE against the textbook's own back-of-book index.
#                  Index main entries are an expert-curated concept list
#                  built independently of us (Alpizar-Chacon & Sosnovsky
#                  2021). Recall = share of index concepts that exist as
#                  graph nodes after the SAME normalisation the indexer
#                  uses.
#   3. RELATION RECALL against index structure. A sub-entry under a
#                  head entry (e.g. "deadlock -> detection") is an
#                  expert-asserted relation between two concepts. We
#                  measure the share of such pairs whose two concepts
#                  are both nodes AND connected within 1 or 2 hops.
#   4. NOISE       numeric-only nodes, very short nodes, nodes with no
#                  alphabetic character.
#
# Edge PRECISION needs human labels; this script writes a stratified
# edge sample sheet for two annotators (--sample-edges N).
#
# The extracted index terms are derived from client-supplied textbooks,
# so they are written under evaluation/gold_index/, which is git-ignored
# (NDA). Do not commit them to the personal remote.
#
# Usage:
#   python scripts/graph_quality_report.py --domains operating_systems,database
#   python scripts/graph_quality_report.py --index-dir indexes_variants/npmi
#   python scripts/graph_quality_report.py --sample-edges 150
# ============================================================
import os
import re
import sys
import json
import random
import argparse
import statistics
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(ROOT, "evaluation", "gold_index")

# Textbooks in the two pilot domains that carry a back-of-book index.
# start_page is the 0-based page where the index begins (found by
# scanning for an "Index" heading); the index runs to the last page.
INDEXED_BOOKS = {
    # source_39 (xv6 book) is excluded: its index lists code symbols
    # (kmap, argc, mpmain) rather than teaching concepts.
    "operating_systems": {"source_5.pdf": 831},
    # source_7 (Elmasri) is excluded: its index layout defeats the parser
    # (spot-check: ~8/20 concepts and ~5/15 relations clean). source_49
    # (Garcia-Molina) parses concepts cleanly but yields no sub-entry
    # structure, so it contributes concepts only. Spot-check precision:
    # source_5 ~19/20 concepts, ~14/15 relations; source_8 ~19/20, ~13/15.
    "database": {"source_8.pdf": 1327, "source_49.pdf": 1219},
}

PAGE_REF = re.compile(r"(,\s*)?\b[ivxlcdm]*\d[\d\s,–\-—fnI]*$", re.I)
SEE = re.compile(r"[.,]?\s*\bsee( also)?\b.*$", re.I)
DANGLING = re.compile(r"^(and|of|in|for|with|on|to|by|vs\.?|versus)\s+|\s+(and|of|in|for|with|on|to|by)$", re.I)


# ── 1. index extraction ────────────────────────────────────
def _clean_term(t):
    import unicodedata
    t = unicodedata.normalize("NFKC", t)        # ligatures: "ﬂoat" -> "float"
    t = SEE.sub("", t)
    t = re.sub(r",\s*\d.*$", "", t)          # drop ", 123, 456-789"
    t = PAGE_REF.sub("", t).strip(" ,.;:")
    t = DANGLING.sub("", t).strip(" ,.;:")
    # Inverted entries ("Pipes, UNIX"; "Hit ratio (H)"): keep the head.
    if ", " in t and len(t.split(", ", 1)[1].split()) <= 2:
        t = t.split(", ", 1)[0]
    t = re.sub(r"\s*\([^)]*\)?$", "", t).strip(" ,.;:")
    return t


def _see_targets(t):
    m = re.search(r"\bsee(?: also)?\s+(.*)$", t, re.I)
    if not m:
        return []
    return [_clean_term(x) for x in re.split(r"[;,]", m.group(1)) if x.strip()]


AUTHOR_RAW = re.compile(r"^[A-Z][A-Za-z'\-]+,\s+([A-Z]\.\s*)+")


def extract_index(pdf_path, start_page):
    """Return (main_terms, relation_pairs) from a back-of-book index.

    Uses line x-positions: within each column the smallest indent is a
    main entry, the next indent is a sub-entry, deeper indents are
    wrapped continuation lines.
    """
    import fitz
    from collections import Counter as _C
    doc = fitz.open(pdf_path)
    mains, pairs = [], []
    for pno in range(start_page, len(doc)):
        page = doc[pno]
        lines = []
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                txt = "".join(s["text"] for s in l["spans"]).strip()
                if txt:
                    lines.append((l["bbox"][0], l["bbox"][1], txt))
        if not lines:
            continue
        # Columns: split the sorted x-positions wherever there is a gap
        # wider than 60pt. Works for 2- and 3-column index layouts.
        xs = sorted({round(x) for x, _, _ in lines})
        bounds, start = [], xs[0]
        for a, b in zip(xs, xs[1:]):
            if b - a > 60:
                bounds.append((start, a))
                start = b
        bounds.append((start, xs[-1]))
        for lo, hi in bounds:
            col_lines = sorted([l for l in lines if lo - 1 <= round(l[0]) <= hi + 1],
                               key=lambda l: l[1])
            if len(col_lines) < 3:
                continue                       # page number / stray label
            base = min(l[0] for l in col_lines)
            offs = _C(round(l[0] - base) for l in col_lines)
            sub_levels = [o for o, c in sorted(offs.items()) if o > 3 and c >= 2]
            sub_off = sub_levels[0] if sub_levels else None
            entries = []
            for x, _, txt in col_lines:
                off = x - base
                if re.fullmatch(r"(index|[A-Z]|\d+)", txt.strip(), re.I):
                    continue
                if off <= 3:
                    entries.append([0, txt])
                elif sub_off is not None and off <= sub_off + 3:
                    entries.append([1, txt])
                elif entries:                  # wrapped continuation
                    entries[-1][1] += " " + txt
            head = None
            for level, txt in entries:
                txt = txt.replace("\xad ", "").replace("\xad", "")
                if AUTHOR_RAW.match(txt):     # author index ("Li, X., 123")
                    if level == 0:
                        head = None
                    continue
                term = _clean_term(txt)
                if level == 0:
                    head = term if len(term) >= 2 else None
                    if head:
                        mains.append(head)
                        for tgt in _see_targets(txt):
                            if tgt:
                                pairs.append((head, tgt, "see"))
                elif head and term:
                    pairs.append((head, term, "sub"))
                    for tgt in _see_targets(txt):
                        if tgt:
                            pairs.append((head, tgt, "see"))
    return mains, pairs


AUTHOR = re.compile(r"^[A-Z][A-Za-z'\-]+,\s+([A-Z]\.\s?)+[A-Z]?\.?$")
FUNCTION_WORDS = {"and", "or", "in", "of", "by", "if", "the", "a", "an", "to",
                  "for", "with", "on", "as", "at", "is", "vs", "versus"}
GENERIC_SUBS = {"defined", "definition", "definitions", "applications",
                "application", "example", "examples", "overview",
                "introduction", "use", "uses", "using", "types", "properties",
                "advantages", "disadvantages", "implementation", "issues",
                "algorithms", "approaches", "design", "description", "history",
                "features", "functions", "operations", "problems", "failure",
                "concept", "concepts", "basics", "summary", "notation"}


def _is_gold_concept(t):
    t = t.strip()
    if len(t) < 2 or AUTHOR.match(t):
        return False
    low = t.lower().strip(" -—–")
    if low in FUNCTION_WORDS or not re.search(r"[a-z]", low):
        return False
    if t.endswith("—") or "(contd" in low:
        return False
    return True


def _filter_gold(mains, pairs):
    mains = [m for m in mains if _is_gold_concept(m)]
    out = []
    for h, s, kind in pairs:
        if not (_is_gold_concept(h) and _is_gold_concept(s)):
            continue
        if kind == "sub" and s.lower() in GENERIC_SUBS:
            continue
        out.append((h, s, kind))
    return mains, out


def build_gold(domain, data_dir):
    books = INDEXED_BOOKS.get(domain, {})
    all_mains, all_pairs, per_book = [], [], {}
    for fname, start in books.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue
        mains, pairs = extract_index(path, start)
        mains, pairs = _filter_gold(mains, pairs)
        per_book[fname] = {"main_terms": len(mains), "pairs": len(pairs)}
        all_mains += mains
        all_pairs += pairs
    return all_mains, all_pairs, per_book


# ── 2. graph measures ──────────────────────────────────────
def structure(g):
    import networkx as nx
    n = g.number_of_nodes()
    deg = sorted((d for _, d in g.degree()), reverse=True)
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    top1pct = max(1, n // 100)
    return {
        "nodes": n,
        "edges": g.number_of_edges(),
        "isolated_pct": round(100 * sum(1 for d in deg if d == 0) / n, 1),
        "components": len(comps),
        "giant_component_pct": round(100 * comps[0] / n, 1),
        "median_degree": deg[len(deg) // 2],
        "max_degree": deg[0],
        # share of all edge endpoints held by the top 1% of nodes
        "hub_concentration_pct": round(100 * sum(deg[:top1pct]) / max(1, sum(deg)), 1),
    }


def noise(g):
    nodes = list(g.nodes())
    return {
        "numeric_only_nodes": sum(1 for x in nodes if re.fullmatch(r"[\d\s.\-–]+", x)),
        "no_alpha_nodes": sum(1 for x in nodes if not re.search(r"[a-z]", x)),
        "short_nodes_le3": sum(1 for x in nodes if len(x) <= 3),
    }


def coverage(g, mains, normalise):
    gold = {normalise(t) for t in mains}
    gold.discard("")
    hits = {t for t in gold if t in g}
    return {"index_concepts": len(gold), "found_as_nodes": len(hits),
            "concept_recall_pct": round(100 * len(hits) / max(1, len(gold)), 1)}, gold, hits


def relation_recall(g, pairs, normalise):
    import networkx as nx
    norm = []
    for h, s, kind in pairs:
        a, b = normalise(h), normalise(s)
        if a and b and a != b:
            norm.append((a, b, kind))
    norm = list({(a, b): (a, b, k) for a, b, k in norm}.values())
    both = [(a, b, k) for a, b, k in norm if a in g and b in g]
    reach = {}
    hop1 = hop2 = 0
    for a, b, _ in both:
        if a not in reach:
            reach[a] = nx.single_source_shortest_path_length(g, a, cutoff=2)
        d = reach[a].get(b)
        if d == 1:
            hop1 += 1
        if d is not None and d <= 2:
            hop2 += 1
    total = len(norm)
    return {
        "index_relations": total,
        "both_concepts_are_nodes": len(both),
        "connected_1hop": hop1,
        "connected_within_2hops": hop2,
        "relation_recall_1hop_pct": round(100 * hop1 / max(1, total), 1),
        "relation_recall_2hop_pct": round(100 * hop2 / max(1, total), 1),
        "of_representable_1hop_pct": round(100 * hop1 / max(1, len(both)), 1),
    }


def edge_sample(g, n, rng):
    """Stratified by edge weight tercile so precision is not dominated
    by the many weight-2 edges."""
    edges = [(u, v, d["weight"]) for u, v, d in g.edges(data=True)]
    edges.sort(key=lambda e: e[2])
    k = len(edges)
    strata = [edges[: k // 3], edges[k // 3: 2 * k // 3], edges[2 * k // 3:]]
    out = []
    for i, s in enumerate(strata):
        for u, v, w in rng.sample(s, min(len(s), n // 3)):
            out.append({"stratum": ["low", "mid", "high"][i], "concept_a": u,
                        "concept_b": v, "weight": w,
                        "a_is_valid_concept": "", "b_is_valid_concept": "",
                        "meaningfully_related": "", "notes": ""})
    rng.shuffle(out)
    return out


# ── main ───────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="operating_systems,database")
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "indexes"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "tested"))
    ap.add_argument("--sample-edges", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild-gold", action="store_true",
                    help="re-extract the index gold lists even if cached")
    args = ap.parse_args()

    import pickle
    from graphrag.graphrag_indexer import normalize_entity, set_entity_profile

    os.makedirs(GOLD_DIR, exist_ok=True)
    rng = random.Random(args.seed)
    report = {"index_dir": args.index_dir, "domains": {}}

    for domain in [d.strip() for d in args.domains.split(",") if d.strip()]:
        print("=" * 72)
        print(domain)
        print("=" * 72)
        gold_path = os.path.join(GOLD_DIR, f"{domain}.json")
        if os.path.exists(gold_path) and not args.rebuild_gold:
            gd = json.load(open(gold_path, encoding="utf-8"))
            mains, pairs, per_book = gd["main_terms"], [tuple(p) for p in gd["pairs"]], gd["per_book"]
            print(f"  gold index loaded: {gold_path}")
        else:
            mains, pairs, per_book = build_gold(domain, args.data_dir)
            json.dump({"main_terms": mains, "pairs": pairs, "per_book": per_book},
                      open(gold_path, "w", encoding="utf-8"), indent=1)
            print(f"  gold index extracted -> {gold_path}")
        for b, s in per_book.items():
            print(f"    {b}: {s['main_terms']} main entries, {s['pairs']} relation pairs")

        # Normalise gold terms with the profile the graph was built with,
        # otherwise a v2 graph is scored with v1 normalisation (or vice versa).
        meta_path = os.path.join(args.index_dir, f"graphrag_{domain}_meta.json")
        profile = "v1"
        if os.path.exists(meta_path):
            profile = json.load(open(meta_path, encoding="utf-8")).get("entity_profile", "v1")
        set_entity_profile(profile)
        print(f"  graph entity profile: {profile}")
        g = pickle.load(open(os.path.join(args.index_dir, f"graphrag_{domain}_graph.pkl"), "rb"))
        st = structure(g)
        nz = noise(g)
        cov, gold, hits = coverage(g, mains, normalize_entity)
        rel = relation_recall(g, [p for p in pairs if p[2] == "sub"], normalize_entity)
        syn = relation_recall(g, [p for p in pairs if p[2] == "see"], normalize_entity)
        syn = {"see_references": syn["index_relations"],
               "both_are_nodes": syn["both_concepts_are_nodes"],
               "merged_into_one_node": sum(1 for h, t, k in pairs if k == "see"
                                           and normalize_entity(h) and normalize_entity(h) == normalize_entity(t)),
               "connected_1hop": syn["connected_1hop"]}

        print(f"\n  STRUCTURE  {st}")
        print(f"  NOISE      {nz}")
        print(f"  COVERAGE   {cov}")
        print(f"  RELATIONS  {rel}")
        print(f"  SYNONYMS   {syn}")
        missing = sorted(gold - hits, key=len)[:25]
        print(f"  sample index concepts missing from graph: {missing}")

        rec = {"entity_profile": profile, "structure": st, "noise": nz, "coverage": cov, "relations": rel, "synonyms": syn,
               "gold_books": per_book}
        if args.sample_edges:
            sample = edge_sample(g, args.sample_edges, rng)
            path = os.path.join(GOLD_DIR, f"{domain}_edge_sample.csv")
            import csv
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(sample[0].keys()))
                w.writeheader()
                w.writerows(sample)
            print(f"  edge sample ({len(sample)}) -> {path}")
        report["domains"][domain] = rec
        print()

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(report, open(path, "w", encoding="utf-8"), indent=2)
        print(f"written: {path}")


if __name__ == "__main__":
    main()
