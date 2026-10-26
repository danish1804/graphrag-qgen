# ============================================================
# evaluation/retrieval_eval.py
# GraphRAG 1.5: judge-independent retrieval evaluation.
#
# Measures retrieval quality WITHOUT the LLM judge, so a retrieval
# change (alias merging, degree penalty, re-index, ...) can be
# scored directly and cheaply, decoupled from generation quality.
#
# Metrics per query:
#   cov@node : fraction of expected entities present among the
#              retrieved graph nodes (seed + neighbour), matched
#              via normalize_entity (survives the alias re-index).
#   cov@text : fraction of expected concepts that appear in the
#              retrieved CHUNK TEXT that reaches the prompt. This is
#              the primary metric — it credits coverage of a concept
#              even if no node is named exactly like it.
#   mean_rel : mean cosine of the final chunks to the query.
#
# Run (uses the current index + retrieval defaults):
#   python evaluation/retrieval_eval.py
# Baseline before the index-time batch, then re-run after the
# re-index and compare the aggregate numbers.
# ============================================================
import os
import sys
import json
import re
from statistics import mean

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graphrag.graphrag_indexer import normalize_entity
from graphrag.graphrag_domain_indexer import load_domain_graph
INDEX_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "indexes")
from graphrag.graphrag_retriever import retrieve_graph_context
from utils.embedder import Embedder

GOLD_PATH = os.path.join(_ROOT, "evaluation", "retrieval_gold.json")

PARAMS = dict(top_k_nodes=5, neighbour_depth=2,
              max_context_chunks=7, seed_ratio=0.4,
              rerank=True, degree_penalty=0.5, depth2_cap=50)


def _norm_set(entities):
    return {normalize_entity(e) for e in entities if e}


def _flatten(s):
    """Lowercase and neutralise hyphenation so 'white-box testing',
    'white box testing' and 'whitebox testing' all compare equal.

    Without this the harness reports false misses on every hyphenated
    compound - 'black-box'/'white-box' testing, 'write-back' cache,
    'multi-level' page table - because textbooks hyphenate and the gold
    set does not. `normalize_entity` already does this for graph nodes;
    the chunk-text check has to match, or cov@text and p@k understate
    any domain whose vocabulary is hyphen-heavy.
    """
    s = (s or "").lower()
    s = re.sub(r"[-‐‑‒–—−]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _in_text(term, text):
    # word-ish substring match, tolerant of plural/spacing/hyphenation
    t = _flatten(term)
    hay = _flatten(text)
    if t in hay or (not t.endswith("s") and (t + "s") in hay):
        return True
    # also try the fully closed-up form ("whitebox testing" for
    # "white box testing"). Guarded on length so short generic pairs
    # cannot match spuriously across word boundaries.
    t_closed = t.replace(" ", "")
    return len(t_closed) > 8 and t_closed in hay.replace(" ", "")


# ============================================================
# Rank / set metrics (task 25)
#
# IMPORTANT — relevance here is a PROXY, not a human judgement.
# A retrieved chunk counts as relevant if it contains at least one
# of the query's expected gold concepts. We do not hold per-chunk
# TREC-style qrels, so these numbers are directional and are NOT
# comparable to published IR benchmarks. Treat them as internal
# before/after signals only.
#
# What each metric is for:
#   hit@k   : floor check — did ANY on-topic chunk reach the prompt
#   p@k     : dilution — what share of the prompt is on-topic.
#             Motivated by Cuconasu et al. (SIGIR 2024): high-scoring
#             near-miss distractors measurably degrade LLM output, so
#             precision is decision-relevant even at fixed k.
#   mrr     : how early the first on-topic chunk appears
#   ndcg@k  : SECONDARY ONLY, for literature comparability. Our labels
#             are binary and all k chunks are concatenated into one
#             prompt, so NDCG's log position discount is mis-specified
#             (see docs/EVALUATION_ROADMAP.md §3.3). Do not tune on it.
#
# Concept-level RECALL is already reported as cov@text: of the
# concepts a good answer needs, the fraction that reached the prompt.
# True chunk-level Recall@k would require knowing every relevant chunk
# in the corpus, which needs full qrels we have not labelled.
# ============================================================

# How many distinct gold concepts a chunk must contain to count as
# relevant. The first version of this harness used 1, which saturated
# every rank metric at 1.00: gold sets contain common words ("table",
# "row", "memory", "layer") that appear in almost any chunk, so every
# query trivially had a relevant chunk at rank 1 and the metrics
# carried no signal. Requiring either a specific multi-word concept or
# two distinct concepts restores discrimination.
MIN_CONCEPTS_FOR_RELEVANCE = 2


def _specific_concepts(expected_raw):
    """Multi-word gold concepts, which are inherently discriminating
    ('inner join', 'demand paging', 'white box testing') as opposed to
    generic single words ('table', 'row', 'memory')."""
    return [e for e in expected_raw if len(e.split()) > 1]


def _chunk_relevant(chunk_text, expected_raw, anchors=None):
    """Proxy relevance for a retrieved chunk.

    A chunk is relevant when it contains either
      - one of the query's ANCHOR concepts (explicit `anchor_entities`
        in the gold file if present, else any multi-word gold concept), or
      - at least MIN_CONCEPTS_FOR_RELEVANCE distinct gold concepts.

    This approximates the TREC standard ("would you use this passage to
    write a report on the topic") far better than a single keyword match.
    It remains a PROXY: real per-chunk qrels are still the correct fix.
    """
    t = (chunk_text or "").lower()
    anchor_terms = anchors if anchors else _specific_concepts(expected_raw)
    if anchor_terms and any(_in_text(a, t) for a in anchor_terms):
        return True
    hits = sum(1 for e in expected_raw if _in_text(e, t))
    return hits >= MIN_CONCEPTS_FOR_RELEVANCE


def _rel_vector(chunks, expected_raw, anchors=None):
    """Binary relevance per retrieved chunk, in rank order."""
    return [1 if _chunk_relevant(c.get("text", ""), expected_raw, anchors)
            else 0 for c in chunks]


def hit_at_k(rel):
    """1.0 if any retrieved chunk is relevant."""
    return 1.0 if any(rel) else 0.0


def precision_at_k(rel, k=None):
    """Fraction of the top-k retrieved chunks that are relevant.

    k=None uses the whole pool (the original behaviour).

    Returns None when the pool holds fewer than k chunks. Computing
    P@10 over a 7-chunk pool by dividing by 7 would silently report a
    different quantity under the same name; computing it by dividing by
    10 would penalise the system for chunks it was never asked to
    retrieve. Neither is honest, so report nothing.
    """
    if not rel:
        return 0.0 if k is None else None
    if k is None:
        return sum(rel) / len(rel)
    if len(rel) < k:
        return None
    return sum(rel[:k]) / k


# Cutoffs for the precision curve. The point of the curve is to separate
# "the right material is at the top" from "noise accumulates further
# down" - a single pooled p@k cannot distinguish those.
P_AT_K_CUTOFFS = (1, 3, 5, 10)


def mrr(rel):
    """Reciprocal rank of the first relevant chunk."""
    for i, r in enumerate(rel, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel):
    """NDCG with binary relevance and linear (log2) discount.
    Secondary metric only — see the module note above."""
    import math
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel, start=1))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(gold_path=GOLD_PATH, index_dir=None, domains=None):
    """Score retrieval against the gold set.

    `index_dir` lets the SAME harness run against a different set of
    indexes - e.g. indexes/_backup_prebatch - so a before/after
    comparison uses one ruler on both sides. Comparing a number measured
    with an older harness against one measured with the current harness
    is not a valid comparison, however tempting.
    """
    gold = json.load(open(gold_path, encoding="utf-8"))
    print("Loading embedder...")
    embedder = Embedder()
    if index_dir:
        print(f"INDEX DIR OVERRIDE: {index_dir}")

    per_domain = {}
    for domain, items in gold.items():
        if domain.startswith("_"):
            continue
        if domains and domain not in domains:
            continue
        # Normalise gold concepts with the entity profile the graph was
        # built with (task 7); older indexes have no field and are v1.
        from graphrag.graphrag_indexer import set_entity_profile
        _meta = os.path.join(index_dir or INDEX_DIR_DEFAULT,
                             f"graphrag_{domain}_meta.json")
        _profile = "v1"
        if os.path.exists(_meta):
            _profile = json.load(open(_meta, encoding="utf-8")).get("entity_profile", "v1")
        set_entity_profile(_profile)
        print(f"[profile] {domain}: {_profile}")
        index, graph, nodes = (load_domain_graph(domain, index_dir=index_dir)
                               if index_dir else load_domain_graph(domain))
        rows = []
        print(f"\n===== {domain} =====")
        for it in items:
            q = it["query"]
            expected_raw = it["expected_entities"]
            expected = _norm_set(expected_raw)
            res = retrieve_graph_context(q, index, graph, nodes, embedder,
                                         **PARAMS)
            retrieved = _norm_set(
                [s["entity"] for s in res.get("seed_nodes", [])] +
                [n["entity"] for n in res.get("neighbour_nodes", [])])
            text = " ".join(c["text"] for c in
                            res.get("chunk_pool", [])).lower()
            rels = [c.get("relevance", 0.0)
                    for c in res.get("chunk_pool", [])
                    if isinstance(c.get("relevance"), float)]

            cov_node = len(expected & retrieved) / len(expected) if expected else 0
            in_text = [e for e in expected_raw if _in_text(e, text)]
            cov_text = len(in_text) / len(expected_raw) if expected_raw else 0
            mrel = round(mean(rels), 3) if rels else None
            missed = [e for e in expected_raw if e not in in_text]

            # rank / set metrics over the retrieved chunk list (task 25)
            chunks = res.get("chunk_pool", [])
            rel = _rel_vector(chunks, expected_raw,
                              it.get("anchor_entities"))
            hit = hit_at_k(rel)
            prec = precision_at_k(rel)
            rr = mrr(rel)
            ndcg = ndcg_at_k(rel)

            pk = {c: precision_at_k(rel, c) for c in P_AT_K_CUTOFFS}

            rows.append((cov_node, cov_text, mrel, hit, prec, rr, ndcg,
                         pk, len(rel)))
            print(f"  {q[:52]:54} cov@node={cov_node:.2f} "
                  f"cov@text={cov_text:.2f} mean_rel={mrel}")
            print(f"  {'':54} hit@k={hit:.0f}    "
                  f"p@k={prec:.2f}      mrr={rr:.2f}   ndcg={ndcg:.2f}")
            pk_s = "  ".join(
                f"p@{c}=" + ("  n/a" if pk[c] is None else f"{pk[c]:.2f}")
                for c in P_AT_K_CUTOFFS)
            print(f"  {'':54} {pk_s}   (pool={len(rel)})")
            if missed:
                print(f"      not in retrieved text: {missed}")
        per_domain[domain] = rows

    print("\n" + "=" * 96)
    print("SUMMARY (mean per domain)")
    print("=" * 96)
    hdr = (f"  {'domain':24}{'cov@node':>10}{'cov@text':>10}{'mean_rel':>10}"
           f"{'hit@k':>8}{'p@k':>8}{'mrr':>8}{'ndcg':>8}")
    print(hdr)
    print("-" * 96)
    agg = {i: [] for i in range(7)}
    pk_agg = {c: [] for c in P_AT_K_CUTOFFS}
    pk_dom = {}
    for domain, rows in per_domain.items():
        vals = []
        for i in range(7):
            xs = [r[i] for r in rows if r[i] is not None]
            vals.append(mean(xs) if xs else 0)
            agg[i] += xs
        pk_dom[domain] = {}
        for c in P_AT_K_CUTOFFS:
            xs = [r[7][c] for r in rows if r[7].get(c) is not None]
            pk_dom[domain][c] = mean(xs) if xs else None
            pk_agg[c] += xs
        print(f"  {domain:24}{vals[0]:>10.2f}{vals[1]:>10.2f}{vals[2]:>10.3f}"
              f"{vals[3]:>8.2f}{vals[4]:>8.2f}{vals[5]:>8.2f}{vals[6]:>8.2f}")
    print("-" * 96)
    o = [mean(agg[i]) if agg[i] else 0 for i in range(7)]
    print(f"  {'OVERALL':24}{o[0]:>10.2f}{o[1]:>10.2f}{o[2]:>10.3f}"
          f"{o[3]:>8.2f}{o[4]:>8.2f}{o[5]:>8.2f}{o[6]:>8.2f}")

    # ── Precision curve ─────────────────────────────────────
    pools = [r[8] for rows in per_domain.values() for r in rows]
    print()
    print("=" * 96)
    print(f"PRECISION AT CUTOFFS   (retrieved pool size: min {min(pools)}, "
          f"max {max(pools)}, k requested = {PARAMS['max_context_chunks']})")
    print("=" * 96)
    hdr = f"  {'domain':24}" + "".join(f"{'p@'+str(c):>9}"
                                       for c in P_AT_K_CUTOFFS)
    print(hdr)
    print("-" * 96)
    for domain in per_domain:
        line = f"  {domain:24}"
        for c in P_AT_K_CUTOFFS:
            v = pk_dom[domain][c]
            line += f"{'n/a':>9}" if v is None else f"{v:>9.2f}"
        print(line)
    print("-" * 96)
    line = f"  {'OVERALL':24}"
    for c in P_AT_K_CUTOFFS:
        xs = pk_agg[c]
        line += f"{mean(xs):>9.2f}" if xs else f"{'n/a':>9}"
    print(line)
    print()
    print("  Reading the curve: if p@1 and p@3 are high but p@5/p@10 fall, the")
    print("  retriever is ranking correctly and diluting further down - the fix")
    print("  is a smaller k, not better retrieval. If p@1 is already low, the")
    print("  ranking itself is wrong.")
    if any(pk_agg[c] == [] for c in P_AT_K_CUTOFFS):
        print(f"\n  n/a means the pool never reached that cutoff. k is currently")
        print(f"  {PARAMS['max_context_chunks']}, so deeper cutoffs are undefined. To")
        print("  measure dilution beyond k, re-run with --max-chunks 10 (this")
        print("  retrieves deeper than production and is a DIAGNOSTIC, not a")
        print("  measurement of the deployed configuration).")
    print()
    print("  cov@text is the headline metric: concept-level recall into the prompt.")
    print("  hit@k / p@k / mrr / ndcg use a PROXY relevance label (chunk contains")
    print("  >=1 expected concept), not human qrels - directional only, and not")
    print("  comparable with published IR benchmarks. Do not tune on ndcg; its")
    print("  position discount is mis-specified for concatenated context.")
    print("  See docs/EVALUATION_ROADMAP.md section 3.3.")
    return per_domain


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=None,
                    help="evaluate a different index set, e.g. "
                         "indexes/_backup_prebatch, for a like-for-like "
                         "before/after on the current metrics")
    ap.add_argument("--gold", default=GOLD_PATH)
    ap.add_argument("--neighbour-weighting", default=None,
                    choices=["degree_discount", "npmi_logcount"],
                    help="override retrieval.neighbour_weighting for this run")
    ap.add_argument("--domains", default=None,
                    help="comma-separated subset, e.g. operating_systems,database "
                         "(needed for variant folders that hold only some domains)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="override max_context_chunks. Production uses 3/5/7 "
                         "by difficulty and this harness defaults to 7. Set "
                         "10 to measure whether precision decays past the "
                         "production cutoff - a DIAGNOSTIC, not the deployed "
                         "configuration.")
    a = ap.parse_args()
    if a.neighbour_weighting:
        PARAMS["neighbour_weighting"] = a.neighbour_weighting
        print(f"[!] neighbour_weighting = {a.neighbour_weighting}")
    if a.max_chunks:
        PARAMS["max_context_chunks"] = a.max_chunks
        print(f"[!] max_context_chunks overridden to {a.max_chunks} "
              f"(diagnostic run, not the production setting)")
    evaluate(gold_path=a.gold, index_dir=a.index_dir,
             domains=a.domains.split(",") if a.domains else None)
