# ============================================================
# scripts/domain_distribution.py
# Represent domain membership as a DISTRIBUTION, not a scalar.
#
# The request (Samar, 24 Sep): cosine similarity squashes the
# comparison to one number and routing then throws away everything
# except the argmax. He wants "10% cybersecurity, 20% X, 70% Y".
#
# The starting point is better than it sounds. route_by_embedding()
# already computes a score for all five domains, and GraphRAG 1.6
# already splits the chunk budget in proportion to those scores. A
# distribution is therefore already being computed and used - it is
# just never named, exposed, or calibrated.
#
# Calibration is the whole problem. Cosine similarities from a
# normalised sentence encoder sit in a narrow band (~0.5-0.8 for
# anything vaguely on-topic), so normalising them proportionally
# produces a near-uniform distribution even when routing is decisive.
# This script demonstrates that, then fixes it with temperature
# scaling fitted on the 75 labelled learning outcomes
# (Guo et al., "On Calibration of Modern Neural Networks", ICML 2017).
#
# Four ways to build the distribution, in increasing cost:
#   A  proportional   s / sum(s)          - current 1.6 behaviour
#   B  softmax        exp(s) / sum        - no better, same narrow band
#   C  temperature    exp(s/T) / sum      - A/B done properly
#   D  retrieval      relevance mass per domain, measured after
#                     retrieval rather than predicted before it
#
# Usage:
#   python scripts/domain_distribution.py --calibrate
#   python scripts/domain_distribution.py --query "Explain deadlock detection"
#   python scripts/domain_distribution.py --query "..." --retrieval
#   python scripts/domain_distribution.py --calibrate --query "..."
# ============================================================
import os
import sys
import csv
import json
import math
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "dataset", "inputs", "input_questions.csv")
CALIB_PATH = os.path.join(ROOT, "evaluation", "domain_temperature.json")


# ── distribution builders ───────────────────────────────────
def proportional(scores):
    """s_i / sum(s) - what GraphRAG 1.6 does for the chunk budget."""
    import numpy as np
    s = np.asarray(scores, dtype=float)
    s = np.clip(s, 1e-9, None)
    return s / s.sum()


def softmax(scores, T=1.0):
    """exp(s/T) / sum. T=1 is the plain softmax."""
    import numpy as np
    s = np.asarray(scores, dtype=float) / T
    s = s - s.max()                     # numerical stability
    e = np.exp(s)
    return e / e.sum()


def entropy(p, normalise=True):
    """Shannon entropy in nats. This - not KL - is the right measure
    of how AMBIGUOUS a single query is. KL needs two distributions;
    ambiguity is a property of one. Normalised entropy is 0 when the
    query belongs entirely to one domain and 1 when it is spread
    evenly across all of them.
    """
    import numpy as np
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    if normalise:
        n = len(p)
        return H / math.log(n) if n > 1 else 0.0
    return H


def js_divergence(p, q):
    """Jensen-Shannon divergence, in nats.

    Preferred over KL for comparing two domain distributions here.
    KL(p||q) is asymmetric and blows up to infinity wherever q has
    zero mass and p does not - which happens constantly with five
    domains and small counts. JSD is symmetric, always finite, and
    bounded by log(2).
    """
    import numpy as np
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ── calibration ─────────────────────────────────────────────
def load_labelled():
    """The 75 learning outcomes filed under exactly one domain."""
    seen, rows = set(), []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            lo = (r.get("learning_objective") or "").strip()
            dom = (r.get("domain") or "").strip()
            if not lo or not dom or "+" in dom or dom.lower() == "auto":
                continue
            key = (lo.lower(), dom)
            if key in seen:
                continue
            seen.add(key)
            rows.append((lo, dom))
    return rows


def expected_calibration_error(probs, correct, n_bins=10):
    """ECE: mean gap between stated confidence and actual accuracy.

    A well-calibrated model that says 70% should be right 70% of the
    time. Low ECE is the whole point of saying "60% database" instead
    of just "database".
    """
    import numpy as np
    conf = np.asarray([p.max() for p in probs])
    acc = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def calibrate(score_rows, domains, verbose=True):
    """Fit one temperature T by minimising negative log-likelihood.

    One free parameter on 75 points is a safe fit. T does not change
    which domain wins - argmax(s/T) == argmax(s) for any T > 0 - so
    routing accuracy is untouched. It only changes how confident the
    reported probabilities are.
    """
    import numpy as np

    S = np.array([r["scores"] for r in score_rows], dtype=float)
    y = np.array([domains.index(r["true"]) for r in score_rows])

    def nll(T):
        P = np.array([softmax(s, T) for s in S])
        return float(-np.log(np.clip(P[np.arange(len(y)), y],
                                     1e-12, None)).mean())

    # Coarse grid then golden-section refine; the objective is smooth
    # and 1-D, so this is plenty and avoids a scipy dependency.
    grid = np.concatenate([np.linspace(0.005, 0.2, 80),
                           np.linspace(0.21, 2.0, 40)])
    losses = [nll(t) for t in grid]
    best_i = int(np.argmin(losses))
    lo = grid[max(best_i - 1, 0)]
    hi = grid[min(best_i + 1, len(grid) - 1)]
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    T = (lo + hi) / 2

    if verbose:
        P1 = [softmax(s, 1.0) for s in S]
        PT = [softmax(s, T) for s in S]
        Pp = [proportional(s) for s in S]
        corr = [int(np.argmax(s) == yi) for s, yi in zip(S, y)]
        acc = sum(corr) / len(corr)

        print(f"\n  Fitted temperature T = {T:.4f}")
        print(f"  Routing accuracy (argmax, unchanged by T): {acc:.1%}\n")
        print(f"  {'method':28}{'NLL':>9}{'ECE':>9}"
              f"{'mean conf':>11}{'mean entropy':>14}")
        print("  " + "-" * 71)
        for name, P in (("proportional (1.6 now)", Pp),
                        ("softmax T=1", P1),
                        (f"softmax T={T:.4f}", PT)):
            mc = float(np.mean([p.max() for p in P]))
            me = float(np.mean([entropy(p) for p in P]))
            n = float(-np.log(np.clip(
                [p[yi] for p, yi in zip(P, y)], 1e-12, None)).mean())
            print(f"  {name:28}{n:>9.3f}"
                  f"{expected_calibration_error(P, corr):>9.3f}"
                  f"{mc:>11.1%}{me:>14.3f}")
        print(f"\n  Actual accuracy is {acc:.1%}. The method whose mean")
        print(f"  confidence sits closest to that, with the lowest ECE,")
        print(f"  is the one telling the truth about what it knows.")
    return T


# ── retrieval-evidence distribution ─────────────────────────
def retrieval_distribution(query, embedder, domains, threshold=0.0):
    """Distribution D: measured from what retrieval actually returned.

    A and C predict the domain mix from a query-vs-profile comparison
    BEFORE looking at the corpus. This measures where the context
    actually came from, by summing chunk relevance per domain. It is
    slower (it retrieves from every domain) but it is evidence rather
    than prediction, and it cannot claim mass for a domain whose graph
    has nothing relevant in it.
    """
    import numpy as np
    from graphrag.graphrag_domain_indexer import load_domain_graph
    from graphrag.graphrag_retriever import retrieve_graph_context

    mass = {}
    for d in domains:
        try:
            index, graph, nodes = load_domain_graph(d)
        except FileNotFoundError:
            mass[d] = 0.0
            continue
        res = retrieve_graph_context(query, index, graph, nodes, embedder,
                                     top_k_nodes=3, max_context_chunks=5)
        rel = [c.get("relevance", 0.0) for c in res.get("chunk_pool", [])]
        rel = [r for r in rel if r > threshold]
        mass[d] = float(sum(rel))

    total = sum(mass.values())
    if total <= 0:
        return np.ones(len(domains)) / len(domains), mass
    return np.array([mass[d] / total for d in domains]), mass


# ── main ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit T on the 75 labelled learning outcomes "
                         "and save it")
    ap.add_argument("--retrieval", action="store_true",
                    help="also compute distribution D (slow: retrieves "
                         "from all five graphs)")
    ap.add_argument("--compare", default="",
                    help="second query; reports JSD between the two")
    args = ap.parse_args()

    import numpy as np
    from utils.embedder import Embedder
    from graphrag.graphrag_domain_indexer import (
        build_domain_profiles, route_by_embedding, DOMAIN_MAP)

    domains = list(DOMAIN_MAP.keys())
    print("Loading embedder...")
    embedder = Embedder()
    build_domain_profiles(embedder)

    T = 1.0
    if os.path.exists(CALIB_PATH):
        T = json.load(open(CALIB_PATH))["temperature"]
        print(f"Loaded calibrated temperature T={T:.4f}")

    if args.calibrate:
        rows = load_labelled()
        print(f"\nCalibrating on {len(rows)} labelled learning outcomes")
        score_rows = []
        for i, (lo, dom) in enumerate(rows, 1):
            sc = route_by_embedding(lo, embedder, return_all_scores=True)
            score_rows.append({"scores": [sc[d] for d in domains],
                               "true": dom})
            if i % 15 == 0:
                print(f"  {i}/{len(rows)}")
        T = calibrate(score_rows, domains)
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        json.dump({"temperature": T, "domains": domains,
                   "n_calibration": len(score_rows),
                   "method": "NLL minimisation, Guo et al. ICML 2017"},
                  open(CALIB_PATH, "w"), indent=2)
        print(f"\n  saved: {CALIB_PATH}")

    if not args.query:
        return

    # ── the worked example ──────────────────────────────────
    print("\n" + "=" * 74)
    print(f"QUERY: {args.query}")
    print("=" * 74)

    sc = route_by_embedding(args.query, embedder, return_all_scores=True)
    s = np.array([sc[d] for d in domains])

    print("\n  Raw cosine scores (what we compute today):")
    for d in domains:
        print(f"    {d:26}{sc[d]:.3f}")
    print(f"\n  Current behaviour: argmax -> '{domains[int(s.argmax())]}'.")
    print(f"  The other four numbers are discarded.\n")

    methods = [("A  proportional  s/sum(s)", proportional(s)),
               ("B  softmax T=1", softmax(s, 1.0)),
               (f"C  softmax T={T:.4f} (calibrated)", softmax(s, T))]

    if args.retrieval:
        print("  Retrieving from all five graphs for method D...")
        pd_, mass = retrieval_distribution(args.query, embedder, domains)
        methods.append(("D  retrieval evidence", pd_))

    width = max(len(m[0]) for m in methods) + 2
    print(f"  {'method':{width}}" +
          "".join(f"{d[:6]:>9}" for d in domains) + f"{'entropy':>10}")
    print("  " + "-" * (width + 9 * len(domains) + 10))
    for name, p in methods:
        print(f"  {name:{width}}" +
              "".join(f"{v:>8.1%} " for v in p) +
              f"{entropy(p):>9.3f}")

    print(f"\n  Entropy is normalised: 0 = certain, 1 = evenly spread.")
    print(f"  This, not KL, is the ambiguity measure for a single query.")

    if len(methods) > 1:
        print(f"\n  Disagreement between methods (JSD):")
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                print(f"    {methods[i][0].split()[0]} vs "
                      f"{methods[j][0].split()[0]}: "
                      f"{js_divergence(methods[i][1], methods[j][1]):.4f}")

    if args.compare:
        sc2 = route_by_embedding(args.compare, embedder,
                                 return_all_scores=True)
        p2 = softmax(np.array([sc2[d] for d in domains]), T)
        p1 = softmax(s, T)
        print(f"\n  Second query: {args.compare}")
        print("  " + "".join(f"{v:>8.1%} " for v in p2))
        print(f"\n  JSD between the two queries: "
              f"{js_divergence(p1, p2):.4f} nats "
              f"(0 = identical, {math.log(2):.3f} = maximally different)")


if __name__ == "__main__":
    main()
