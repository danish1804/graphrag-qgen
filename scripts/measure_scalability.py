# ============================================================
# scripts/measure_scalability.py
# What does approximate search actually cost us?
#
# graphrag_indexer.py switches from exact to approximate search
# above 1,000 nodes (build_graphrag_index, ~line 668):
#
#     if n_nodes > 1000:
#         n_clusters = min(int(np.sqrt(n_nodes)), 128)
#         index = faiss.IndexIVFFlat(quantizer, DIMENSION, n_clusters,
#                                    faiss.METRIC_INNER_PRODUCT)
#         index.nprobe = min(n_clusters, 16)
#     else:
#         index = faiss.IndexFlatIP(DIMENSION)      # exact
#
# load_graphrag_index then forces nprobe = 16 on load (~line 1059).
#
# Every domain graph is above that threshold, so all retrieval in the
# system uses approximate search. We traded accuracy for scale and
# never measured the price. This measures it.
#
# Method: pull the stored vectors out of the production index itself
# (make_direct_map + reconstruct_n) and build an exact IndexFlatIP over
# those same vectors. Both indexes then hold byte-identical data, so
# any difference in results is purely the approximation — not an
# artefact of re-embedding. This matters because the indexer embeds
# "node_name: first_two_chunks"[:500], not the bare node name, so
# re-embedding by hand is easy to get subtly wrong.
#
# Reports per domain:
#   - build time, query latency, memory               (scalability)
#   - top-k agreement between approximate and exact   (accuracy cost)
#   - an nprobe sweep, so the trade can be tuned rather than guessed
#
# Read-only. Writes nothing to indexes/.
#
# Usage:
#   python scripts/measure_scalability.py
#   python scripts/measure_scalability.py --domains database,operating_systems
#   python scripts/measure_scalability.py --k 3 --nprobe-sweep 1,4,8,16,32
#   python scripts/measure_scalability.py --json outputs/scalability.json
# ============================================================
import os
import sys
import json
import time
import argparse
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "evaluation", "retrieval_gold.json")
ALL_DOMAINS = ["database", "deep_learning", "software_engineering",
               "operating_systems", "computer_architecture"]


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def vectors_from_index(index, n, dim):
    """Recover the stored vectors from a built FAISS index.

    IndexIVFFlat stores vectors scattered across inverted lists, so a
    direct map is needed before reconstruct_n works. Returns None if
    reconstruction is unsupported, and the caller falls back.
    """
    import numpy as np
    try:
        if hasattr(index, "make_direct_map"):
            index.make_direct_map()
        out = np.zeros((n, dim), dtype="float32")
        index.reconstruct_n(0, n, out)
        return out
    except Exception as e:
        print(f"  [warn] could not reconstruct vectors from index: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default=",".join(ALL_DOMAINS))
    ap.add_argument("--k", type=int, default=3,
                    help="top-k to compare; production top_k_nodes is 3")
    ap.add_argument("--nprobe-sweep", default="1,2,4,8,16,32,64",
                    help="nprobe values to evaluate; production is "
                         "min(n_clusters, 16)")
    ap.add_argument("--queries", type=int, default=0,
                    help="cap number of gold queries (0 = all)")
    ap.add_argument("--json", default="",
                    help="optional path to write results as JSON")
    ap.add_argument("--synthetic", type=int, default=0, metavar="N",
                    help="skip the embedder and use N synthetic queries "
                         "per domain instead: real node vectors with "
                         "Gaussian noise added, renormalised. Only 15 gold "
                         "queries exist, so this gives a far tighter "
                         "estimate of recall - at the cost of the queries "
                         "not being real ones. Report both.")
    ap.add_argument("--noise", type=float, default=0.6,
                    help="noise sigma for --synthetic. 0.0 puts the query "
                         "exactly on a node (easy case, optimistic); "
                         "0.6 approximates a query near but not on one.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import faiss
    from graphrag.graphrag_domain_indexer import load_domain_graph
    from graphrag.graphrag_indexer import embed_graph_nodes

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    sweep = sorted({int(x) for x in args.nprobe_sweep.split(",")})
    rng = np.random.default_rng(args.seed)

    queries, qvecs, embedder = [], None, None
    if args.synthetic:
        print(f"SYNTHETIC mode: {args.synthetic} queries/domain, "
              f"noise sigma={args.noise}, seed={args.seed}")
        print("Queries are perturbed node vectors, not real questions. "
              "Good for a tight\nrecall estimate; not a substitute for the "
              "gold-query run.\n")
    else:
        # Reuse the gold set so this reflects real traffic.
        with open(GOLD, encoding="utf-8") as f:
            gold = json.load(f)
        for dom, items in gold.items():
            if dom.startswith("_") or not isinstance(items, list):
                continue
            for it in items:
                q = it.get("query") if isinstance(it, dict) else None
                if q:
                    queries.append(q)
        if args.queries:
            queries = queries[:args.queries]
        if not queries:
            print("No queries found in the gold set. Aborting.")
            return
        print(f"Using {len(queries)} gold queries, k={args.k}")
        if len(queries) < 50:
            print(f"  NOTE: {len(queries)} queries is a small sample - a "
                  f"single query shifts recall\n  by "
                  f"{1.0/len(queries):.3f}. Cross-check with --synthetic.")
        print()
        print("Loading embedder...")
        from utils.embedder import Embedder
        embedder = Embedder()
        qvecs = embedder.model.encode(
            queries, normalize_embeddings=True).astype(np.float32)

    rows = []
    for domain in domains:
        print("=" * 78)
        print(domain)
        print("=" * 78)

        try:
            index, graph, nodes = load_domain_graph(domain)
        except FileNotFoundError as e:
            print(f"  [skip] {e}\n")
            continue

        n_nodes = index.ntotal
        dim = index.d
        is_ivf = hasattr(index, "nprobe")
        print(f"  nodes={n_nodes}  dim={dim}  "
              f"index={'IndexIVFFlat' if is_ivf else 'IndexFlatIP'}")

        if not is_ivf:
            print("  Already exact — no approximation to measure.\n")
            rows.append({"domain": domain, "nodes": n_nodes,
                         "exact_already": True})
            continue

        # Vectors straight from the production index: identical data in
        # both indexes, so the comparison isolates the approximation.
        t0 = time.perf_counter()
        vecs = vectors_from_index(index, n_nodes, dim)
        src = "reconstructed from index"
        if vecs is None:
            # Fallback: re-embed exactly the way the indexer does.
            if embedder is None:
                from utils.embedder import Embedder
                embedder = Embedder()
            print("  falling back to re-embedding via embed_graph_nodes()")
            _, vecs = embed_graph_nodes(graph, embedder)
            src = "re-embedded"
        recon_s = time.perf_counter() - t0
        print(f"  vectors {src} in {recon_s:.1f}s  shape={vecs.shape}")
        assert vecs.shape == (n_nodes, dim), \
            f"vector shape {vecs.shape} != ({n_nodes}, {dim})"

        # Synthetic queries are built per domain, from that domain's own
        # vectors, so they sit in the right region of the space.
        if args.synthetic:
            nq = min(args.synthetic, n_nodes)
            pick = rng.choice(n_nodes, nq, replace=False)
            qvecs = vecs[pick].copy()
            if args.noise > 0:
                qvecs = qvecs + rng.normal(
                    0, args.noise, qvecs.shape).astype(np.float32)
            qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
            qvecs = qvecs.astype(np.float32)
        n_queries = len(qvecs)

        # ── exact baseline ──────────────────────────────────
        t0 = time.perf_counter()
        flat = faiss.IndexFlatIP(dim)
        flat.add(vecs)
        flat_build = time.perf_counter() - t0

        flat.search(qvecs[:1], args.k)            # warm up
        t0 = time.perf_counter()
        _, flat_idx = flat.search(qvecs, args.k)
        flat_ms = (time.perf_counter() - t0) / n_queries * 1000

        n_clusters = getattr(index, "nlist", min(int(np.sqrt(n_nodes)), 128))
        prod_nprobe = min(n_clusters, 16)
        print(f"  build: exact {flat_build:.2f}s   "
              f"n_clusters={n_clusters}  production nprobe={prod_nprobe}")
        print(f"  query: exact {flat_ms:.3f} ms/query\n")

        print(f"  {'nprobe':>7}{'recall@k':>11}{'ms/query':>11}"
              f"{'speedup':>10}")
        print("  " + "-" * 45)

        prod_recall = None
        sweep_rows = []
        original_nprobe = index.nprobe
        for nprobe in sweep:
            if nprobe > n_clusters:
                continue
            index.nprobe = nprobe
            index.search(qvecs[:1], args.k)       # warm up
            t0 = time.perf_counter()
            _, ivf_idx = index.search(qvecs, args.k)
            ivf_ms = (time.perf_counter() - t0) / n_queries * 1000

            # recall@k = fraction of the exact top-k that IVF also finds
            overlaps = [len(set(a) & set(b)) / args.k
                        for a, b in zip(flat_idx, ivf_idx)]
            recall = statistics.mean(overlaps)
            speed = flat_ms / ivf_ms if ivf_ms > 0 else float("inf")

            tag = "  <- PRODUCTION" if nprobe == prod_nprobe else ""
            if nprobe == prod_nprobe:
                prod_recall = recall
            print(f"  {nprobe:>7}{recall:>11.3f}{ivf_ms:>11.3f}"
                  f"{speed:>9.1f}x{tag}")
            sweep_rows.append({"nprobe": nprobe, "recall": recall,
                               "ms": ivf_ms, "speedup": speed})
        index.nprobe = original_nprobe            # leave as we found it

        # How many queries lose their TOP result, not just tail ranks.
        index.nprobe = prod_nprobe
        _, ivf_idx = index.search(qvecs, args.k)
        top1_miss = sum(1 for a, b in zip(flat_idx, ivf_idx)
                        if a[0] != b[0])
        print(f"\n  top-1 differs from exact on "
              f"{top1_miss}/{n_queries} queries at production nprobe")
        index.nprobe = original_nprobe

        mem = n_nodes * dim * 4
        print(f"  vectors in memory: {human_bytes(mem)} "
              f"({n_nodes} x {dim} x float32)\n")

        rows.append({
            "domain": domain, "nodes": n_nodes, "dim": dim,
            "n_clusters": int(n_clusters), "prod_nprobe": int(prod_nprobe),
            "recall_at_prod": prod_recall, "top1_miss": top1_miss,
            "n_queries": n_queries, "exact_ms": flat_ms,
            "exact_build_s": flat_build, "bytes": mem,
            "sweep": sweep_rows,
        })

    # ── summary ─────────────────────────────────────────────
    measured = [r for r in rows if not r.get("exact_already")]
    print("=" * 78)
    print("SUMMARY - production configuration")
    print("=" * 78)
    print(f"  {'domain':24}{'nodes':>8}{'clusters':>10}{'nprobe':>8}"
          f"{'recall@k':>11}{'top1 miss':>11}{'exact ms':>10}")
    print("  " + "-" * 80)
    for r in measured:
        rec = "n/a" if r["recall_at_prod"] is None \
            else f"{r['recall_at_prod']:.3f}"
        print(f"  {r['domain']:24}{r['nodes']:>8}{r['n_clusters']:>10}"
              f"{r['prod_nprobe']:>8}{rec:>11}"
              f"{r['top1_miss']:>6}/{r['n_queries']:<4}{r['exact_ms']:>10.3f}")

    recs = [r["recall_at_prod"] for r in measured
            if r["recall_at_prod"] is not None]
    if recs:
        print(f"\n  mean recall@{args.k} at production nprobe: "
              f"{statistics.mean(recs):.3f}  "
              f"(worst {min(recs):.3f})")
    print(f"  total vectors in memory: "
          f"{human_bytes(sum(r['bytes'] for r in measured))}")

    if args.json:
        path = args.json if os.path.isabs(args.json) \
            else os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"k": args.k,
                       "mode": "synthetic" if args.synthetic else "gold",
                       "noise": args.noise if args.synthetic else None,
                       "seed": args.seed,
                       "domains": rows}, f, indent=2)
        print(f"\n  written: {path}")

    print("\n" + "-" * 78)
    print("  recall@k here = the fraction of the EXACT top-k that")
    print("  approximate search also returns. 1.000 means nothing was")
    print("  given up. Below ~0.90 means retrieval is missing seed nodes")
    print("  exact search would have found, and every downstream number")
    print("  (cov@node, cov@text, p@k) inherits that loss.")
    print()
    print("  'top1 miss' is the sharper figure: seed nodes are ranked,")
    print("  and the first one anchors graph traversal. A different top-1")
    print("  can change the whole neighbourhood that gets expanded.")
    print()
    print("  If recall is low, raise nprobe - the sweep shows what that")
    print("  costs in latency. At these graph sizes exact search may")
    print("  simply be affordable: check the exact ms/query column before")
    print("  assuming the approximation is needed at all.")


if __name__ == "__main__":
    main()
