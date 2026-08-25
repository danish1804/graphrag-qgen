# ============================================================
# scripts/attach_context.py
# Attach the retrieval context to an already-generated corpus.
#
# WHY THIS EXISTS
# ---------------
# The first corpus was generated before `retrieved_chunks` was saved.
# The question rubric needs that context to judge faithfulness and
# groundedness - the dimension we most want to compare against the human
# raters - so without it the corpus cannot support post-hoc judging.
#
# Regenerating is the obvious fix but the WRONG one: generation samples
# at temperature 0.7, so new questions would differ from the ones already
# built into the rater workbook. Retrieval, by contrast, is deterministic
# (embedding similarity + graph traversal, no sampling), so the context
# can be reconstructed exactly without touching a single question.
#
# Multi-domain items are skipped: their context is assembled by a
# separate merge path that cannot be faithfully replayed here. They are
# reported so the gap is visible rather than silent.
#
# Usage:
#   python scripts/attach_context.py
#   python scripts/attach_context.py --corpus outputs/question_corpus.json
# ============================================================
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "outputs", "question_corpus.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT)
    ap.add_argument("--out", default=None,
                    help="defaults to overwriting --corpus in place")
    args = ap.parse_args()
    out = args.out or args.corpus

    if not os.path.isfile(args.corpus):
        sys.exit(f"[!] corpus not found: {args.corpus}")

    payload = json.load(open(args.corpus, encoding="utf-8"))
    qs = payload.get("questions", [])
    if not qs:
        sys.exit("[!] corpus has no questions")

    todo = [q for q in qs if not q.get("retrieved_chunks")]
    print(f"{len(qs)} questions, {len(todo)} missing context")
    if not todo:
        print("nothing to do.")
        return

    from utils.config import cfg
    from utils.embedder import Embedder
    from graphrag.graphrag_domain_indexer import load_domain_graph
    from graphrag.graphrag_retriever import retrieve_graph_context

    print("Loading embedder (no Qwen needed)...")
    embedder = Embedder()

    cache = {}

    def graph_for(domain):
        if domain not in cache:
            cache[domain] = load_domain_graph(domain)
        return cache[domain]

    attached = skipped = failed = 0
    for q in todo:
        dom = q.get("domain")
        if not dom or dom == "multi_domain":
            skipped += 1
            continue
        diff = q.get("difficulty", "medium")
        dcfg = cfg["difficulty"].get(diff, cfg["difficulty"]["medium"])
        try:
            index, graph, nodes = graph_for(dom)
            res = retrieve_graph_context(
                query=q.get("learning_outcome", ""),
                index=index, graph=graph, nodes=nodes, embedder=embedder,
                top_k_nodes=dcfg["top_k_nodes"],
                neighbour_depth=dcfg["neighbour_depth"],
                max_context_chunks=dcfg["max_chunks"],
                seed_ratio=dcfg.get("seed_ratio", 0.6),
            )
            chunks = [c.get("text", "") for c in res.get("chunk_pool", [])]
            q["retrieved_chunks"] = chunks
            q["context"] = res.get("context", "")
            q["context_source"] = "reconstructed"
            attached += 1
            print(f"  {q['question_id']} {dom:22} {len(chunks)} chunks")
        except Exception as exc:
            failed += 1
            print(f"  {q['question_id']} FAILED: {exc}")

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {out}")
    print(f"  attached : {attached}")
    print(f"  skipped  : {skipped}  (multi-domain - merge path not replayable)")
    print(f"  failed   : {failed}")
    if skipped:
        print("\n  Multi-domain items can still be judged on context-free")
        print("  dimensions (clarity, standalone quality) but not on")
        print("  faithfulness or groundedness. Report that limitation.")


if __name__ == "__main__":
    main()
