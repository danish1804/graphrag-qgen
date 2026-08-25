# ============================================================
# scripts/judge_corpus.py
# Score the FROZEN question corpus with the independent judge.
#
# This is the second half of the calibration design. The corpus was
# generated with the judge OFF, so humans see the natural quality
# distribution. Judge scores are added here, afterwards, over the same
# fixed questions. Two consequences worth stating:
#
#   * Humans and judge score IDENTICAL questions, joined on question_id,
#     so the comparison is exact rather than approximate.
#   * Changing judge later re-runs only this script. No human label is
#     invalidated and no question is regenerated.
#
# ONE VARIABLE AT A TIME: this uses the EXISTING rubrics unchanged. The
# roadmap recommends a 1-10 scale over the current 1-5, but changing the
# scale and the judge model together would confound the two effects -
# any score shift could be either. Swap the model first, measure, then
# consider the scale separately.
#
# Setup:
#   config.yml -> judge.backend: api  (+ the api block)
#   set JUDGE_API_KEY=...
#   pip install openai
#
# Usage:
#   python scripts/judge_corpus.py --limit 3      # cheap smoke test
#   python scripts/judge_corpus.py                # full corpus
# ============================================================
import os
import sys
import json
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "outputs", "question_corpus.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT)
    ap.add_argument("--out", default=None, help="default: in place")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N (0 = all)")
    ap.add_argument("--skip-answer", action="store_true",
                    help="score questions only, halving API calls")
    args = ap.parse_args()
    out = args.out or args.corpus

    # Check the backend BEFORE the heavy imports, so a misconfiguration
    # produces a clear message rather than a torch/transformers traceback.
    from utils.config import cfg

    backend = cfg.get("judge", {}).get("backend")
    if backend == "reuse_generator":
        sys.exit("[!] judge.backend is 'reuse_generator' - the generator "
                 "would be grading its own output, which is the bias this "
                 "whole exercise exists to remove.\n"
                 "    Set judge.backend: api in config.yml first.")
    if backend == "api" and not cfg["judge"]["api"].get("base_url"):
        sys.exit("[!] judge.api.base_url is empty. Fill in one of the "
                 "presets in config.yml (OpenRouter / Azure / vLLM).")
    key_env = cfg["judge"]["api"].get("api_key_env", "JUDGE_API_KEY")
    if backend == "api" and not os.getenv(key_env):
        sys.exit(f"[!] environment variable {key_env} is not set.")

    from utils.judge import build_judge
    from evaluation.rubrics import (
        QUESTION_SYSTEM_PROMPT, build_question_prompt, QUESTION_SCORE_KEYS,
        ANSWER_SYSTEM_PROMPT, build_answer_prompt, ANSWER_SCORE_KEYS)
    from evaluation.storage import parse_judge_output

    payload = json.load(open(args.corpus, encoding="utf-8"))
    qs = payload.get("questions", [])
    todo = qs[:args.limit] if args.limit else qs

    judge = build_judge(cfg)
    print(f"judge: {judge.name!r}  backend={judge.backend!r}")
    print(f"scoring {len(todo)} of {len(qs)} questions\n")

    scored = failed = no_context = 0
    t0 = time.time()

    for i, q in enumerate(todo, start=1):
        qid = q.get("question_id")
        context = q.get("context") or "\n\n".join(
            q.get("retrieved_chunks") or [])
        if not context:
            no_context += 1
            context = "(retrieval context unavailable for this item)"

        try:
            raw = judge.generate(
                QUESTION_SYSTEM_PROMPT,
                build_question_prompt(context, q.get("question", ""),
                                      q.get("difficulty", "medium"),
                                      q.get("question_type", "conceptual")))
            parsed = parse_judge_output(raw, QUESTION_SCORE_KEYS)
            entry = {
                "question_overall": parsed.get("overall"),
                "question_scores": parsed.get("scores"),
                "parse_status": parsed.get("parse_status"),
            }

            if not args.skip_answer and q.get("sample_answer"):
                raw_a = judge.generate(
                    ANSWER_SYSTEM_PROMPT,
                    build_answer_prompt(context, q.get("question", ""),
                                        q.get("sample_answer", "")))
                pa = parse_judge_output(raw_a, ANSWER_SCORE_KEYS)
                entry["answer_overall"] = pa.get("overall")
                entry["answer_scores"] = pa.get("scores")

            entry["judge_model"] = judge.name
            entry["judge_backend"] = judge.backend
            entry["context_available"] = bool(q.get("context")
                                              or q.get("retrieved_chunks"))
            q["independent_judge"] = entry
            scored += 1
            print(f"  [{i:>3}/{len(todo)}] {qid} "
                  f"q={entry['question_overall']} "
                  f"a={entry.get('answer_overall')}")
        except Exception as exc:
            failed += 1
            print(f"  [{i:>3}/{len(todo)}] {qid} FAILED: {exc}")

    payload["independent_judge_model"] = judge.name
    payload["independent_judge_backend"] = judge.backend
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    vals = [q["independent_judge"]["question_overall"] for q in todo
            if q.get("independent_judge")
            and q["independent_judge"].get("question_overall") is not None]
    print(f"\nWrote {out}   ({time.time() - t0:.0f}s)")
    print(f"  scored      : {scored}")
    print(f"  failed      : {failed}")
    print(f"  no context  : {no_context}  (multi-domain items)")
    if vals:
        vals_sorted = sorted(vals)
        print(f"  question_overall: min {min(vals):.2f}  "
              f"median {vals_sorted[len(vals) // 2]:.2f}  max {max(vals):.2f}")
        print("\n  A healthy calibration corpus should show SPREAD here. If "
              "every score is near the maximum, the judge is not "
              "discriminating and will not agree with human raters who do.")
    print("\nNext: python scripts/build_eval_sheet.py "
          "--corpus outputs/question_corpus.json")
    print("      (populates System_Key - keep it closed until labelling ends)")


if __name__ == "__main__":
    main()
