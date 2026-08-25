# ============================================================
# scripts/test_api_judge.py
# Verify the EXTERNAL API judge works in isolation — one rubric
# call, no Qwen, no generation, no pipeline. Use this to confirm
# the endpoint/model/key are wired before running the full system.
#
# Setup:
#   1. config.yml -> judge.backend: api  (+ api.base_url / api.model
#      / api.api_version for Azure)
#   2. pip install openai
#   3. set the key:  set JUDGE_API_KEY=...   (Windows)
#
# Run:
#   python scripts/test_api_judge.py
#
# What goes to the API: only a tiny sample context chunk + the
# retrieval rubric — nothing from your corpus, code, or graphs.
# ============================================================
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import cfg
from utils.judge import build_judge
from evaluation.rubrics import (
    RETRIEVAL_SYSTEM_PROMPT, build_retrieval_prompt, RETRIEVAL_SCORE_KEYS)
from evaluation.storage import parse_judge_output


def main():
    backend = cfg.get("judge", {}).get("backend")
    print(f"configured judge.backend = {backend!r}")
    if backend != "api":
        print("NOTE: set 'judge.backend: api' in config.yml to test the "
              "external judge (currently it will build the local backend).")

    judge = build_judge(cfg)
    print(f"judge name = {judge.name!r} | backend = {judge.backend!r}")

    # A tiny, non-sensitive sample — this is all that leaves the machine.
    chunks = ["An inner join connects two tables on a column with the same "
              "data type. Only the rows where the column values match are "
              "returned in the result."]
    prompt = build_retrieval_prompt("Explain how an inner join works", chunks)

    print("\nCalling the judge (one rubric call)...")
    raw = judge.generate(RETRIEVAL_SYSTEM_PROMPT, prompt)
    print("--- raw output (first 600 chars) ---")
    print(raw[:600])

    parsed = parse_judge_output(raw, RETRIEVAL_SCORE_KEYS)
    print(f"\nparsed scores : {parsed['scores']}")
    print(f"overall       : {parsed['overall']}   (status: {parsed['parse_status']})")
    if parsed["scores"] and parsed["parse_status"] != "failed":
        print("\n[OK] API judge reachable and returned scores.")
    else:
        print("\n[!] No scores parsed — check base_url / model / key / "
              "api_version.")


if __name__ == "__main__":
    main()
