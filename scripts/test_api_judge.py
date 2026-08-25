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
# utils.judge and evaluation.* are imported inside main(), AFTER the
# config guards, so a misconfiguration reports in a second instead of
# waiting for torch and transformers to load first.


def main():
    backend = cfg.get("judge", {}).get("backend")
    print(f"configured judge.backend = {backend!r}")
    if backend != "api":
        # Previously this only warned and then crashed inside
        # run_judge with 'NoneType has no attribute apply_chat_template',
        # because reuse_generator expects a Qwen model this script never
        # loads. Fail clearly instead.
        sys.exit("[!] This script tests the EXTERNAL judge, but "
                 f"judge.backend is {backend!r}.\n"
                 "    Set 'judge.backend: api' in config.yml, fill in "
                 "judge.api.base_url, and set the API key env var.")
    api = cfg["judge"]["api"]
    if not api.get("base_url"):
        sys.exit("[!] judge.api.base_url is empty - fill in one of the "
                 "presets in config.yml.")
    key_env = api.get("api_key_env", "JUDGE_API_KEY")
    if not os.getenv(key_env):
        sys.exit(f"[!] environment variable {key_env} is not set.\n"
                 f"    Windows:  set {key_env}=your-key-here")
    print(f"endpoint = {api['base_url']}   model = {api['model']}")

    from utils.judge import build_judge
    from evaluation.rubrics import (
        RETRIEVAL_SYSTEM_PROMPT, build_retrieval_prompt,
        RETRIEVAL_SCORE_KEYS)
    from evaluation.storage import parse_judge_output

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
