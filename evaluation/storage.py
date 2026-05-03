# ============================================================
# evaluation/storage.py
# Two responsibilities:
#
# 1. Parse raw judge output into structured reasoning + scores.
#    The judge is prompted to emit:
#
#       REASONING:
#       <free text>
#
#       SCORES:
#       {"criterion_1": 4, "criterion_2": 5, ...}
#
#    This module extracts both parts robustly.
#
# 2. Save / load evaluation records as JSON on disk.
# ============================================================

import os
import re
import json
import uuid
from datetime import datetime
from pathlib import Path

import sys
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.config import cfg


# ------------------------------------------------------------
# Resolve the output directory for saved evaluations
# ------------------------------------------------------------
BASE_DIR = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

EVAL_DIR = os.path.join(
    BASE_DIR,
    cfg["judge"].get("eval_output_dir", "outputs/evaluations"),
)


# ============================================================
# 1. PARSER — extract reasoning + scores from judge output
# ============================================================
def parse_judge_output(raw_output: str, expected_keys: list) -> dict:
    """
    Parse the judge's raw text response into structured data.

    Args:
        raw_output:    Raw string from the judge model.
        expected_keys: List of score keys we expect (e.g.
                       ["relevance", "sufficiency", "noise"]).

    Returns:
        {
            "reasoning":      "<extracted reasoning text>",
            "scores":         {"criterion_1": 4, ...},
            "overall":        4.33,        # mean of scores
            "parse_status":   "ok" | "partial" | "failed",
            "parse_notes":    "<diagnostic info if not ok>",
            "raw_output":     "<original text for audit>"
        }

    Robustness:
        - Handles missing REASONING: or SCORES: labels.
        - Handles JSON wrapped in ```json ... ``` fences.
        - Handles missing keys (fills with None, marks partial).
        - Handles trailing commentary after the JSON block.
    """
    result = {
        "reasoning":    "",
        "scores":       {},
        "overall":      None,
        "parse_status": "ok",
        "parse_notes":  "",
        "raw_output":   raw_output,
    }

    # ---- Extract REASONING section ----
    reasoning_match = re.search(
        r"REASONING:\s*(.+?)(?=SCORES:|$)",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    else:
        # Fall back: everything before a JSON-like block is reasoning
        pre_json = re.split(r"\{", raw_output, maxsplit=1)[0].strip()
        result["reasoning"] = pre_json
        result["parse_notes"] += "No REASONING: label found. "

    # ---- Extract SCORES section — find the first JSON object ----
    # Strip markdown code fences if the model wrapped the JSON in them
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_output)
    cleaned = re.sub(r"```", "", cleaned)

    # Find the first balanced JSON object in the text
    json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)

    if not json_match:
        result["parse_status"] = "failed"
        result["parse_notes"]  += "No JSON object found in output."
        return result

    json_str = json_match.group(0)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Sometimes the model emits trailing commas or comments.
        # Try a cleanup pass.
        fixed = re.sub(r",\s*}", "}", json_str)
        fixed = re.sub(r"//.*", "", fixed)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            result["parse_status"] = "failed"
            result["parse_notes"]  += f"JSON decode error: {e}"
            return result

    # ---- Validate + coerce scores ----
    scores = {}
    for key in expected_keys:
        if key not in parsed:
            scores[key] = None
            result["parse_status"] = "partial"
            result["parse_notes"]  += f"Missing key: {key}. "
            continue

        val = parsed[key]
        # Coerce to int, clamp to 1-5 range
        try:
            val_int = int(round(float(val)))
            val_int = max(1, min(5, val_int))
            scores[key] = val_int
        except (TypeError, ValueError):
            scores[key] = None
            result["parse_status"] = "partial"
            result["parse_notes"]  += f"Invalid value for {key}: {val}. "

    result["scores"] = scores

    # Overall = mean of valid scores
    valid = [v for v in scores.values() if isinstance(v, int)]
    if valid:
        result["overall"] = round(sum(valid) / len(valid), 2)

    return result


# ============================================================
# 2. STORAGE — save / load JSON evaluation records
# ============================================================
def _ensure_eval_dir():
    """Create the evaluations directory if it does not exist."""
    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)


def make_eval_id() -> str:
    """Generate a unique evaluation id (timestamp + short uuid)."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"eval_{ts}_{short}"


def save_evaluation(record: dict, eval_id: str = None) -> str:
    """
    Save an evaluation record as a JSON file.

    Args:
        record:  Dictionary containing input, retrieval, question,
                 answer and metadata.
        eval_id: Optional id; generated automatically if not given.

    Returns:
        The full path to the saved file.
    """
    _ensure_eval_dir()

    if eval_id is None:
        eval_id = record.get("eval_id") or make_eval_id()
    record["eval_id"] = eval_id

    path = os.path.join(EVAL_DIR, f"{eval_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"[Storage] Evaluation saved: {path}")
    return path


def load_evaluation(eval_id: str) -> dict:
    """Load a single saved evaluation by id."""
    path = os.path.join(EVAL_DIR, f"{eval_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_evaluations() -> list:
    """List all saved evaluation ids, most recent first."""
    _ensure_eval_dir()
    files = [
        f.replace(".json", "")
        for f in os.listdir(EVAL_DIR)
        if f.endswith(".json")
    ]
    return sorted(files, reverse=True)


# ============================================================
# TEST — run this file directly to verify parsing works
#   python evaluation/storage.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STORAGE / PARSER TEST")
    print("=" * 60)

    # ---- Test 1: clean judge output ----
    sample_clean = """REASONING:
The chunks directly cover parameter sharing and convolution.
All three chunks are on-topic and complementary.
No irrelevant material is present.

SCORES:
{"relevance": 5, "sufficiency": 4, "noise": 5}
"""

    parsed = parse_judge_output(
        sample_clean,
        expected_keys=["relevance", "sufficiency", "noise"],
    )
    print("\n--- Test 1: clean output ---")
    print(f"Status : {parsed['parse_status']}")
    print(f"Scores : {parsed['scores']}")
    print(f"Overall: {parsed['overall']}")
    print(f"Reasoning preview: {parsed['reasoning'][:80]}...")

    # ---- Test 2: judge wraps JSON in markdown fences ----
    sample_fenced = """REASONING:
Decent coverage but one chunk is off-topic.

SCORES:
```json
{"relevance": 4, "sufficiency": 3, "noise": 2}
```
"""
    parsed = parse_judge_output(
        sample_fenced,
        expected_keys=["relevance", "sufficiency", "noise"],
    )
    print("\n--- Test 2: markdown-fenced JSON ---")
    print(f"Status : {parsed['parse_status']}")
    print(f"Scores : {parsed['scores']}")
    print(f"Overall: {parsed['overall']}")

    # ---- Test 3: missing key ----
    sample_missing = """REASONING:
Partial output — missing one score.

SCORES:
{"relevance": 5, "sufficiency": 4}
"""
    parsed = parse_judge_output(
        sample_missing,
        expected_keys=["relevance", "sufficiency", "noise"],
    )
    print("\n--- Test 3: missing key ---")
    print(f"Status : {parsed['parse_status']}")
    print(f"Notes  : {parsed['parse_notes']}")
    print(f"Scores : {parsed['scores']}")

    # ---- Test 4: save + load roundtrip ----
    print("\n--- Test 4: save/load roundtrip ---")
    fake_record = {
        "input": {"learning_objective": "test objective"},
        "retrieval": {"scores": {"relevance": 5}, "reasoning": "..."},
        "question":  {"generated": "What is X?"},
        "answer":    {"generated": "X is Y."},
    }
    eval_id = make_eval_id()
    path = save_evaluation(fake_record, eval_id)
    loaded = load_evaluation(eval_id)
    print(f"Roundtrip OK: {loaded['eval_id'] == eval_id}")

    print("\nStorage / parser test complete.")