# ============================================================
# scripts/generate_one.py
# Generate ONE exam question end-to-end from a learning outcome,
# on the current (rebuilt) domain indexes. Exercises the full
# pipeline: route -> retrieve -> generate question + answer -> judge.
#
# Usage:
#   python scripts/generate_one.py
#   python scripts/generate_one.py "Explain how paging maps virtual to physical memory" medium conceptual
#   python scripts/generate_one.py "<learning outcome>" [easy|medium|hard] [recall|conceptual|application|case_study]
#
# Loads Qwen 2.5 7B (+ LoRA) and the judge, so the first run takes
# a minute or two to load the models.
# ============================================================
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.graphrag_pipeline import GraphRAGPipeline

LEARNING_OUTCOME = (sys.argv[1] if len(sys.argv) > 1 else
    "Explain how normalisation reduces data redundancy in relational databases")
DIFFICULTY = sys.argv[2] if len(sys.argv) > 2 else "medium"
QTYPE      = sys.argv[3] if len(sys.argv) > 3 else "conceptual"


def show(label, value):
    print(f"{label}: {value}")


def main():
    print("Loading models (Qwen 2.5 7B + judge) — first load takes a minute...")
    p = GraphRAGPipeline()
    p.load_models()
    p.load_judge_model()

    print("\n" + "=" * 72)
    print("LEARNING OUTCOME:", LEARNING_OUTCOME)
    print(f"difficulty={DIFFICULTY}  question_type={QTYPE}  domain=auto")
    print("=" * 72)

    res = p.generate_with_routing(
        learning_objective=LEARNING_OUTCOME,
        difficulty=DIFFICULTY,
        question_type=QTYPE,
        domain=None,          # auto-route from the learning outcome
    )

    md = res.get("metadata", {})
    print("\n" + "-" * 72)
    print("GENERATED QUESTION:")
    print(" ", res.get("question", "(none)"))
    print("\nSAMPLE ANSWER:")
    print(" ", md.get("sample_answer", "(none)"))
    print("-" * 72)
    show("\nrouted domain", md.get("domain"))
    show("routing_confidence", md.get("routing_confidence"))
    show("sources", md.get("sources"))
    show("key_concepts", md.get("key_concepts"))
    show("judge_scores", json.dumps(md.get("judge_scores", {}), indent=2))
    print("\n(Reminder: judge_scores come from the SAME Qwen model that "
          "generated the question — the self-bias we still need to fix "
          "with an independent judge.)")


if __name__ == "__main__":
    main()
