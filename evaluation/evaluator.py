# ============================================================
# evaluation/evaluator.py
# LLM-as-Judge orchestrator.
#
# This module EVALUATES outputs produced by the main pipeline.
# It does NOT generate anything and does NOT touch FAISS or
# the graph — all inputs are passed in as arguments.
#
# Integration contract:
#   The caller (main_gr.py or any test harness) runs the
#   generation pipeline, then hands the outputs to
#   evaluate_generation() which returns the full evaluation
#   record (also saved to disk as JSON by default).
# ============================================================

# ---- Make project root importable ----
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import datetime

from utils.config import cfg
from utils.judge import load_judge, unload_model, run_judge

from evaluation.rubrics import (
    RETRIEVAL_SYSTEM_PROMPT, build_retrieval_prompt, RETRIEVAL_SCORE_KEYS,
    QUESTION_SYSTEM_PROMPT,  build_question_prompt,  QUESTION_SCORE_KEYS,
    ANSWER_SYSTEM_PROMPT,    build_answer_prompt,    ANSWER_SCORE_KEYS,
)
from evaluation.storage import (
    parse_judge_output,
    save_evaluation,
    make_eval_id,
)


# ============================================================
# Public entrypoint — evaluate already-generated outputs
# ============================================================
def evaluate_generation(
    learning_objective: str,
    retrieved_chunks:   list,
    question:           str,
    answer:             str,
    difficulty:         str = "medium",
    question_type:      str = "conceptual",
    save:               bool = True,
    extra_metadata:     dict = None,
    judge_model=None,          # ← NEW
    judge_tokenizer=None,      # ← NEW
) -> dict:
    """
    Run all three LLM-as-Judge evaluations on outputs produced
    by the main generation pipeline.

    If judge_model and judge_tokenizer are provided, they are
    reused (no load/unload). This is the mode used by the
    FastAPI server. If not provided, the judge is loaded and
    unloaded for this single call (standalone mode).
    """
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks is empty — nothing to evaluate.")
    if not question or not answer:
        raise ValueError("Both question and answer are required.")

    eval_id = make_eval_id()
    started = datetime.now().isoformat(timespec="seconds")
    context = "\n\n".join(retrieved_chunks)

    # Decide whether to manage judge lifecycle internally
    owns_judge = (judge_model is None or judge_tokenizer is None)
    if owns_judge:
        judge_model, judge_tokenizer, _ = load_judge()

    try:
        print(f"\n[Evaluator] JUDGING — {eval_id}")

        # 1. Retrieval quality
        raw = run_judge(
            judge_model, judge_tokenizer,
            RETRIEVAL_SYSTEM_PROMPT,
            build_retrieval_prompt(learning_objective, retrieved_chunks),
        )
        retrieval_eval = parse_judge_output(raw, RETRIEVAL_SCORE_KEYS)
        print(f"[Evaluator] Retrieval: {retrieval_eval['scores']} "
              f"(overall={retrieval_eval['overall']})")

        # 2. Question quality
        raw = run_judge(
            judge_model, judge_tokenizer,
            QUESTION_SYSTEM_PROMPT,
            build_question_prompt(context, question, difficulty, question_type),
        )
        question_eval = parse_judge_output(raw, QUESTION_SCORE_KEYS)
        print(f"[Evaluator] Question:  {question_eval['scores']} "
              f"(overall={question_eval['overall']})")

        # 3. Answer quality
        raw = run_judge(
            judge_model, judge_tokenizer,
            ANSWER_SYSTEM_PROMPT,
            build_answer_prompt(context, question, answer),
        )
        answer_eval = parse_judge_output(raw, ANSWER_SCORE_KEYS)
        print(f"[Evaluator] Answer:    {answer_eval['scores']} "
              f"(overall={answer_eval['overall']})")

    finally:
        if owns_judge:
            unload_model(judge_model, judge_tokenizer)

    # ---- Build record (same as before) ----
    record = {
        "eval_id":         eval_id,
        "timestamp":       started,
        "generator_model": cfg["model"]["llm_name"],
        "judge_model":     cfg["judge"]["judge_name"],
        "input": {
            "learning_objective": learning_objective,
            "difficulty":         difficulty,
            "question_type":      question_type,
        },
        "retrieval": {
            "chunks":       retrieved_chunks,
            "scores":       retrieval_eval["scores"],
            "overall":      retrieval_eval["overall"],
            "reasoning":    retrieval_eval["reasoning"],
            "parse_status": retrieval_eval["parse_status"],
        },
        "question": {
            "generated":    question,
            "scores":       question_eval["scores"],
            "overall":      question_eval["overall"],
            "reasoning":    question_eval["reasoning"],
            "parse_status": question_eval["parse_status"],
        },
        "answer": {
            "generated":    answer,
            "scores":       answer_eval["scores"],
            "overall":      answer_eval["overall"],
            "reasoning":    answer_eval["reasoning"],
            "parse_status": answer_eval["parse_status"],
        },
    }

    if extra_metadata:
        record["metadata"] = extra_metadata

    if save:
        save_evaluation(record, eval_id)

    return record


# ============================================================
# TEST — judge-only smoke test with hardcoded realistic inputs.
# No FAISS, no generator loading. Just verifies that the judge
# loads, runs all three rubric prompts, and parses correctly.
#
#   python evaluation/evaluator.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATOR — JUDGE-ONLY SMOKE TEST")
    print("=" * 60)

    # Hardcoded realistic example (taken from a typical CNN lecture).
    # Replace these later with real outputs from your pipeline.
    sample_chunks = [
        "Parameter sharing is a key idea behind convolutional neural networks. "
        "Instead of learning a separate weight for every pixel, a single filter "
        "is applied across all spatial locations of the input. This drastically "
        "reduces the number of parameters compared to a fully connected layer.",

        "Convolution uses the same set of weights (the kernel) at every position "
        "in the input. Because the kernel is much smaller than the input image, "
        "and the same kernel is reused everywhere, CNNs have far fewer parameters "
        "than densely connected networks for image data.",

        "Another consequence of parameter sharing is translation equivariance: "
        "if a feature appears in one part of the image, the same filter will "
        "detect it in any other location as well.",
    ]

    sample_question = (
        "Explain how parameter sharing in convolutional neural networks reduces "
        "the number of trainable parameters compared to fully connected networks."
    )

    sample_answer = (
        "In convolutional neural networks, parameter sharing means that a single "
        "filter's weights are reused across every spatial location of the input, "
        "rather than learning separate weights for each input position. In a fully "
        "connected layer, every input unit connects to every output unit with its "
        "own weight, producing a parameter count proportional to the product of "
        "the two layer sizes. By contrast, a convolutional filter has a fixed, "
        "small number of weights determined only by the kernel size and channel "
        "count, regardless of the spatial size of the input. This weight reuse "
        "dramatically lowers the total parameter count and also provides "
        "translation equivariance for the detected features."
    )

    record = evaluate_generation(
        learning_objective="What is parameter sharing in CNNs?",
        retrieved_chunks=sample_chunks,
        question=sample_question,
        answer=sample_answer,
        difficulty="medium",
        question_type="conceptual",
    )

    print("\nJudge-only smoke test complete. JSON saved under outputs/evaluations/.")