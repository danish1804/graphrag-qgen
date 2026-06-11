"""
Per-question-type prompt fragments for exam generation.
"""

from __future__ import annotations

CASE_STUDY_INSTRUCTIONS = """Question type: case_study

You MUST follow this structure in your output:
1. SCENARIO — Write a short realistic scenario (3-6 sentences) using ONLY facts from the context.
   Use concrete details (names, numbers, system states) implied by the material.
2. QUESTION — On a new line after the scenario, ask ONE exam question that requires the student
   to apply course concepts to that scenario.

Example format:
SCENARIO: A company runs five database transactions concurrently on a shared inventory table...
QUESTION: Given this scenario, how would you design a locking strategy to prevent deadlock?

Rules for case_study:
- The scenario must be answerable from the context alone
- Do NOT add external technologies or facts not supported by the context
- Return the SCENARIO block and QUESTION line only — no answer, no multiple choice"""


def instructions_for_question_type(question_type: str) -> str:
    """Extra user-prompt instructions beyond the base question request."""
    if question_type == "case_study":
        return CASE_STUDY_INSTRUCTIONS
    return (
        f"Question type: {question_type}\n"
        "Generate one exam question based strictly on the context above."
    )
