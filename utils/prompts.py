"""
Prompt templates for question and answer generation.

Prompt text lives here; decoding parameters live in config.yml.
"""

from __future__ import annotations

try:
    from utils.config import cfg
except ImportError:
    from config import cfg

from utils.question_type_prompts import instructions_for_question_type

PROMPT_VERSION = cfg.get("prompts", {}).get("version", "1.1.0")

QUESTION_SYSTEM_PROMPT = """You are an expert exam question generator for IT education.
Your job is to generate exactly ONE high-quality exam question.

Rules:
- The question MUST be answerable using ONLY the provided context
- Do NOT introduce concepts not present in the context
- Return ONLY the question - no preamble, no explanation, no answer
- The question should be clear and unambiguous
- Do NOT generate multiple choice questions
- Do NOT generate A) B) C) D) options"""

ANSWER_SYSTEM_PROMPT = (
    "You are an expert university educator. "
    "Generate clear, accurate sample answers. "
    "Do not reference 'the context' or 'as provided'. "
    "Write as if answering from knowledge directly."
)

ANSWER_DEPTH_BY_DIFFICULTY = {
    "easy": "Write a brief, direct answer in 2-3 sentences.",
    "medium": "Write a clear answer in 3-5 sentences covering the key points.",
    "hard": (
        "Write a detailed answer in 5-8 sentences covering concepts, "
        "relationships and examples."
    ),
}


def build_question_messages(
    *,
    context: str,
    learning_objective: str,
    difficulty: str,
    question_type: str,
) -> list[dict[str, str]]:
    type_block = instructions_for_question_type(question_type)
    user_prompt = f"""Context from course material:
{context}

Learning objective: {learning_objective}
Difficulty: {difficulty}

{type_block}"""
    return [
        {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_answer_messages(
    *,
    context: str,
    question: str,
    difficulty: str,
) -> list[dict[str, str]]:
    depth_instruction = ANSWER_DEPTH_BY_DIFFICULTY.get(
        difficulty,
        ANSWER_DEPTH_BY_DIFFICULTY["medium"],
    )
    user_prompt = f"""Using the context below, write a sample answer for the exam question.

Context:
{context}

Question:
{question}

Instructions:
- {depth_instruction}
- Base your answer only on the provided context
- Use precise academic language appropriate for university level
- Do not repeat the question in your answer
- Do not use bullet points — write in prose

Sample Answer:"""
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
