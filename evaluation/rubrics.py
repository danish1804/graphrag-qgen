# ============================================================
# evaluation/rubrics.py
# Defines the prompts and rubrics used by the LLM-as-Judge
# for three evaluations:
#
#   1. Retrieval quality  — are the chunks relevant/sufficient?
#   2. Question quality   — is the generated question good?
#   3. Answer quality     — is the generated answer good?
#
# Design pattern: "Reason-then-score"
#   The judge is forced to write step-by-step reasoning BEFORE
#   emitting JSON scores. This reduces score inflation and
#   produces more reliable evaluations (Zheng et al., 2023).
# ============================================================

RUBRIC_VERSION = "1.0.0"


# ------------------------------------------------------------
# 1. RETRIEVAL QUALITY
# ------------------------------------------------------------
RETRIEVAL_SYSTEM_PROMPT = """You are an expert evaluator of retrieval systems for educational content.
You judge whether retrieved text chunks are relevant and sufficient to support generating an exam question on a given learning objective.

You are strict but fair. You do NOT invent information. You evaluate ONLY what is shown to you.
You always respond in the exact format requested."""


RETRIEVAL_USER_TEMPLATE = """Evaluate the quality of the retrieved chunks below for the given learning objective.

LEARNING OBJECTIVE:
{learning_objective}

RETRIEVED CHUNKS:
{chunks_formatted}

RUBRIC (score each criterion from 1 to 5):

1. RELEVANCE — How directly do the chunks relate to the learning objective?
   - 1: Chunks are unrelated to the objective.
   - 3: Chunks are partially relevant; some cover the topic, others do not.
   - 5: All chunks directly address the learning objective.

2. SUFFICIENCY — Is there enough information in the chunks to generate a good exam question?
   - 1: Insufficient — key concepts missing entirely.
   - 3: Some useful information, but gaps remain.
   - 5: Fully sufficient — all needed concepts are present.

3. NOISE — How free are the chunks from irrelevant/distracting content? (HIGHER = LESS noise)
   - 1: Mostly noise; very little usable content.
   - 3: Moderate noise; some irrelevant passages mixed in.
   - 5: Clean — almost no irrelevant content.

INSTRUCTIONS:
- First, write your REASONING: analyze each criterion against the chunks in 2-3 sentences each.
- Then output SCORES as a valid JSON object.
- Do not wrap the JSON in markdown code fences.

Output format (exactly):

REASONING:
<your analysis here>

SCORES:
{{"relevance": <1-5>, "sufficiency": <1-5>, "noise": <1-5>}}
"""


# ------------------------------------------------------------
# 2. QUESTION QUALITY
# ------------------------------------------------------------
QUESTION_SYSTEM_PROMPT = """You are an expert evaluator of exam questions for IT/computer science education.
You assess whether a generated exam question is well-formed, faithful to its source material, and appropriately calibrated.

You are strict but fair. You only evaluate what is shown. You never invent information.
You always respond in the exact format requested."""


QUESTION_USER_TEMPLATE = """Evaluate the generated exam question below.

SOURCE CONTEXT:
{context}

GENERATED QUESTION:
{question}

REQUESTED PARAMETERS:
- Difficulty : {difficulty}
- Question type : {question_type}

RUBRIC (score each criterion from 1 to 5):

1. FAITHFULNESS — Can the question be fully answered using ONLY the source context?
   - 1: Requires knowledge not in the context.
   - 3: Mostly answerable but requires some outside knowledge.
   - 5: Completely answerable from context alone.

2. CLARITY — Is the question clear, unambiguous, and well-phrased?
   - 1: Confusing or ambiguous.
   - 3: Understandable but awkward or vague.
   - 5: Crisp, precise, and easy to understand.

3. DIFFICULTY_ALIGNMENT — Does the question match the requested difficulty level?
   (easy = recall/basic understanding; medium = explanation/conceptual; hard = application/analysis)
   - 1: Severely mismatched (e.g. trivial recall labelled hard).
   - 3: Slightly off target.
   - 5: Perfectly calibrated to the requested difficulty.

4. TYPE_ALIGNMENT — Does the question match the requested type?
   (recall = define/list/state; conceptual = explain/compare; application = design/analyse/apply)
   - 1: Wrong type entirely.
   - 3: Ambiguous type or partial fit.
   - 5: Clearly matches the requested type.

5. STANDALONE_QUALITY — Is the question self-contained, without referencing "the context" or "the passage"?
   - 1: Explicitly refers to the passage/context/text.
   - 3: Implicit but noticeable dependence on the source.
   - 5: Reads as a genuine exam question a student could see.

INSTRUCTIONS:
- First, write your REASONING: analyze each criterion against the question in 2-3 sentences each.
- Then output SCORES as a valid JSON object.
- Do not wrap the JSON in markdown code fences.

Output format (exactly):

REASONING:
<your analysis here>

SCORES:
{{"faithfulness": <1-5>, "clarity": <1-5>, "difficulty_alignment": <1-5>, "type_alignment": <1-5>, "standalone_quality": <1-5>}}
"""


# ------------------------------------------------------------
# 3. ANSWER QUALITY
# ------------------------------------------------------------
ANSWER_SYSTEM_PROMPT = """You are an expert evaluator of sample answers for university-level IT/computer science exam questions.
You assess whether a generated answer is correct, complete, grounded in the source material, and written in appropriate academic style.

You are strict but fair. You only evaluate what is shown. You never invent information.
You always respond in the exact format requested."""


ANSWER_USER_TEMPLATE = """Evaluate the generated sample answer below.

SOURCE CONTEXT:
{context}

EXAM QUESTION:
{question}

GENERATED ANSWER:
{answer}

RUBRIC (score each criterion from 1 to 5):

1. CORRECTNESS — Is the answer factually correct according to the source context?
   - 1: Contains clear factual errors.
   - 3: Mostly correct with minor inaccuracies.
   - 5: Fully correct and consistent with the context.

2. COMPLETENESS — Does the answer cover the key points needed to fully address the question?
   - 1: Misses most required points.
   - 3: Covers main idea but omits important details.
   - 5: Addresses all key points relevant to the question.

3. GROUNDEDNESS — Is the answer grounded in the source context (no hallucinations)?
   - 1: Contains significant information not present in the context.
   - 3: Mostly grounded but with a few unsupported additions.
   - 5: Entirely grounded in the provided context.

4. ACADEMIC_TONE — Is the language appropriate for university-level writing?
   - 1: Casual or inappropriate tone; uses bullet points where prose is expected.
   - 3: Acceptable but inconsistent.
   - 5: Precise, academic prose suitable for a university setting.

INSTRUCTIONS:
- First, write your REASONING: analyze each criterion against the answer in 2-3 sentences each.
- Then output SCORES as a valid JSON object.
- Do not wrap the JSON in markdown code fences.

Output format (exactly):

REASONING:
<your analysis here>

SCORES:
{{"correctness": <1-5>, "completeness": <1-5>, "groundedness": <1-5>, "academic_tone": <1-5>}}
"""


# ------------------------------------------------------------
# Helpers — format chunks cleanly for the judge prompt
# ------------------------------------------------------------
def format_chunks_for_prompt(chunks: list) -> str:
    """
    Render a list of retrieved chunks as a numbered block for the prompt.
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Chunk {i}]")
        lines.append(chunk.strip())
        lines.append("")  # blank line between chunks
    return "\n".join(lines).strip()


# ------------------------------------------------------------
# Builders — produce the final user-prompt string for each eval
# ------------------------------------------------------------
def build_retrieval_prompt(learning_objective: str, chunks: list) -> str:
    return RETRIEVAL_USER_TEMPLATE.format(
        learning_objective=learning_objective,
        chunks_formatted=format_chunks_for_prompt(chunks),
    )


def build_question_prompt(context: str, question: str,
                          difficulty: str, question_type: str) -> str:
    extra = ""
    if question_type == "case_study":
        extra = (
            "\nNote: case_study output should include a SCENARIO block "
            "and a QUESTION line grounded in that scenario.\n"
        )
    return QUESTION_USER_TEMPLATE.format(
        context=context,
        question=question + extra,
        difficulty=difficulty,
        question_type=question_type,
    )


def build_answer_prompt(context: str, question: str, answer: str) -> str:
    return ANSWER_USER_TEMPLATE.format(
        context=context,
        question=question,
        answer=answer,
    )


# ------------------------------------------------------------
# Score keys — used by the parser/storage downstream
# ------------------------------------------------------------
RETRIEVAL_SCORE_KEYS = ["relevance", "sufficiency", "noise"]
QUESTION_SCORE_KEYS  = ["faithfulness", "clarity", "difficulty_alignment",
                        "type_alignment", "standalone_quality"]
ANSWER_SCORE_KEYS    = ["correctness", "completeness", "groundedness",
                        "academic_tone"]