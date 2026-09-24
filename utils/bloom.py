"""
Bloom's revised taxonomy support for question generation.

WHY THIS EXISTS
---------------
`config.yml` has carried a `bloom_level` against every question type since
the original build, but until now **no code read it**. The cognitive level
was an assertion in a config file, not something the system acted on: the
generation prompt said only "Question type: conceptual" and left the
cognitive demand entirely to the model.

This module turns that dead field into a prompt instruction, and provides
a deterministic verb-based check so we can measure whether the requested
level is actually reached rather than assuming it.

TWO THINGS THIS DOES NOT CLAIM
------------------------------
1. Asking for a level is not the same as hitting it. `classify_by_verb()`
   exists precisely so the gap can be measured.
2. Verb-based classification is a coarse proxy. Human raters agree on
   Bloom level at only Cohen's kappa 0.12-0.60, and GPT-4 agreed with
   teacher classification ~65-69% with roughly half its "Analysing"
   labels rated "Evaluating" by a human. Treat the output as a signal,
   not a label. Report the coarse split (remember/understand versus
   apply-and-above) rather than fine-grained levels.

Reference: Anderson & Krathwohl (2001), revised Bloom's taxonomy.
"""

from __future__ import annotations

import re

try:
    from utils.config import cfg
except ImportError:                                  # standalone import
    from config import cfg


# ── The six levels of the revised taxonomy ──────────────────
# `verbs` are the action verbs characteristic of the level and are used
# both to steer generation and to classify the result.
BLOOM_LEVELS: dict[int, dict] = {
    1: {
        "name": "Remember",
        "demand": "recall or recognise facts, terms and basic concepts",
        "verbs": ["define", "list", "state", "identify", "name", "recall",
                  "label", "recognise", "recognize", "match", "select"],
        # Interrogative openers are level-1 ONLY when they open the
        # question. Matched anywhere they produce false positives:
        # "When a virtual machine executes an interrupt, why must the
        # VMM take control..." is a level-2 question that was being
        # classified level 1 on the leading "when".
        "openers": ["what is", "what are", "what does", "what was",
                    "which of", "who ", "when did", "where is",
                    # Added after two corpus questions came back
                    # unclassified: "...what role does X play" and
                    # "...what happens when". Both are recall phrasings
                    # the original list missed.
                    "what role", "what happens", "what type", "what kind",
                    "what purpose", "what function", "how many"],
    },
    2: {
        "name": "Understand",
        "demand": "explain ideas or concepts in the student's own words",
        "verbs": ["explain", "describe", "summarise", "summarize",
                  "interpret", "classify", "discuss", "illustrate",
                  "paraphrase", "clarify"],
        "openers": ["why does", "why must", "why is", "why are",
                    "how does", "how do ", "how is", "how are"],
    },
    3: {
        "name": "Apply",
        "demand": "use knowledge in a new or concrete situation",
        "verbs": ["apply", "implement", "solve", "demonstrate", "calculate",
                  "execute", "use", "compute", "determine", "show how",
                  "modify", "operate"],
    },
    4: {
        "name": "Analyse",
        "demand": ("break material into parts and show how the parts "
                   "relate, distinguish between them, or trace causes"),
        "verbs": ["analyse", "analyze", "differentiate", "distinguish",
                  "compare", "contrast", "examine", "deconstruct",
                  "categorise", "categorize", "investigate", "attribute"],
    },
    5: {
        "name": "Evaluate",
        "demand": "make and defend a judgement against explicit criteria",
        "verbs": ["evaluate", "justify", "critique", "defend", "appraise",
                  "argue", "assess", "judge", "recommend", "prioritise",
                  "prioritize", "which is better"],
    },
    6: {
        "name": "Create",
        "demand": "combine elements into a new coherent whole or design",
        "verbs": ["design", "construct", "formulate", "devise", "compose",
                  "propose", "develop", "plan", "produce", "generate"],
    },
}

# Verbs that are widely discouraged in learning-outcome and item writing
# because they are not observable and cannot be assessed.
UNASSESSABLE_VERBS = ["understand", "know", "appreciate", "be aware",
                      "be familiar", "learn", "grasp"]


def bloom_level_for(question_type: str) -> int | None:
    """The configured Bloom level for a question type, or None.

    Reads `question_types.<type>.bloom_level` from config.yml. Returns
    None rather than a default so the caller can tell "not configured"
    from "configured as level 1".
    """
    qt = (cfg.get("question_types") or {}).get(question_type) or {}
    level = qt.get("bloom_level")
    return level if level in BLOOM_LEVELS else None


def bloom_block(question_type: str) -> str:
    """Prompt fragment stating the target cognitive level.

    Returns "" when the type has no configured level, so callers can
    concatenate unconditionally.

    The prohibition on lower-level verbs is the part that actually
    bites. Without it a request for "Apply" is frequently answered with
    "Define X", because a definition question is easier to produce from
    retrieved context than an application one.
    """
    level = bloom_level_for(question_type)
    if level is None:
        return ""

    spec = BLOOM_LEVELS[level]
    preferred = ", ".join(spec["verbs"][:6])

    lines = [
        f"Cognitive level: {level} - {spec['name']} (Bloom's revised taxonomy)",
        f"The question must require the student to {spec['demand']}.",
        f"Prefer verbs such as: {preferred}.",
    ]

    # Anything at Apply or above must not collapse into recall.
    if level >= 3:
        lower = []
        for lv in range(1, min(level, 3)):
            lower.extend(BLOOM_LEVELS[lv]["verbs"][:4])
        lines.append(
            "Do NOT write a question that only asks the student to "
            f"{', '.join(sorted(set(lower)))}. Recalling or restating the "
            "material is below the required level."
        )

    lines.append(
        "Do not use unassessable verbs such as "
        f"{', '.join(UNASSESSABLE_VERBS[:4])}."
    )
    return "\n".join(lines)


# ── Measurement ─────────────────────────────────────────────
_WORD = re.compile(r"[a-z]+")


def classify_by_verb(question: str) -> tuple[int | None, str | None]:
    """Infer the Bloom level a question's wording implies.

    Deterministic and rule-based by design: on item-writing flaws,
    rule-based detection outperformed GPT-4 (91% vs 79%), and the
    project standard is rules for structural checks, LLM for semantic
    judgement.

    Scans for the first matching level verb, scanning HIGHEST level
    first. A question containing both "describe" and "evaluate" is an
    evaluation question - the highest cognitive demand present governs,
    because the student must satisfy it to answer fully.

    Returns (level, matched_verb) or (None, None).
    """
    if not question:
        return None, None

    text = question.lower()
    # For case-study output, judge the QUESTION line, not the scenario.
    if "question:" in text:
        text = text.split("question:", 1)[1]
    text = text.strip()

    # Action verbs anywhere in the text, highest level first.
    for level in sorted(BLOOM_LEVELS, reverse=True):
        for verb in BLOOM_LEVELS[level]["verbs"]:
            if " " in verb:                      # phrase, e.g. "show how"
                if verb in text:
                    return level, verb
            elif re.search(rf"\b{re.escape(verb)}\b", text):
                return level, verb

    # No action verb. Fall back to interrogative openers, which only
    # carry a level in clause-initial position - matched anywhere they
    # misfire on subordinate clauses.
    #
    # Check the start of each clause, not just the whole string, because
    # a leading subordinate clause is common: "When a VM executes an
    # interrupt, why must the VMM take control?" is a level-2 question
    # whose interrogative sits after the comma.
    clauses = [text] + [c.strip() for c in text.split(",")[1:]]
    for level in sorted(BLOOM_LEVELS, reverse=True):
        for opener in BLOOM_LEVELS[level].get("openers", []):
            if any(c.startswith(opener) for c in clauses):
                return level, opener
    return None, None


def coarse_band(level: int | None) -> str | None:
    """Collapse to the split that can be reported with confidence.

    Fine-grained Bloom classification is unreliable - the
    analyse/evaluate/create boundary is where both human and automated
    classification break down. The lower/higher-order split is robust
    enough to report.
    """
    if level is None:
        return None
    return "lower-order" if level <= 2 else "higher-order"


def bloom_name(level: int | None) -> str | None:
    return BLOOM_LEVELS[level]["name"] if level in BLOOM_LEVELS else None
