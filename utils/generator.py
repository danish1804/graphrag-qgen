# ============================================================
# utils/generator.py
# Loads Qwen 2.5 7B in 4-bit and generates exam questions
# from retrieved context
# ============================================================

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from utils.config import cfg
except ImportError:
    from config import cfg

from utils.prompts import (
    PROMPT_VERSION,
    build_answer_messages,
    build_question_messages,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

MODEL_NAME = cfg["model"]["llm_name"]
MODEL_CFG = cfg["model"]
MAX_NEW_TOKENS_QUESTION = MODEL_CFG.get(
    "max_new_tokens_question",
    MODEL_CFG.get("max_new_tokens", 150),
)
MAX_NEW_TOKENS_ANSWER = MODEL_CFG.get(
    "max_new_tokens_answer",
    MODEL_CFG.get("max_new_tokens", 512),
)
TEMPERATURE = MODEL_CFG.get("temperature", 0.7)
DO_SAMPLE = MODEL_CFG.get("do_sample", True)

BASE_DIR = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


def load_model():
    """Load Qwen 2.5 in 4-bit quantisation."""
    print(f"Loading model: {MODEL_NAME}")
    print("This will take 2-3 minutes and download ~4GB on first run...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Model loaded successfully.")
    return model, tokenizer


def _generate_from_messages(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float | None = None,
    do_sample: bool | None = None,
) -> str:
    """Shared chat-template generation path for question and answer."""
    temperature = TEMPERATURE if temperature is None else temperature
    do_sample = DO_SAMPLE if do_sample is None else do_sample

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _max_tokens_for_question_type(question_type: str) -> int:
    qt_cfg = cfg.get("question_types", {}).get(question_type, {})
    return int(qt_cfg.get("max_new_tokens", MAX_NEW_TOKENS_QUESTION))


def generate_question(
    context: str,
    learning_objective: str,
    model,
    tokenizer,
    difficulty: str = "medium",
    question_type: str = "conceptual",
) -> str:
    """Generate an exam question from retrieved context."""
    messages = build_question_messages(
        context=context,
        learning_objective=learning_objective,
        difficulty=difficulty,
        question_type=question_type,
    )
    return _generate_from_messages(
        model,
        tokenizer,
        messages,
        max_new_tokens=_max_tokens_for_question_type(question_type),
    )


def generate_answer(
    context: str,
    question: str,
    model,
    tokenizer,
    difficulty: str = "medium",
    max_new_tokens: int | None = None,
) -> str:
    """Generate a sample answer for a given question and context."""
    messages = build_answer_messages(
        context=context,
        question=question,
        difficulty=difficulty,
    )
    return _generate_from_messages(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens or MAX_NEW_TOKENS_ANSWER,
    )


def get_generation_trace() -> dict:
    """Metadata fields to attach to generation responses/eval records."""
    return {
        "prompt_version": PROMPT_VERSION,
        "generation_model": MODEL_NAME,
        "generation_temperature": TEMPERATURE,
        "max_new_tokens_question": MAX_NEW_TOKENS_QUESTION,
        "max_new_tokens_answer": MAX_NEW_TOKENS_ANSWER,
    }
