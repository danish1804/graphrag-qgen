# ============================================================
# src/generator.py
# Loads Qwen 2.5 7B in 4-bit and generates exam questions
# from retrieved context
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def load_model():
    """Load Qwen 2.5 7B in 4-bit quantisation."""
    print(f"Loading model: {MODEL_NAME}")
    print("This will take 2-3 minutes and download ~4GB on first run...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print("Model loaded successfully.")
    return model, tokenizer


def generate_question(context: str, learning_objective: str,
                      model, tokenizer,
                      difficulty: str = "medium",
                      question_type: str = "conceptual") -> str:
    """
    Generate an exam question from retrieved context.

    Args:
        context: retrieved text from FAISS
        learning_objective: educator's query
        model: loaded Qwen model
        tokenizer: Qwen tokenizer
        difficulty: easy / medium / hard
        question_type: recall / conceptual / application / analytical

    Returns:
        Generated question string
    """

    system_prompt = """You are an expert exam question generator for IT education.
Your job is to generate exactly ONE high-quality exam question.

Rules:
- The question MUST be answerable using ONLY the provided context
- Do NOT introduce concepts not present in the context
- Return ONLY the question — no preamble, no explanation, no answer
- The question should be clear and unambiguous"""

    user_prompt = f"""Context from course material:
{context}

Learning objective: {learning_objective}
Difficulty: {difficulty}
Question type: {question_type}

Generate one exam question based strictly on the context above."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    # Apply Qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    question = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return question
