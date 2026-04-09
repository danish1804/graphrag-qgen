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
- Return ONLY the question - no preamble, no explanation, no answer
- The question should be clear and unambiguous
- Do NOT generate multiple choice questions
- Do NOT generate A) B) C) D) options"""

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


# ============================================================
# TEST
# python src/generator.py
# ============================================================
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from src.document_loader import load_and_chunk
    from src.embedder import (build_index, load_index,
                               search, Embedder)

    print("=" * 50)
    print("GENERATOR TEST")
    print("=" * 50)

    INDEX_DIR = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen\indexes"
    PDF_PATH  = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen\data\source_1.pdf"

    # Load existing index or build it
    index_faiss_path = os.path.join(INDEX_DIR, "cnn_lecture.faiss")
    if os.path.exists(index_faiss_path):
        print("Loading existing FAISS index...")
        index, chunks = load_index(INDEX_DIR, "cnn_lecture")
    else:
        print("Building FAISS index...")
        result = load_and_chunk(PDF_PATH)
        build_index(result["chunks"], INDEX_DIR, "cnn_lecture")
        index, chunks = load_index(INDEX_DIR, "cnn_lecture")

    # Load embedding model for search
    embedder = Embedder()

    # Load Qwen
    model, tokenizer = load_model()

    # Test questions with different objectives
    test_objectives = [
        ("What is parameter sharing in CNNs?",    "medium", "conceptual"),
        ("How does max pooling reduce dimensions?", "medium", "application"),
        ("What is sparse connectivity?",           "easy",   "recall"),
    ]
    
    for objective, difficulty, qtype in test_objectives:
        print(f"\n{'='*50}")
        print(f"Objective : {objective}")
        print(f"Difficulty: {difficulty} | Type: {qtype}")
        print("-" * 50)

        # Retrieve top 3 chunks
        results = search(objective, index, chunks, embedder, k=3)
        context = "\n\n".join([r["chunk"] for r in results])

        # Generate question
        question = generate_question(
            context, objective, model, tokenizer,
            difficulty, qtype
        )

        print(f"Generated question:\n{question}")

    print("\n" + "=" * 50)
    print("Generator test complete.")

   
