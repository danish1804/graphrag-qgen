# ============================================================
# src/generator.py
# Loads Qwen 2.5 7B in 4-bit and generates exam questions
# from retrieved context
# ============================================================

from dataclasses_json import cfg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import sys
from config import cfg
sys.path.append(os.path.dirname(os.path.dirname(__file__)))



MODEL_NAME     = cfg["model"]["llm_name"]
MAX_NEW_TOKENS = cfg["model"]["max_new_tokens"]
TEMPERATURE    = cfg["model"]["temperature"]
BASE_DIR = os.getenv(
    "GRAPHRAG_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

def load_model():
    """Load Qwen 2.5 in 4-bit quantisation."""
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

def generate_answer(
        context:            str,
        question:           str,
        model,
        tokenizer,
        difficulty:         str = "medium",
        max_new_tokens:     int = 256
) -> str:
    """
    Generate a sample answer for a given question and context.
    Uses the same retrieved context as question generation.

    Args:
        context:        Retrieved chunks from GraphRAG
        question:       The generated exam question
        model:          Loaded Qwen 2.5 model
        tokenizer:      Qwen tokenizer
        difficulty:     easy/medium/hard — controls answer depth
        max_new_tokens: Maximum tokens in answer

    Returns:
        Sample answer string
    """
    # Depth of answer varies by difficulty
    depth_instruction = {
        "easy":   "Write a brief, direct answer in 2-3 sentences.",
        "medium": "Write a clear answer in 3-5 sentences covering the key points.",
        "hard":   "Write a detailed answer in 5-8 sentences covering concepts, relationships and examples."
    }.get(difficulty, "Write a clear answer in 3-5 sentences.")

    prompt = f"""You are an expert university educator.
Using the context below, write a sample answer for the exam question.

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

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert university educator. "
                "Generate clear, accurate sample answers "
                "for exam questions based on provided context."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [text], return_tensors="pt"
    ).to(model.device)

    with __import__('torch').no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            model_inputs.input_ids, generated_ids)
    ]

    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    return answer

# ============================================================
# TEST
# python src/generator.py
# ============================================================
if __name__ == "__main__":

    from utils.document_loader import load_and_chunk
    from utils.embedder import (build_index, load_index,
                                 search, Embedder)

    print("=" * 50)
    print("GENERATOR TEST")
    print("=" * 50)

    INDEX_DIR = os.path.join(BASE_DIR, "indexes")
    PDF_PATH  = os.path.join(BASE_DIR, "data", "tested", "source_1.pdf")

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

   
