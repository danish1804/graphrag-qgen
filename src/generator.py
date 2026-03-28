# ============================================================
# src/generator.py
# Loads Qwen 2.5 7B in 4-bit and generates exam questions
# from retrieved context
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


