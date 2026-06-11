"""
Fine-tune Qwen 2.5 7B on GraphRAG question prompts (context + LO -> exam question).

Requires GPU (~8 GB+ VRAM with 4-bit base + LoRA). Run after build_graphrag_question_dataset.py.

Usage (from repo root):
  python finetune_data/build_graphrag_question_dataset.py
  python finetune_data/train_question_lora.py
  python finetune_data/train_question_lora.py --epochs 2 --max-seq-len 2048
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config import cfg  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = DATA_DIR / "graphrag_question_train.jsonl"
DEFAULT_EVAL = DATA_DIR / "graphrag_question_eval.jsonl"
DEFAULT_OUT = DATA_DIR / "question_lora"


def _messages_to_text(tokenizer, messages: list[dict]) -> str:
    """Chat template for causal LM training (assistant turn included)."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ) + tokenizer.eos_token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    if not args.train_file.is_file():
        print(f"Missing {args.train_file}")
        print("Run: python finetune_data/build_graphrag_question_dataset.py")
        return 1

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if not torch.cuda.is_available():
        print("CUDA GPU required for QLoRA training.")
        return 1

    model_id = cfg["model"]["llm_name"]
    print(f"Base model: {model_id}")
    print(f"Train file: {args.train_file}")

    raw_train = load_dataset("json", data_files=str(args.train_file), split="train")
    eval_ds = None
    if args.eval_file.is_file():
        eval_ds = load_dataset("json", data_files=str(args.eval_file), split="train")
        print(f"Eval rows: {len(eval_ds)}")
    print(f"Train rows: {len(raw_train)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_row(row):
        text = _messages_to_text(tokenizer, row["messages"])
        return {"text": text}

    train_formatted = raw_train.map(format_row, remove_columns=raw_train.column_names)
    eval_formatted = None
    if eval_ds is not None:
        eval_formatted = eval_ds.map(format_row, remove_columns=eval_ds.column_names)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )

    train_tokenized = train_formatted.map(tokenize_fn, batched=True)
    eval_tokenized = (
        eval_formatted.map(tokenize_fn, batched=True) if eval_formatted is not None else None
    )

    lengths = [len(x) for x in train_tokenized["input_ids"]]
    print(
        "Token lengths — min/mean/max:",
        min(lengths),
        int(np.mean(lengths)),
        max(lengths),
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def collator_with_labels(features):
        # Strip non-tokenizer fields (e.g. "text") before padding
        clean = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch = tokenizer.pad(clean, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps" if eval_tokenized is not None else "no",
        eval_steps=50 if eval_tokenized is not None else None,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=eval_tokenized is not None,
        metric_for_best_model="eval_loss" if eval_tokenized is not None else None,
        greater_is_better=False,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=collator_with_labels,
        processing_class=tokenizer,
    )

    torch.cuda.empty_cache()
    trainer.train()

    final_dir = args.output_dir
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved question LoRA adapter to: {final_dir}")
    print("Enable in config.yml: finetune.use_question_lora: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
