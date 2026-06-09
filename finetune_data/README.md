# finetune_data/ — LoRA adapters and training pipeline

Two adapters, two jobs:

| Adapter | Folder | Trained on | Used for |
|---------|--------|------------|----------|
| **Answer LoRA** (friend) | `finetune_data/` (root) | Question → answer (`qa_pairs.jsonl`) | Sample answer generation |
| **Question LoRA** (yours) | `finetune_data/question_lora/` | Context + LO → question (GraphRAG prompts) | Question generation |

Controlled in root `config.yml` → `finetune:` section.

---

## Files

| File | Purpose |
|------|---------|
| `adapter_config.json` + `adapter_model.safetensors` | Friend's answer LoRA (already trained) |
| `fine-tuning-qwen-2-5-edgenai 1.ipynb` | Friend's Kaggle notebook (QA fine-tune) |
| `build_graphrag_question_dataset.py` | Build GraphRAG question JSONL from judge eval JSON |
| `train_question_lora.py` | Train question LoRA on your RTX 4060 |
| `graphrag_question_train.jsonl` | Generated training set |
| `graphrag_question_eval.jsonl` | Held-out eval set |
| `question_lora/` | Output after question fine-tune |

---

## Step-by-step: GraphRAG question fine-tune

### 1. Build dataset (from your existing judge runs)

```powershell
python finetune_data/build_graphrag_question_dataset.py
```

Options:

```powershell
# Stricter quality filter
python finetune_data/build_graphrag_question_dataset.py --min-score 4.5

# Re-fetch full GraphRAG context (slower, closer to production prompts)
python finetune_data/build_graphrag_question_dataset.py --refresh-context
```

Requires `outputs/evaluations/eval_*.json` from generations with the judge enabled.

### 2. Train question LoRA (~2–4 hours on RTX 4060)

```powershell
python finetune_data/train_question_lora.py
```

Defaults: 2 epochs, max seq 2048, LoRA r=16 (same as friend's run).

### 3. Enable in config.yml

```yaml
finetune:
  use_answer_lora: true
  use_question_lora: true   # flip after step 2
```

### 4. Restart API and compare

```powershell
python main_gr.py --mode full
```

Re-run the same learning objectives; compare judge `question_overall` before/after.

---

## Answer LoRA only (no retrain)

Friend's adapter is already in this folder. Set:

```yaml
finetune:
  use_answer_lora: true
  use_question_lora: false
```

Restart the server — answers use LoRA, questions use base model + GraphRAG prompts.

---

## Optional: friend's training data

If you obtain `qa_pairs.jsonl` (Kaggle: `shrinidhikukde/qa-pair`), place it here for reference or to re-run her notebook. It is **not** used for GraphRAG question fine-tune (wrong prompt shape).

---

## Regenerate diagram assets

```powershell
cd docs/images
python render_diagrams.py
```
