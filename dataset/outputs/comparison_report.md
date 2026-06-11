# Base vs Fine-tuned (Question LoRA) — Paired Comparison

- Base dataset: `P000233DS_Question_Dataset_BASE_MODEL.xlsx` (240 done rows)
- Fine-tuned dataset: `P000233DS_Question_Dataset_BASE_MODEL.xlsx` (240 done rows)
- Matched pairs (same Q_ID): **240**
- Test: two-sided Wilcoxon signed-rank on paired differences (zero-difference pairs excluded). `*` marks p < 0.05.
- Effect size: matched-pairs rank-biserial correlation (positive favours the fine-tuned model).

### Overall (all matched pairs)

| Group | n | Base mean±sd | FT mean±sd | Δ mean | W/T/L | p (Wilcoxon) | Effect (r) |
|---|---|---|---|---|---|---|---|
| Q Score (/5) | 238 | 4.68±0.55 | 4.68±0.55 | +0.00 | 0/238/0 | n/a | +nan |
| A Score (/5) | 234 | 3.88±1.02 | 3.88±1.02 | +0.00 | 0/234/0 | n/a | +nan |
| R Score (/5) | 240 | 1.85±0.72 | 1.85±0.72 | +0.00 | 0/240/0 | n/a | +nan |

### Q Score by difficulty

| Group | n | Base mean±sd | FT mean±sd | Δ mean | W/T/L | p (Wilcoxon) | Effect (r) |
|---|---|---|---|---|---|---|---|
| easy | 73 | 4.75±0.56 | 4.75±0.56 | +0.00 | 0/73/0 | n/a | +nan |
| hard | 15 | 4.07±0.58 | 4.07±0.58 | +0.00 | 0/15/0 | n/a | +nan |
| medium | 150 | 4.70±0.50 | 4.70±0.50 | +0.00 | 0/150/0 | n/a | +nan |

### Q Score by domain

| Group | n | Base mean±sd | FT mean±sd | Δ mean | W/T/L | p (Wilcoxon) | Effect (r) |
|---|---|---|---|---|---|---|---|
| auto | 15 | 4.07±0.58 | 4.07±0.58 | +0.00 | 0/15/0 | n/a | +nan |
| computer_architecture | 15 | 4.87±0.25 | 4.87±0.25 | +0.00 | 0/15/0 | n/a | +nan |
| database | 15 | 4.52±0.96 | 4.52±0.96 | +0.00 | 0/15/0 | n/a | +nan |
| database+computer_architecture | 15 | 4.71±0.44 | 4.71±0.44 | +0.00 | 0/15/0 | n/a | +nan |
| database+operating_systems | 15 | 4.67±0.56 | 4.67±0.56 | +0.00 | 0/15/0 | n/a | +nan |
| database+software_engineering | 15 | 4.79±0.37 | 4.79±0.37 | +0.00 | 0/15/0 | n/a | +nan |
| deep_learning | 13 | 4.78±0.53 | 4.78±0.53 | +0.00 | 0/13/0 | n/a | +nan |
| deep_learning+computer_architecture | 15 | 4.68±0.61 | 4.68±0.61 | +0.00 | 0/15/0 | n/a | +nan |
| deep_learning+database | 15 | 4.76±0.39 | 4.76±0.39 | +0.00 | 0/15/0 | n/a | +nan |
| deep_learning+operating_systems | 15 | 4.68±0.46 | 4.68±0.46 | +0.00 | 0/15/0 | n/a | +nan |
| deep_learning+software_engineering | 15 | 4.49±0.76 | 4.49±0.76 | +0.00 | 0/15/0 | n/a | +nan |
| operating_systems | 15 | 4.80±0.45 | 4.80±0.45 | +0.00 | 0/15/0 | n/a | +nan |
| operating_systems+computer_architecture | 15 | 4.84±0.42 | 4.84±0.42 | +0.00 | 0/15/0 | n/a | +nan |
| software_engineering | 15 | 4.79±0.37 | 4.79±0.37 | +0.00 | 0/15/0 | n/a | +nan |
| software_engineering+computer_architecture | 15 | 4.55±0.52 | 4.55±0.52 | +0.00 | 0/15/0 | n/a | +nan |
| software_engineering+operating_systems | 15 | 4.87±0.29 | 4.87±0.29 | +0.00 | 0/15/0 | n/a | +nan |

### Q Score by question type

| Group | n | Base mean±sd | FT mean±sd | Δ mean | W/T/L | p (Wilcoxon) | Effect (r) |
|---|---|---|---|---|---|---|---|
| application | 15 | 4.07±0.58 | 4.07±0.58 | +0.00 | 0/15/0 | n/a | +nan |
| conceptual | 150 | 4.70±0.50 | 4.70±0.50 | +0.00 | 0/150/0 | n/a | +nan |
| recall | 73 | 4.75±0.56 | 4.75±0.56 | +0.00 | 0/73/0 | n/a | +nan |

### Judge-triggered regeneration attempts

| Group | n | Base mean±sd | FT mean±sd | Δ mean | W/T/L | p (Wilcoxon) | Effect (r) |
|---|---|---|---|---|---|---|---|
| Attempts (lower=better) | 240 | 1.13±0.45 | 1.13±0.45 | +0.00 | 0/240/0 | n/a | +nan |

_Note: for Attempts a negative Δ is an improvement._
