# Investigating LoRA Rank in Small-Model DPO Fine-Tuning

## Project Overview

This project investigates how LoRA adapter rank affects preference alignment performance when fine-tuning `Qwen/Qwen2.5-0.5B-Instruct` using Direct Preference Optimization (DPO).

The original baseline uses LoRA rank 16. I compared this configuration with rank 8 to examine whether increasing adapter capacity improves preference win rate under the same training setup.

## Experimental Setup

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Dataset:** `Anthropic/hh-rlhf`
- **Training data:** 2,000 single-turn preference pairs
- **Evaluation:** 200 held-out prompts
- **Reward model:** `OpenAssistant/reward-model-deberta-v3-large-v2`
- **LoRA ranks:** 8 and 16
- **DPO beta:** 0.1
- **Learning rate:** 5e-5
- **Training epochs:** 1
- **Random seeds:** 0, 1, 2
- **Environment:** Google Colab with NVIDIA T4 GPU

Each configuration was evaluated by comparing the DPO-tuned model's responses against those of the original base model. A win was counted when the reward model assigned a higher score to the tuned response.

## Results

| Seed | Rank 16 Win Rate | Rank 8 Win Rate |
|---|---|---|
| 0 | 47.0% | 41.0% |
| 1 | 45.0% | 35.0% |
| 2 | 54.5% | 43.5% |
| **Mean** | **48.83%** | **39.83%** |
| **Sample Standard Deviation** | **5.01 percentage points** | **4.37 percentage points** |

Rank 16 achieved a mean win rate 9 percentage points higher than rank 8 across the three seeds. Although rank 16 performed better in all three runs, the limited number of seeds and evaluation prompts prevents a definitive conclusion about statistical significance.

Neither configuration consistently exceeded a 50% win rate against the base model under this evaluation procedure.

## Implementation Changes and Troubleshooting

### 1. Evaluation Data Overlap

The original training script selected evaluation prompts from examples already used for training.

I changed the evaluation selection to use examples immediately following the 2,000 training examples, ensuring the training and evaluation index ranges do not overlap.

These evaluation examples are held out from training but are still drawn from the dataset's original training split.

Because the evaluation methodology changed, the original reported baseline win rate of 71% is not directly comparable to the results above.

### 2. GPU Memory Management

An initial full training attempt encountered a CUDA out-of-memory error on the Colab T4 GPU.

To reduce memory usage, I changed the per-device training batch size from 4 to 1 and increased gradient accumulation steps from 4 to 16. This preserved the effective batch size of 16 while allowing training to complete.

### 3. Evaluation Runtime

Training took approximately 7–8 minutes per run, while end-to-end execution took substantially longer.

I investigated batched response generation as a possible optimization. After observing differences between batched and individual generation, I retained the original single-prompt evaluation approach to maintain consistency across experiments.

## Reproducing the Experiments

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the rank-16 baseline:

```bash
python dpo_train.py --n_train 2000 --n_eval 200 --lora_r 16 --beta 0.1 --lr 5e-5 --epochs 1 --seed 0 --out rank16_seed0.json
```

Run the rank-8 comparison:

```bash
python dpo_train.py --n_train 2000 --n_eval 200 --lora_r 8 --beta 0.1 --lr 5e-5 --epochs 1 --seed 0 --out rank8_seed0.json
```

Repeat with `--seed 1` and `--seed 2` for both ranks.

For reproduction, use the corrected `dpo_train.py` on the `tanisha-exp` branch.

## Results and Reproducibility Notes

The `results/` directory contains five saved experiment JSON files. The rank-8, seed-0 run completed with a recorded win rate of 41.0%, but its JSON output was lost after a Colab runtime disconnection. This result is included from the recorded experiment logs rather than a recovered JSON artifact.

The random seed also affects dataset shuffling and selection, so evaluation examples may differ between seeds. Results should therefore be interpreted as variability across complete seeded runs, rather than evaluations on one fixed prompt set.

## Limitations and Future Work

The evaluation relies on an automated reward model, which may not fully capture human preferences. The experiment also uses a relatively small training dataset, 200 evaluation prompts per run, and only three random seeds.

Future work could include evaluating on a separate test split, using a fixed evaluation set across seeds, measuring peak GPU memory and adapter parameter counts, and investigating additional LoRA ranks and DPO beta values.

## Project Files

- `dpo_train.py` — DPO fine-tuning and evaluation script
- `requirements.txt` — Python dependencies
- `results/` — saved experiment outputs
- `ASSIGNMENT.md` — original research assignment
- `AI_USE.md` — AI assistance documentation
- `report.pdf` — final research report
