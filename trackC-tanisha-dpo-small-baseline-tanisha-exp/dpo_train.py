"""LoRA + DPO fine-tuning of a small language model on Anthropic HH-RLHF,
then win rate of the tuned model vs. the base model judged by a reward model.

Designed for a free Colab T4 (fp16). Expect ~30-45 minutes end to end.
"""
import argparse
import json
import random

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)
from trl import DPOConfig, DPOTrainer

POLICY = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE = "OpenAssistant/reward-model-deberta-v3-large-v2"
SEP = "\n\nAssistant:"


def split_example(ex):
    """HH-RLHF stores full dialogues; split into prompt / chosen / rejected."""
    i = ex["chosen"].rfind(SEP) + len(SEP)
    j = ex["rejected"].rfind(SEP) + len(SEP)
    return {"prompt": ex["chosen"][:i], "chosen": ex["chosen"][i:], "rejected": ex["rejected"][j:]}


def single_turn(ex):
    return ex["prompt"].count("Human:") == 1 and len(ex["prompt"]) < 400


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=128):
    model.eval()
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)
        g = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        text = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(text.split("\n\nHuman:")[0].strip())
    return outs


@torch.no_grad()
def score(judge, jtok, prompts, answers):
    s = []
    for p, a in zip(prompts, answers):
        q = p.replace("\n\nHuman:", "").replace(SEP, "").strip()
        inp = jtok(q, a, return_tensors="pt", truncation=True, max_length=512).to(judge.device)
        s.append(judge(**inp).logits[0].item())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw = load_dataset("Anthropic/hh-rlhf", split="train")
    pool = raw.map(split_example).filter(single_turn).shuffle(seed=args.seed)
    train_ds = pool.select(range(args.n_train))
    eval_prompts = pool.select(
        range(args.n_train, args.n_train + args.n_eval)
    )["prompt"]

    tok = AutoTokenizer.from_pretrained(POLICY)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(POLICY, torch_dtype=torch.float16, device_map="auto")

    cfg = DPOConfig(
        output_dir="dpo_out", per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=args.lr, num_train_epochs=args.epochs, beta=args.beta,
        max_length=512, max_prompt_length=256, logging_steps=10, save_strategy="no",
        fp16=True, report_to="none", seed=args.seed,
    )
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=train_ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    policy = trainer.model

    with policy.disable_adapter():
        base_answers = generate(policy, tok, eval_prompts)
    dpo_answers = generate(policy, tok, eval_prompts)

    jtok = AutoTokenizer.from_pretrained(JUDGE)
    judge = AutoModelForSequenceClassification.from_pretrained(JUDGE).to(policy.device).eval()
    s_base = score(judge, jtok, eval_prompts, base_answers)
    s_dpo = score(judge, jtok, eval_prompts, dpo_answers)
    win = sum(d > b for d, b in zip(s_dpo, s_base)) / len(s_base)
    print(f"win rate (DPO vs base, {len(s_base)} prompts): {win:.3f}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "win_rate": win,
                   "examples": [{"prompt": p, "base": b, "dpo": d}
                                for p, b, d in list(zip(eval_prompts, base_answers, dpo_answers))[:10]]},
                  f, indent=2)


if __name__ == "__main__":
    main()
