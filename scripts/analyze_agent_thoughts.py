"""Per-agent thought-type distribution analysis (SEAL Figure-1 style).

For each LatentMAS role (planner / critic / refiner / judger) we generate the
role's reasoning as text (from its role prompt), classify every step into
execution / reflection / transition, and report the per-role distribution. We
then plot the fractions so we can explain *why* SEAL (which suppresses
reflection/transition) helps some roles and not others.

Note: the latent sub-agents emit no text in the real pipeline; here we decode
each role's reasoning from its own role prompt (isolation) as a faithful proxy
for that role's intrinsic reasoning style. A context-aware (sequential) variant
is a straightforward follow-up.

Example:
  python scripts/analyze_agent_thoughts.py --model_name Qwen/Qwen3-14B \
      --task gsm8k --n 60 --max_new_tokens 768 \
      --out_json artifacts/analysis/agent_thoughts_gsm8k.json \
      --out_plot artifacts/plots/agent_thoughts_gsm8k.png
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_gsm8k, load_arc_challenge, load_medqa  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.thought_classifier import classify_step, split_into_steps, THOUGHT_TYPES  # noqa: E402

ROLES = ["planner", "critic", "refiner", "judger"]


def load_questions(task, split, limit):
    if task == "gsm8k":
        it = load_gsm8k(split=split)
    elif task == "arc_challenge":
        it = load_arc_challenge(split=split)
    elif task == "medqa":
        it = load_medqa(split=split)
    else:
        raise ValueError(f"unsupported task: {task}")
    out = []
    for ex in it:
        out.append(ex["question"])
        if len(out) >= limit:
            break
    return out


@torch.no_grad()
def gen_text(model, tokenizer, messages, device, max_new_tokens, temperature, top_p):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    plen = enc["input_ids"].shape[1]
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p, pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


def make_plot(per_role, out_plot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    roles = [r for r in ROLES if r in per_role]
    exec_frac = [per_role[r]["fractions"]["execution"] for r in roles]
    refl_frac = [per_role[r]["fractions"]["reflection"] for r in roles]
    trans_frac = [per_role[r]["fractions"]["transition"] for r in roles]

    x = np.arange(len(roles))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, exec_frac, label="execution")
    ax.bar(x, refl_frac, bottom=exec_frac, label="reflection")
    ax.bar(x, trans_frac, bottom=[e + r for e, r in zip(exec_frac, refl_frac)], label="transition")
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylabel("fraction of reasoning steps")
    ax.set_ylim(0, 1)
    ax.set_title("Thought-type distribution per LatentMAS agent")
    ax.legend(loc="upper right")
    for i, r in enumerate(roles):
        non_exec = refl_frac[i] + trans_frac[i]
        ax.text(i, 1.01, f"non-exec {non_exec:.0%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)), exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    print(f"[analyze] plot -> {out_plot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--task", type=str, default="gsm8k")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=60, help="questions per role")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--roles", type=str, default=",".join(ROLES))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_plot", type=str, required=True)
    args = ap.parse_args()

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    ).to(device).eval()

    questions = load_questions(args.task, args.split, args.n)
    prompt_args = SimpleNamespace(model_name=args.model_name)
    print(f"[analyze] {len(questions)} questions x {len(roles)} roles on {args.task}")

    per_role = {}
    for role in roles:
        counts = {t: 0 for t in THOUGHT_TYPES}
        for qi, q in enumerate(questions):
            messages = build_agent_message_sequential_latent_mas(
                role=role, question=q, context="", method="latent_mas", args=prompt_args
            )
            text = gen_text(model, tokenizer, messages, device,
                            args.max_new_tokens, args.temperature, args.top_p)
            for step in split_into_steps(text):
                counts[classify_step(step)] += 1
            if (qi + 1) % 20 == 0:
                print(f"[analyze] role={role} q={qi+1} counts={counts}", flush=True)
        total = max(1, sum(counts.values()))
        per_role[role] = {
            "counts": counts,
            "total_steps": sum(counts.values()),
            "fractions": {t: counts[t] / total for t in THOUGHT_TYPES},
            "non_execution_fraction": (counts["reflection"] + counts["transition"]) / total,
        }
        print(f"[analyze] {role}: {per_role[role]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"task": args.task, "model": args.model_name, "n": args.n, "per_role": per_role}, f, indent=2)
    print(f"[analyze] json -> {args.out_json}")
    make_plot(per_role, args.out_plot)


if __name__ == "__main__":
    main()
