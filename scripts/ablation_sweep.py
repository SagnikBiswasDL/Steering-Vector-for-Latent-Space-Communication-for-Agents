"""LatentMAS agent-ablation sweep: do the upstream latent agents earn their keep?

Loads the model ONCE and evaluates several agent-chain configurations by swapping
``method.agents`` between runs (Judger always last). For each config it reports
accuracy, Judger output tokens, wall-clock latency, and analytic latent-compute
(upstream forward passes), plus a simple failure taxonomy (correct / wrong /
no_answer). Output CSV feeds scripts/plot_ablation.py.

Design notes:
  - The clean *prefix chain* (judger_only -> planner_judger ->
    planner_critic_judger -> full) measures the marginal value of each added
    upstream agent, with every agent's prompt receiving the upstream context it
    expects. Prefer this for the causal claim.
  - The *leave-one-out* configs (no_planner, critic_judger, refiner_judger) feed
    Critic/Refiner prompts that reference upstream artifacts that no longer exist
    (see prompts.py), so they measure "agent removed without prompt repair" and
    should be interpreted with that caveat.
  - GSM8K is near-ceiling for Qwen3-14B (~94-95%), so it is best for the
    compute/token axis and harness validation; run the accuracy question on a
    task with headroom (MedQA, GPQA).

Example:
  python scripts/ablation_sweep.py --model_name Qwen/Qwen3-14B --task gsm8k \
      --split test --n 300 \
      --configs judger_only planner_judger planner_critic_judger full \
      --latent_steps 40 --max_new_tokens 2048 --generate_bs 50 \
      --out_csv artifacts/sweeps/ablation_gsm8k.csv
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (  # noqa: E402
    load_gsm8k, load_arc_challenge, load_arc_easy, load_medqa, load_gpqa_diamond,
)
from methods import agents_from_spec  # noqa: E402
from methods.latent_mas import LatentMASMethod  # noqa: E402
from models import ModelWrapper  # noqa: E402
from utils import auto_device, set_seed  # noqa: E402

# Friendly config name -> upstream role spec (Judger always appended by agents_from_spec).
CONFIG_SPECS = {
    "judger_only": "judger",
    "planner_judger": "planner",
    "planner_critic_judger": "planner,critic",   # == no_refiner
    "full": "planner,critic,refiner,judger",
    # leave-one-out / single-upstream (prompt-dependency caveat applies)
    "no_planner": "critic,refiner",
    "no_critic": "planner,refiner",
    "no_refiner": "planner,critic",
    "critic_judger": "critic",
    "refiner_judger": "refiner",
}


def load_dataset(task, split, n, offset=0):
    loaders = {
        "gsm8k": lambda: load_gsm8k(split=split),
        "arc_challenge": lambda: load_arc_challenge(split=split),
        "arc_easy": lambda: load_arc_easy(split=split),
        "medqa": lambda: load_medqa(split=split),
        "gpqa": lambda: load_gpqa_diamond(split=split),
    }
    if task not in loaders:
        raise ValueError(f"unsupported task: {task}")
    out = []
    for i, ex in enumerate(loaders[task]()):
        if i < offset:
            continue
        out.append(ex)
        if n > 0 and len(out) >= n:
            break
    return out


def evaluate_config(method, model, dataset, generate_bs, spec, latent_steps, seed):
    set_seed(seed)
    method.agents = agents_from_spec(spec)
    roles = [a.role for a in method.agents]
    n_upstream = sum(1 for r in roles if r != "judger")

    n_correct, n_no_answer, total_tokens, n = 0, 0, 0, 0
    t0 = time.time()
    for i in range(0, len(dataset), generate_bs):
        batch = dataset[i:i + generate_bs]
        results = method.run_batch(batch)
        for res in results:
            n += 1
            if res.get("correct"):
                n_correct += 1
            elif not str(res.get("prediction", "")).strip():
                n_no_answer += 1
            total_tokens += int(res.get("output_tokens", 0))
    dt = time.time() - t0
    n_wrong = n - n_correct - n_no_answer
    return {
        "roles": "|".join(roles),
        "n_upstream": n_upstream,
        "latent_forwards": n_upstream * (int(latent_steps) + 1),
        "n": n,
        "accuracy": n_correct / n if n else 0.0,
        "correct": n_correct,
        "wrong": n_wrong,
        "no_answer": n_no_answer,
        "mean_output_tokens": total_tokens / n if n else 0.0,
        "total_sec": round(dt, 1),
        "sec_per_sample": round(dt / n, 3) if n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--task", type=str, default="gsm8k")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--data_offset", type=int, default=0)
    ap.add_argument("--configs", type=str, nargs="+",
                    default=["judger_only", "planner_judger", "planner_critic_judger", "full"],
                    help=f"subset of {sorted(CONFIG_SPECS)}")
    ap.add_argument("--latent_steps", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--generate_bs", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", type=str, default="sequential")
    ap.add_argument("--latent_space_realign", action="store_true",
                    help="enable latent realignment W_a (faithfulness check for the handoff)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--device2", type=str, default="cuda:1")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    for c in args.configs:
        if c not in CONFIG_SPECS:
            raise ValueError(f"unknown config '{c}'; valid: {sorted(CONFIG_SPECS)}")

    # Namespace fields ModelWrapper / LatentMASMethod expect.
    args.method = "latent_mas"
    args.use_vllm = False
    args.think = False
    args.seal = False
    args.kvsteer = False
    args.enable_prefix_caching = False
    args.use_second_HF_model = False
    args.capture_acts = None
    args.capture_layer = -1
    args.agents = None
    args.latent_only = False
    args.sequential_info_only = False

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    method = LatentMASMethod(
        model, latent_steps=args.latent_steps, judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, generate_bs=args.generate_bs, args=args,
    )

    dataset = load_dataset(args.task, args.split, args.n, offset=args.data_offset)
    print(f"[ablation] {len(dataset)} eval items on {args.task}/{args.split} "
          f"(offset={args.data_offset}, realign={args.latent_space_realign})")

    # De-duplicate configs that resolve to the same chain (e.g. planner_critic_judger == no_refiner).
    seen_chains = {}
    ordered_configs = []
    for c in args.configs:
        chain = tuple(a.role for a in agents_from_spec(CONFIG_SPECS[c]))
        if chain in seen_chains:
            print(f"[ablation] '{c}' resolves to same chain as '{seen_chains[chain]}'; skipping duplicate")
            continue
        seen_chains[chain] = c
        ordered_configs.append(c)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fieldnames = ["config", "roles", "n_upstream", "latent_forwards", "n", "accuracy",
                  "correct", "wrong", "no_answer", "mean_output_tokens", "total_sec", "sec_per_sample"]
    rows = []
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in ordered_configs:
            res = evaluate_config(method, model, dataset, args.generate_bs,
                                  CONFIG_SPECS[c], args.latent_steps, args.seed)
            row = {"config": c, **res}
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"[ablation] {c:>22} [{res['roles']}] -> acc={res['accuracy']*100:.1f}% "
                  f"tokens={res['mean_output_tokens']:.1f} lat_fwd={res['latent_forwards']} "
                  f"({res['sec_per_sample']}s/ex)", flush=True)

    with open(args.out_csv.replace(".csv", ".json"), "w") as f:
        json.dump({"meta": {"task": args.task, "split": args.split, "n": args.n,
                            "model": args.model_name, "seed": args.seed,
                            "latent_steps": args.latent_steps,
                            "latent_space_realign": args.latent_space_realign},
                   "rows": rows}, f, indent=2)
    print(f"[ablation] done -> {args.out_csv}")


if __name__ == "__main__":
    main()
