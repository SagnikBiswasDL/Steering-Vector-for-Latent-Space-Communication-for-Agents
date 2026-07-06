"""Single-load evaluation sweep for one-shot KV-cache steering of the handoff.

Loads the model ONCE and evaluates many KV-steering configurations by mutating
the ModelWrapper's ``kvsteer`` (positions x c_v [x c_k]) between runs, instead of
re-launching run.py per config. For each configuration it reports downstream
accuracy and Judger ``mean_output_tokens`` on the eval set, so we can plot the
accuracy-token Pareto frontier (via scripts/plot_pareto.py) and answer whether
steering the latent-agent handoff (arXiv:2507.08799) makes the 4-agent system
more correct and/or cheaper.

The always-included ``control`` row is plain LatentMAS (c_v = c_k = 0, a no-op
by construction). ``judger_token`` is the control arm that steers the Judger's
own final prompt token rather than the handoff.

Example:
  python scripts/kv_eval_sweep.py \
      --model_name Qwen/Qwen3-14B --task gsm8k --split test --n 150 \
      --kvsteer_vector artifacts/kv_steer_vectors/qwen3-14b/gsm8k_cot.pt \
      --positions handoff_last handoff_lastk handoff_all judger_token \
      --cvs -8 -4 4 8 --latent_steps 40 --max_new_tokens 2048 \
      --out_csv artifacts/sweeps/kv_gsm8k.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (  # noqa: E402
    load_gsm8k, load_arc_challenge, load_medqa, load_arc_easy,
)
from methods.latent_mas import LatentMASMethod  # noqa: E402
from models import ModelWrapper  # noqa: E402
from utils import auto_device, set_seed  # noqa: E402


def load_dataset(task, split, n, offset=0):
    if task == "gsm8k":
        it = load_gsm8k(split=split)
    elif task == "arc_challenge":
        it = load_arc_challenge(split=split)
    elif task == "arc_easy":
        it = load_arc_easy(split=split)
    elif task == "medqa":
        it = load_medqa(split=split)
    else:
        raise ValueError(f"unsupported task: {task}")
    out = []
    for i, ex in enumerate(it):
        if i < offset:
            continue
        out.append(ex)
        if n > 0 and len(out) >= n:
            break
    return out


def evaluate_config(method, model, dataset, generate_bs, *, positions, c_v, c_k, last_k, seed):
    set_seed(seed)
    st = model.kvsteer
    st.positions = positions
    st.c_v = float(c_v)
    st.c_k = float(c_k)
    st.last_k = int(last_k)

    n_correct, total_tokens, n = 0, 0, 0
    t0 = time.time()
    for i in range(0, len(dataset), generate_bs):
        batch = dataset[i:i + generate_bs]
        results = method.run_batch(batch)
        for res in results:
            n += 1
            n_correct += 1 if res.get("correct") else 0
            total_tokens += int(res.get("output_tokens", 0))
    dt = time.time() - t0
    acc = n_correct / n if n else 0.0
    mean_tokens = total_tokens / n if n else 0.0
    return {"n": n, "accuracy": acc, "correct": n_correct,
            "mean_output_tokens": mean_tokens, "sec": round(dt, 1),
            "sec_per_sample": round(dt / n, 3) if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--task", type=str, default="gsm8k")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--data_offset", type=int, default=0)
    ap.add_argument("--kvsteer_vector", type=str, required=True)
    ap.add_argument("--positions", type=str, nargs="+",
                    default=["handoff_last", "handoff_lastk", "handoff_all", "judger_token"])
    ap.add_argument("--cvs", type=float, nargs="+", default=[-8, -4, 4, 8],
                    help="value coefficients to sweep (both signs); 0 is added as control")
    ap.add_argument("--ck", type=float, default=0.0, help="fixed key coefficient")
    ap.add_argument("--last_k", type=int, default=40)
    ap.add_argument("--latent_steps", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--generate_bs", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", type=str, default="sequential")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--device2", type=str, default="cuda:1")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    # Namespace fields the ModelWrapper / method expect.
    args.method = "latent_mas"
    args.use_vllm = False
    args.latent_space_realign = False
    args.think = False
    args.seal = False
    args.enable_prefix_caching = False
    args.use_second_HF_model = False
    args.capture_acts = None
    args.capture_layer = -1
    # kvsteer must be on so ModelWrapper builds model.kvsteer; we mutate it per config.
    args.kvsteer = True
    args.kvsteer_cv = 0.0
    args.kvsteer_ck = args.ck
    args.kvsteer_positions = "handoff_last"
    args.kvsteer_last_k = args.last_k

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    if getattr(model, "kvsteer", None) is None:
        raise RuntimeError("model.kvsteer not initialized; check --kvsteer_vector")

    method = LatentMASMethod(
        model, latent_steps=args.latent_steps, judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, generate_bs=args.generate_bs, args=args,
    )

    dataset = load_dataset(args.task, args.split, args.n, offset=args.data_offset)
    print(f"[kv-sweep] {len(dataset)} eval items on {args.task}/{args.split} (offset={args.data_offset})")

    # control (c_v = c_k = 0, no-op) then each (positions x c_v).
    configs = [("control", "handoff_last", 0.0, 0.0)]
    for pos in args.positions:
        for cv in args.cvs:
            if cv == 0:
                continue
            configs.append((pos, pos, cv, args.ck))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fieldnames = ["group", "positions", "coef", "ck", "last_k", "n", "accuracy",
                  "correct", "mean_output_tokens", "sec", "sec_per_sample"]
    rows = []
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (group, pos, cv, ck) in configs:
            res = evaluate_config(method, model, dataset, args.generate_bs,
                                  positions=pos, c_v=cv, c_k=ck,
                                  last_k=args.last_k, seed=args.seed)
            row = {"group": group, "positions": pos, "coef": cv, "ck": ck,
                   "last_k": args.last_k, **res}
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"[kv-sweep] {group:>14} pos={pos:<14} c_v={cv:>5} c_k={ck} -> "
                  f"acc={res['accuracy']*100:.1f}% tokens={res['mean_output_tokens']:.1f} "
                  f"({res['sec']}s)", flush=True)

    with open(args.out_csv.replace(".csv", ".json"), "w") as f:
        json.dump({"meta": {"task": args.task, "split": args.split, "n": args.n,
                            "model": args.model_name, "seed": args.seed,
                            "kvsteer_vector": args.kvsteer_vector,
                            "positions": args.positions, "cvs": args.cvs, "ck": args.ck},
                   "rows": rows}, f, indent=2)
    print(f"[kv-sweep] done -> {args.out_csv}")


if __name__ == "__main__":
    main()
