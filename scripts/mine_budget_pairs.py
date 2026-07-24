#!/usr/bin/env python3
"""Mine (K_full vs K_low) pairs for Claim B training.

Uses frozen MedQA splits: train=[100,220), never touches test=[0,100).

Lean 4–6h default: --max_samples 50 on train (≈40–60 min on H200 @14B).

Example:
  python scripts/mine_budget_pairs.py \\
    --model_name Qwen/Qwen3-14B --k_full 10 --k_low 5 \\
    --split train --max_samples 50 --max_new_tokens 4096 \\
    --out_dir artifacts/ces/pairs_k10_vs_k5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_medqa  # noqa: E402
from methods.latent_mas import LatentMASMethod  # noqa: E402
from models import ModelWrapper  # noqa: E402
from utils import set_seed, auto_device  # noqa: E402


def load_items(split: str, max_samples: int, skip_samples: int = 0) -> List[Dict]:
    items = list(load_medqa(split=split))
    if skip_samples > 0:
        items = items[skip_samples:]
    if max_samples > 0:
        items = items[:max_samples]
    return items


def make_args(model_name: str, max_new_tokens: int, temperature: float, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        task="medqa",
        prompt="sequential",
        think=False,
        latent_only=False,
        sequential_info_only=False,
        agents=None,
        use_vllm=False,
        device="cuda",
        device2="cuda:1",
        max_new_tokens=max_new_tokens,
        text_mas_context_length=-1,
        temperature=temperature,
        top_p=1.0,
        seed=seed,
        seal=False,
        kvsteer=False,
        ces=False,
        capture_acts=None,
        planner_steps=None,
        critic_steps=None,
        refiner_steps=None,
        latent_steps=0,
        latent_space_realign=False,
    )


def run_budget(
    method: LatentMASMethod,
    items: List[Dict],
    *,
    k: int,
    generate_bs: int,
) -> List[Dict]:
    method.latent_steps = int(k)
    method.planner_steps = None
    method.critic_steps = None
    method.refiner_steps = None
    rows: List[Dict] = []
    for i in range(0, len(items), generate_bs):
        batch = items[i : i + generate_bs]
        t0 = time.perf_counter()
        outs = method.run_batch(batch)
        dt = time.perf_counter() - t0
        per = dt / max(len(batch), 1)
        for item, out in zip(batch, outs):
            rows.append(
                {
                    "idx": item.get("idx", i),
                    "question": item["question"],
                    "gold": item["gold"],
                    "solution": item.get("solution", item["gold"]),
                    "k": k,
                    "correct": bool(out.get("correct")),
                    "prediction": out.get("prediction", ""),
                    "raw_prediction": out.get("raw_prediction", ""),
                    "output_tokens": int(out.get("output_tokens", 0)),
                    "latency_sec": per,
                    "latent_forwards": method.latent_forwards(),
                    "budget": list(method.budget_tuple()),
                }
            )
        print(
            f"[mine] K={k} {min(i + generate_bs, len(items))}/{len(items)} "
            f"acc_so_far={sum(r['correct'] for r in rows)/len(rows):.3f}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--skip_samples", type=int, default=0,
                    help="Skip first N examples of the split (for dual-GPU continuation)")
    ap.add_argument("--k_full", type=int, default=10)
    ap.add_argument("--k_low", type=int, default=5)
    ap.add_argument("--only_k", type=int, default=None,
                    help="If set, run only this K and write rows_K{k}.json (for dual-GPU split mining)")
    ap.add_argument("--single_k", type=int, default=None,
                    help="Boost framing: mine correctness pairs at ONE K. Writes pairs.json with "
                         "y+=gold and y-=model prediction (distractor fallback applied by the trainer "
                         "when the model was already correct). Wrong items are listed first.")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--generate_bs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/ces/pairs_k10_vs_k5")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    items = load_items(args.split, args.max_samples, args.skip_samples)
    print(
        f"[mine] split={args.split} skip={args.skip_samples} n={len(items)} "
        f"model={args.model_name}",
        flush=True,
    )

    ns = make_args(args.model_name, args.max_new_tokens, args.temperature, args.seed)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    method = LatentMASMethod(
        wrapper,
        latent_steps=int(args.only_k or args.k_full),
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        generate_bs=args.generate_bs,
        args=ns,
    )

    def save_rows(k: int, rows: List[Dict], tag: str = "") -> str:
        suffix = tag if tag else ""
        path = os.path.join(args.out_dir, f"rows_K{k}{suffix}.json")
        with open(path, "w") as f:
            json.dump(rows, f)
        print(f"[mine] wrote {path} n={len(rows)}", flush=True)
        return path

    shard_tag = f"_skip{args.skip_samples}" if args.skip_samples else ""

    if args.single_k is not None:
        k = int(args.single_k)
        rows = run_budget(method, items, k=k, generate_bs=args.generate_bs)
        save_rows(k, rows, tag=shard_tag)
        # Boost-framing pairs: every item is a training example.
        #   y+ = gold, y- = model prediction (the trainer swaps in a distractor
        #   letter when the model was already correct, i.e. prediction == gold).
        pairs: List[Dict] = []
        for r in rows:
            pairs.append(
                {
                    "idx": r["idx"],
                    "question": r["question"],
                    "gold": r["gold"],
                    "solution": r["solution"],
                    "k": k,
                    "correct": bool(r["correct"]),
                    "prediction": r["prediction"],
                    # trainer reads `low_prediction` as the negative trajectory
                    "low_prediction": r["prediction"],
                    "output_tokens": r["output_tokens"],
                }
            )
        # Wrong items first: they carry the informative negatives for boost training,
        # so `--max_pairs N` in the trainer prioritizes real failures.
        pairs.sort(key=lambda p: (p["correct"], p["idx"]))
        n_wrong = sum(1 for p in pairs if not p["correct"])
        with open(os.path.join(args.out_dir, "pairs.json"), "w") as f:
            json.dump(pairs, f)
        with open(os.path.join(args.out_dir, "all_joined.json"), "w") as f:
            json.dump(pairs, f)
        summary = {
            "config": vars(args),
            "mode": "single_k",
            "single_k": k,
            "n": len(pairs),
            "acc": sum(r["correct"] for r in rows) / max(len(rows), 1),
            "n_wrong": n_wrong,
            "n_correct": len(pairs) - n_wrong,
            "latent_forwards": 3 * (k + 1),
        }
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        print(
            f"[mine] wrote {args.out_dir}  single_k={k} pairs={len(pairs)} wrong={n_wrong}",
            flush=True,
        )
        return

    if args.only_k is not None:
        rows = run_budget(method, items, k=int(args.only_k), generate_bs=args.generate_bs)
        save_rows(int(args.only_k), rows, tag=shard_tag)
        summary = {
            "config": vars(args),
            "only_k": int(args.only_k),
            "n": len(rows),
            "acc": sum(r["correct"] for r in rows) / max(len(rows), 1),
        }
        with open(os.path.join(args.out_dir, f"summary_K{args.only_k}{shard_tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        return

    full_rows = run_budget(method, items, k=args.k_full, generate_bs=args.generate_bs)
    save_rows(args.k_full, full_rows, tag=shard_tag)
    low_rows = run_budget(method, items, k=args.k_low, generate_bs=args.generate_bs)
    save_rows(args.k_low, low_rows, tag=shard_tag)

    by_idx_full = {r["idx"]: r for r in full_rows}
    by_idx_low = {r["idx"]: r for r in low_rows}
    pairs: List[Dict] = []
    all_joined: List[Dict] = []
    for idx in sorted(set(by_idx_full) & set(by_idx_low)):
        f, lo = by_idx_full[idx], by_idx_low[idx]
        joined = {
            "idx": idx,
            "question": f["question"],
            "gold": f["gold"],
            "solution": f["solution"],
            "full_correct": f["correct"],
            "low_correct": lo["correct"],
            "full_prediction": f["prediction"],
            "low_prediction": lo["prediction"],
            "full_raw": f["raw_prediction"],
            "low_raw": lo["raw_prediction"],
            "full_tokens": f["output_tokens"],
            "low_tokens": lo["output_tokens"],
            "full_latency": f["latency_sec"],
            "low_latency": lo["latency_sec"],
            "k_full": args.k_full,
            "k_low": args.k_low,
            "is_pair": bool(f["correct"] and not lo["correct"]),
        }
        all_joined.append(joined)
        if joined["is_pair"]:
            pairs.append(joined)

    summary = {
        "config": vars(args),
        "n": len(all_joined),
        "full_acc": sum(r["full_correct"] for r in all_joined) / max(len(all_joined), 1),
        "low_acc": sum(r["low_correct"] for r in all_joined) / max(len(all_joined), 1),
        "n_pairs": len(pairs),
        "latent_forwards_full": 3 * (args.k_full + 1),
        "latent_forwards_low": 3 * (args.k_low + 1),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "all_joined.json"), "w") as f:
        json.dump(all_joined, f)
    with open(os.path.join(args.out_dir, "pairs.json"), "w") as f:
        json.dump(pairs, f)
    with open(os.path.join(args.out_dir, "all_joined.csv"), "w", newline="") as f:
        if all_joined:
            w = csv.DictWriter(f, fieldnames=list(all_joined[0].keys()))
            w.writeheader()
            w.writerows(all_joined)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[mine] wrote {args.out_dir}  pairs={len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
