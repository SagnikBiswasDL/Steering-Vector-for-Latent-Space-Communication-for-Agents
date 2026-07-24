#!/usr/bin/env python3
"""Coef-sweep diagnostic for a trained CES vector.

Loads the model once, runs the unsteered baseline once, then re-applies the SAME
trained vector at several coefficients (and a negative) to answer: does *stronger*
application of the learned latent-steering direction move accuracy at all, or is the
near-zero effect a true null rather than a too-weak-application artifact?

Example:
  python scripts/coef_sweep.py --model_name Qwen/Qwen3-4B \
    --ces_vector artifacts/ces/boost_4b_tuned/train_k10_L20/ces_latest.pt \
    --k 10 --coefs 4,8,16,-8 --split dev --max_samples 40 --max_new_tokens 2048 \
    --out_dir artifacts/ces/coef_sweep_4b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_medqa  # noqa: E402
from methods.latent_mas import LatentMASMethod  # noqa: E402
from models import ModelWrapper  # noqa: E402
from seal.ces_steerer import TrainableSteerer  # noqa: E402
from utils import set_seed, auto_device  # noqa: E402


def build_steerer(wrapper, *, layer, coef, agents, steer_phase, apply_to, vector):
    """Construct + register a steerer directly (attach_ces lacks apply_to on the pod)."""
    hidden = int(wrapper.model.config.hidden_size)
    ces = TrainableSteerer(
        hidden_size=hidden, layer_index=int(layer), coef=float(coef), init_std=0.0,
        agents=set(agents), steer_phase=steer_phase, apply_to=apply_to,
    )
    with torch.no_grad():
        ces.v.copy_(vector.float().view(-1))
    ces.register(wrapper.model)
    ces.to(wrapper.device)
    ces.disable()
    wrapper.ces = ces
    return ces


def bootstrap_mean_ci(x, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a, b, n_boot: int = 2000, seed: int = 0):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    return {"mean_diff": m, "ci_lo": lo, "ci_hi": hi,
            "credible_positive": lo > 0, "credible_negative": hi < 0}


def make_ns(args) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name, task="medqa", prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=args.max_new_tokens,
        text_mas_context_length=-1, temperature=0.0, top_p=1.0, seed=args.seed,
        seal=False, kvsteer=False, ces=False, capture_acts=None, planner_steps=None,
        critic_steps=None, refiner_steps=None, latent_steps=0, latent_space_realign=False,
    )


def eval_arm(method, items, k, label) -> List[Dict]:
    rows = []
    for i, item in enumerate(items):
        t0 = time.perf_counter()
        out = method.run_batch([item])[0]
        dt = time.perf_counter() - t0
        rows.append({
            "idx": item.get("idx", i),
            "correct": bool(out.get("correct")),
            "prediction": out.get("prediction", ""),
            "output_tokens": int(out.get("output_tokens", 0)),
            "latency_sec": dt,
        })
        if (i + 1) % 10 == 0 or i == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"[{label}] {i+1}/{len(items)} acc={acc:.3f}", flush=True)
    return rows


def acc_arr(rows):
    return np.array([1.0 if r["correct"] else 0.0 for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B")
    ap.add_argument("--ces_vector", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--coefs", default="4,8,16,-8")
    ap.add_argument("--split", default="dev", choices=["test", "train", "dev"])
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--ces_layer", type=int, default=-1, help="-1 = artifact layer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/ces/coef_sweep")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    items = list(load_medqa(split=args.split))
    if args.max_samples > 0:
        items = items[: args.max_samples]

    ckpt = torch.load(args.ces_vector, map_location="cpu")
    is_d = isinstance(ckpt, dict)
    layer = int(args.ces_layer) if args.ces_layer >= 0 else int(ckpt.get("layer_index", 28) if is_d else 28)
    v = (ckpt["v"] if is_d else ckpt).float().view(-1)
    agents = set(ckpt.get("agents", ["planner", "critic", "refiner"])) if is_d else {"planner", "critic", "refiner"}
    steer_phase = ckpt.get("steer_phase", "latent_only") if is_d else "latent_only"
    apply_to = ckpt.get("apply_to", "last") if is_d else "last"
    coefs = [float(c) for c in args.coefs.split(",") if c.strip()]

    ns = make_ns(args)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    method = LatentMASMethod(
        wrapper, latent_steps=args.k, judger_max_new_tokens=args.max_new_tokens,
        temperature=0.0, top_p=1.0, generate_bs=1, args=ns,
    )
    method.latent_steps = args.k
    method.planner_steps = method.critic_steps = method.refiner_steps = None

    # unsteered baseline (no steerer registered)
    if getattr(wrapper, "ces", None) is not None:
        wrapper.ces.remove()
        wrapper.ces = None
    uns = eval_arm(method, items, args.k, "unsteered")
    a_uns = acc_arr(uns)

    ces = build_steerer(wrapper, layer=layer, coef=1.0, agents=agents,
                        steer_phase=steer_phase, apply_to=apply_to, vector=v)

    arms = {"unsteered": {"acc": float(a_uns.mean()), "n": len(uns),
                          "mean_tokens": float(np.mean([r["output_tokens"] for r in uns]))}}
    paired = {}
    for c in coefs:
        with torch.no_grad():
            ces.v.copy_(v.to(ces.v.device))
        ces.coef = float(c)
        ces.enable()
        rows = eval_arm(method, items, args.k, f"coef{c}")
        a = acc_arr(rows)
        arms[f"coef_{c}"] = {"acc": float(a.mean()), "n": len(rows),
                             "mean_tokens": float(np.mean([r["output_tokens"] for r in rows]))}
        paired[f"coef_{c}"] = paired_diff_ci(a, a_uns, seed=args.seed)
    ces.disable()

    report = {
        "model": args.model_name, "vector": args.ces_vector, "layer": layer,
        "apply_to": apply_to, "vector_norm": float(v.norm()), "k": args.k,
        "split": args.split, "n": len(items), "coefs": coefs, "arms": arms,
        "paired_vs_unsteered": paired,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"SWEEP_SUMMARY apply_to={apply_to} vector_norm={round(float(v.norm()),3)} unsteered={round(arms['unsteered']['acc'],3)}")
    for c in coefs:
        p = paired[f"coef_{c}"]
        print(f"SWEEP_SUMMARY coef={c} acc={round(arms['coef_'+str(c)]['acc'],3)} "
              f"diff={round(p['mean_diff'],3)} ci=[{round(p['ci_lo'],3)},{round(p['ci_hi'],3)}]")
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()
