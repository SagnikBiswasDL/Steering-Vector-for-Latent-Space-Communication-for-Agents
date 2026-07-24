#!/usr/bin/env python3
"""Held-out MedQA test eval for Claim B (frozen indices [0,100)).

Compares:
  K_full unsteered | K_low unsteered | K_low + CES vector

Reports recovery_fraction and paired bootstrap CIs.

Example:
  python scripts/eval_ces_heldout.py \\
    --model_name Qwen/Qwen3-14B --ces_vector artifacts/ces/train_.../ces_latest.pt \\
    --k_full 10 --k_low 5 --max_samples 100 --out_dir artifacts/ces/heldout_14b
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


def build_steerer(wrapper, *, layer, coef, agents, steer_phase, apply_to, vector, init_std=0.0):
    """Construct + register a steerer directly so we can pass apply_to (attach_ces on
    the pod lacks that arg). Sets wrapper.ces and returns it (disabled)."""
    hidden = int(wrapper.model.config.hidden_size)
    ces = TrainableSteerer(
        hidden_size=hidden, layer_index=int(layer), coef=float(coef), init_std=float(init_std),
        agents=set(agents), steer_phase=steer_phase, apply_to=apply_to,
    )
    if vector is not None:
        with torch.no_grad():
            ces.v.copy_(vector.float().view(-1))
    ces.register(wrapper.model)
    ces.to(wrapper.device)
    ces.disable()
    wrapper.ces = ces
    return ces


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        means.append(x[idx].mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m, lo, hi = bootstrap_mean_ci(diff, n_boot=n_boot, seed=seed)
    return {
        "mean_diff": m,
        "ci_lo": lo,
        "ci_hi": hi,
        "credible_positive": lo > 0,
        "credible_negative": hi < 0,
    }


def make_ns(args) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name,
        task="medqa",
        prompt="sequential",
        think=False,
        latent_only=False,
        sequential_info_only=False,
        agents=None,
        use_vllm=False,
        device=args.device,
        device2="cuda:1",
        max_new_tokens=args.max_new_tokens,
        text_mas_context_length=-1,
        temperature=args.temperature,
        top_p=1.0,
        seed=args.seed,
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


def _acc_array(rows: List[Dict]) -> np.ndarray:
    return np.array([1.0 if r["correct"] else 0.0 for r in rows], dtype=float)


def _expand_controls(control: str) -> set:
    if control == "all":
        return {"negate", "random", "zero"}
    if control == "none":
        return set()
    return {control}


def eval_config(method: LatentMASMethod, items: List[Dict], *, k: int, label: str) -> List[Dict]:
    method.latent_steps = int(k)
    method.planner_steps = method.critic_steps = method.refiner_steps = None
    rows = []
    for i, item in enumerate(items):
        t0 = time.perf_counter()
        out = method.run_batch([item])[0]
        dt = time.perf_counter() - t0
        rows.append(
            {
                "idx": item.get("idx", i),
                "label": label,
                "k": k,
                "correct": bool(out.get("correct")),
                "prediction": out.get("prediction", ""),
                "output_tokens": int(out.get("output_tokens", 0)),
                "latency_sec": dt,
                "latent_forwards": method.latent_forwards(),
            }
        )
        if (i + 1) % 5 == 0 or i == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"[{label}] {i+1}/{len(items)} acc={acc:.3f}", flush=True)
    return rows


def run_recovery(args, wrapper, method, items, ckpt, layer, coef, v) -> None:
    """full / low / steered comparison + recovery_fraction (original claim-B path)."""
    all_rows: Dict[str, List[Dict]] = {}
    if not args.skip_full:
        if getattr(wrapper, "ces", None) is not None:
            wrapper.ces.disable()
        all_rows["full"] = eval_config(method, items, k=args.k_full, label="full")

    # low unsteered
    if getattr(wrapper, "ces", None) is not None:
        wrapper.ces.disable()
    all_rows["low"] = eval_config(method, items, k=args.k_low, label="low")

    # low + CES
    build_steerer(
        wrapper,
        layer=layer,
        coef=coef,
        agents=set(ckpt.get("agents", ["planner", "critic", "refiner"])),
        steer_phase=ckpt.get("steer_phase", "latent_only"),
        apply_to=ckpt.get("apply_to", "last"),
        vector=v.float(),
    )
    wrapper.ces.enable()
    all_rows["steered"] = eval_config(method, items, k=args.k_low, label="steered")

    a_low = _acc_array(all_rows["low"])
    a_steer = _acc_array(all_rows["steered"])
    if "full" in all_rows:
        a_full = _acc_array(all_rows["full"])
        full_acc = float(a_full.mean())
    else:
        a_full = None
        full_acc = float(args.full_acc_hint)

    low_acc = float(a_low.mean())
    steer_acc = float(a_steer.mean())
    gap = full_acc - low_acc
    recovery = (steer_acc - low_acc) / gap if abs(gap) > 1e-9 else float("nan")

    report = {
        "config": vars(args),
        "mode": "recovery",
        "n": len(items),
        "full_acc": full_acc,
        "low_acc": low_acc,
        "steered_acc": steer_acc,
        "recovery_fraction": recovery,
        "paired_steered_minus_low": paired_diff_ci(a_steer, a_low, seed=args.seed),
        "mean_tokens": {
            k: float(np.mean([r["output_tokens"] for r in rows])) for k, rows in all_rows.items()
        },
        "mean_latency": {
            k: float(np.mean([r["latency_sec"] for r in rows])) for k, rows in all_rows.items()
        },
        "latent_forwards": {
            "full": 3 * (args.k_full + 1),
            "low": 3 * (args.k_low + 1),
        },
        "decision": (
            "STRONG" if recovery >= 0.8 else
            "MODERATE" if recovery >= 0.4 else
            "FAIL"
        ),
    }
    if a_full is not None:
        report["paired_full_minus_steered"] = paired_diff_ci(a_full, a_steer, seed=args.seed)

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(all_rows, f)
    print(json.dumps(report, indent=2), flush=True)


def run_boost(args, wrapper, method, items, ckpt, layer, coef, v) -> None:
    """Boost framing: unsteered vs steered@K (=k_low) with causal controls.

    Arms:
      unsteered        - CES off (baseline at the operating point)
      steered_pos      - trained v, coef=+|coef| (the effect we care about)
      steered_neg      - trained v, coef=-|coef|   (negate control; should not help)
      steered_random   - matched-norm random v, coef=+|coef| (random control)
      steered_zero     - trained v, coef=0          (alpha=0 sanity; must equal unsteered)

    One steerer is attached once and its v/coef are swapped per arm so hooks never stack.
    """
    coef = abs(float(coef))
    K = int(args.k_low)
    v = v.float().view(-1)
    controls = _expand_controls(args.control)
    agents = set(ckpt.get("agents", ["planner", "critic", "refiner"]))
    steer_phase = ckpt.get("steer_phase", "latent_only")
    apply_to = ckpt.get("apply_to", "last")

    # Clean baseline: make sure no steerer is registered.
    if getattr(wrapper, "ces", None) is not None:
        wrapper.ces.remove()
        wrapper.ces = None

    arms: Dict[str, List[Dict]] = {}
    arms["unsteered"] = eval_config(method, items, k=K, label="unsteered")

    ces = build_steerer(
        wrapper, layer=layer, coef=coef, agents=agents,
        steer_phase=steer_phase, apply_to=apply_to, vector=v,
    )

    def _set_arm(vec: torch.Tensor, arm_coef: float) -> None:
        with torch.no_grad():
            ces.v.copy_(vec.float().view(-1).to(ces.v.device))
        ces.coef = float(arm_coef)
        ces.enable()

    _set_arm(v, coef)
    arms["steered_pos"] = eval_config(method, items, k=K, label="steered_pos")

    if "negate" in controls:
        _set_arm(v, -coef)
        arms["steered_neg"] = eval_config(method, items, k=K, label="steered_neg")

    if "random" in controls:
        gen = torch.Generator().manual_seed(int(args.seed))
        rv = torch.randn(v.numel(), generator=gen)
        rv = rv / rv.norm().clamp_min(1e-8) * v.norm()
        _set_arm(rv, coef)
        arms["steered_random"] = eval_config(method, items, k=K, label="steered_random")

    if "zero" in controls:
        _set_arm(v, 0.0)
        arms["steered_zero"] = eval_config(method, items, k=K, label="steered_zero")

    ces.disable()

    def _summ(rows: List[Dict]) -> Dict:
        acc, lo, hi = bootstrap_mean_ci(_acc_array(rows), seed=args.seed)
        return {
            "acc": acc,
            "acc_ci": [lo, hi],
            "mean_tokens": float(np.mean([r["output_tokens"] for r in rows])),
            "mean_latency": float(np.mean([r["latency_sec"] for r in rows])),
        }

    a_unsteered = _acc_array(arms["unsteered"])
    arm_summary = {name: _summ(rows) for name, rows in arms.items()}
    paired = {
        name: paired_diff_ci(_acc_array(rows), a_unsteered, seed=args.seed)
        for name, rows in arms.items()
        if name != "unsteered"
    }

    # alpha=0 sanity: predictions must match the unsteered arm exactly (greedy decode).
    zero_sanity_ok = None
    if "steered_zero" in arms:
        zero_sanity_ok = all(
            a.get("prediction") == b.get("prediction")
            for a, b in zip(arms["unsteered"], arms["steered_zero"])
        )

    pos = paired.get("steered_pos", {})
    decision = (
        "POSITIVE" if pos.get("credible_positive")
        else "NEGATIVE" if pos.get("credible_negative")
        else "NULL"
    )

    report = {
        "config": vars(args),
        "mode": "boost",
        "n": len(items),
        "k": K,
        "layer": layer,
        "coef": coef,
        "vector_norm": float(v.norm()),
        "arms": arm_summary,
        "paired_vs_unsteered": paired,
        "zero_sanity_ok": zero_sanity_ok,
        "latent_forwards": 3 * (K + 1),
        "decision": decision,
    }

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(arms, f)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--ces_vector", required=True)
    ap.add_argument("--k_full", type=int, default=10)
    ap.add_argument("--k_low", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--split", default="test", choices=["test", "train", "dev"],
                    help="MedQA split to evaluate. Use 'dev' for the Phase-A go/no-go so the "
                         "test split [0,100) stays held out; 'test' for the reported result.")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ces_layer", type=int, default=-1, help="-1 = use artifact layer")
    ap.add_argument("--ces_coef", type=float, default=-1.0, help="-1 = use artifact coef")
    ap.add_argument("--skip_full", action="store_true", help="Reuse gate1 full numbers; only run low+/-steer")
    ap.add_argument("--full_acc_hint", type=float, default=0.83)
    ap.add_argument("--mode", default="recovery", choices=["recovery", "boost"],
                    help="recovery: full/low/steered + recovery_fraction. "
                         "boost: unsteered vs steered@K (=k_low) with causal controls.")
    ap.add_argument("--control", default="none", choices=["none", "negate", "random", "zero", "all"],
                    help="Boost-mode causal controls: negate (alpha<0), random (matched-norm v), "
                         "zero (alpha=0 sanity), or all.")
    ap.add_argument("--out_dir", default="artifacts/ces/heldout_claim_b")
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
    layer = int(args.ces_layer) if args.ces_layer >= 0 else int(ckpt.get("layer_index", 28))
    coef = float(args.ces_coef) if args.ces_coef >= 0 else float(ckpt.get("coef", 1.0))
    v = ckpt["v"] if isinstance(ckpt, dict) else ckpt

    ns = make_ns(args)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    method = LatentMASMethod(
        wrapper,
        latent_steps=args.k_low,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        generate_bs=1,
        args=ns,
    )

    if args.mode == "boost":
        run_boost(args, wrapper, method, items, ckpt, layer, coef, v)
    else:
        run_recovery(args, wrapper, method, items, ckpt, layer, coef, v)


if __name__ == "__main__":
    main()
