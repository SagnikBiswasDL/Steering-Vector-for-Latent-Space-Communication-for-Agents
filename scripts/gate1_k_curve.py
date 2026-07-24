#!/usr/bin/env python3
"""Gate 1: unsteered LatentMAS accuracy–compute curve over latent-step budgets K.

Primary compute metric: wall-clock latency per example.
Secondary: latent_forwards ≈ n_upstream * (K + 1) — not equal-cost (KV grows).

Selection rule: paired bootstrap CIs on accuracy differences; choose K_full /
K_small with a credible accuracy loss AND meaningful latency reduction.
No hard-coded point-gap threshold.

Examples:
  # Probe only
  python scripts/gate1_k_curve.py --task aime2024 --k_grid 0,10,40 ...

  # Primary: pooled AIME
  python scripts/gate1_k_curve.py --task aime_pooled --k_grid 0,5,10,20,40 ...

  # Primary alternative
  python scripts/gate1_k_curve.py --task medqa --k_grid 0,5,10,20,40 --max_samples 100 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import (  # noqa: E402
    load_aime2024,
    load_aime2025,
    load_aime_pooled,
    load_gsm8k,
    load_medqa,
)
from methods.latent_mas import LatentMASMethod  # noqa: E402
from models import ModelWrapper  # noqa: E402
from utils import set_seed, auto_device  # noqa: E402


def load_task(task: str, split: str, max_samples: int) -> List[Dict]:
    if task == "gsm8k":
        it = load_gsm8k(split=split)
    elif task == "aime2024":
        it = load_aime2024(split="train")
    elif task == "aime2025":
        it = load_aime2025(split="train")
    elif task == "aime_pooled":
        it = load_aime_pooled()
    elif task == "medqa":
        it = load_medqa(split=split)
    else:
        raise ValueError(f"Unsupported Gate 1 task: {task}")
    items = list(it)
    if max_samples > 0:
        items = items[:max_samples]
    # Stable ids for pairing across K
    for i, ex in enumerate(items):
        ex["example_id"] = i
        ex.setdefault("source", task)
    return items


def peak_gpu_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) for the mean of `values` via percentile bootstrap."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = values.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_diff_ci(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict:
    """Paired accuracy difference A - B with bootstrap CI."""
    diff = correct_a.astype(np.float64) - correct_b.astype(np.float64)
    mean, lo, hi = bootstrap_mean_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "mean_diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "credible_positive": bool(lo > 0),  # A reliably better than B
        "credible_negative": bool(hi < 0),
    }


def run_one_k(
    method: LatentMASMethod,
    items: List[Dict],
    *,
    k: int,
    generate_bs: int,
) -> List[Dict]:
    method.latent_steps = int(k)
    rows: List[Dict] = []
    n_up = sum(1 for a in method.agents if a.role != "judger")
    latent_forwards = n_up * (int(k) + 1)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for start in tqdm(range(0, len(items), generate_bs), desc=f"K={k}"):
        batch = items[start : start + generate_bs]
        t0 = time.perf_counter()
        try:
            outs = method.run_batch(batch)
            err = None
        except Exception as e:  # noqa: BLE001 — record failures, do not drop
            outs = [
                {
                    "prediction": "",
                    "gold": it.get("gold", ""),
                    "correct": False,
                    "output_tokens": 0,
                    "raw_prediction": "",
                    "error": str(e),
                }
                for it in batch
            ]
            err = str(e)
        elapsed = time.perf_counter() - t0
        per = elapsed / max(1, len(batch))
        for it, out in zip(batch, outs):
            rows.append(
                {
                    "example_id": it["example_id"],
                    "source": it.get("source", ""),
                    "question": it["question"],
                    "gold": it.get("gold", ""),
                    "k": int(k),
                    "correct": bool(out.get("correct", False)),
                    "prediction": out.get("prediction", ""),
                    "output_tokens": int(out.get("output_tokens", 0) or 0),
                    "latency_sec": float(per),
                    "batch_latency_sec": float(elapsed),
                    "latent_forwards": int(latent_forwards),
                    "n_upstream": int(n_up),
                    "error": out.get("error", err),
                }
            )
    return rows


def summarize_k(rows: List[Dict], *, seed: int) -> Dict:
    correct = np.array([1.0 if r["correct"] else 0.0 for r in rows], dtype=np.float64)
    latency = np.array([r["latency_sec"] for r in rows], dtype=np.float64)
    toks = np.array([r["output_tokens"] for r in rows], dtype=np.float64)
    acc_m, acc_lo, acc_hi = bootstrap_mean_ci(correct, seed=seed)
    lat_m, lat_lo, lat_hi = bootstrap_mean_ci(latency, seed=seed + 1)
    return {
        "k": int(rows[0]["k"]) if rows else None,
        "n": len(rows),
        "accuracy": acc_m,
        "accuracy_ci": [acc_lo, acc_hi],
        "mean_latency_sec": lat_m,
        "latency_ci": [lat_lo, lat_hi],
        "mean_output_tokens": float(toks.mean()) if len(toks) else float("nan"),
        "latent_forwards": int(rows[0]["latent_forwards"]) if rows else None,
        "n_failures": sum(1 for r in rows if r.get("error")),
        "n_no_answer": sum(1 for r in rows if not str(r.get("prediction", "")).strip()),
    }


def suggest_budgets(summaries: List[Dict], per_k_correct: Dict[int, np.ndarray], *, seed: int) -> Dict:
    """Heuristic: K_full = highest-K with best (or near-best) accuracy;
    K_small = lowest K with credible accuracy loss vs K_full and lower latency.
    """
    if not summaries:
        return {"k_full": None, "k_small": None, "reason": "no data"}
    by_k = {s["k"]: s for s in summaries}
    ks = sorted(by_k.keys())
    k_full = ks[-1]
    best_acc = max(s["accuracy"] for s in summaries)
    # Prefer largest K among those within 1 SE-ish of best; use CI overlap loosely
    for k in reversed(ks):
        if by_k[k]["accuracy"] >= best_acc - 0.02:
            k_full = k
            break

    candidates = []
    for k in ks:
        if k >= k_full:
            continue
        paired = paired_diff_ci(per_k_correct[k_full], per_k_correct[k], seed=seed)
        lat_ratio = by_k[k]["mean_latency_sec"] / max(by_k[k_full]["mean_latency_sec"], 1e-9)
        latency_saving = 1.0 - lat_ratio
        # Credible loss means K_full - K_small > 0 with CI
        credible_loss = paired["credible_positive"]
        meaningful_latency = latency_saving >= 0.15  # ≥15% wall-clock reduction
        candidates.append(
            {
                "k_small": k,
                "k_full": k_full,
                "paired_acc_diff_full_minus_small": paired,
                "latency_reduction": latency_saving,
                "credible_accuracy_loss": credible_loss,
                "meaningful_latency_reduction": meaningful_latency,
                "selected": bool(credible_loss and meaningful_latency),
            }
        )
    selected = [c for c in candidates if c["selected"]]
    # Prefer the smallest selected K_small that still has meaningful latency saving
    pick = selected[-1] if selected else (candidates[0] if candidates else None)
    return {
        "k_full": k_full,
        "k_small": (pick["k_small"] if pick else None),
        "candidates": candidates,
        "recommendation": pick,
        "note": (
            "Selected via paired bootstrap CI (credible accuracy loss) and "
            "≥15% wall-clock latency reduction; not a fixed point-gap rule."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate 1: LatentMAS K accuracy–compute curve")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B",
                    choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-14B"])
    ap.add_argument("--task", type=str, default="aime_pooled",
                    choices=["aime2024", "aime2025", "aime_pooled", "medqa", "gsm8k"])
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--k_grid", type=str, default="0,5,10,20,40")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--generate_bs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="<=0 uses greedy decoding")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", type=str, default="sequential", choices=["sequential", "hierarchical"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="artifacts/gate1")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print(
            "ERROR: CUDA GPU required for Gate 1 LatentMAS runs. "
            "Provide GPU info / host when ready; scaffolding is otherwise complete.",
            file=sys.stderr,
        )
        sys.exit(2)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip() != ""]
    items = load_task(args.task, args.split, args.max_samples)
    print(f"[gate1] task={args.task} n={len(items)} k_grid={k_grid} model={args.model_name}")

    device = auto_device(args.device)
    # Minimal namespace for ModelWrapper / LatentMASMethod
    ns = argparse.Namespace(
        method="latent_mas",
        model_name=args.model_name,
        task=args.task if args.task != "aime_pooled" else "aime2024",
        prompt=args.prompt,
        think=False,
        latent_space_realign=False,
        use_vllm=False,
        seal=False,
        kvsteer=False,
        capture_acts=None,
        ces=False,
        agents=None,
        device=str(device),
        device2="cuda:1",
        enable_prefix_caching=False,
        use_second_HF_model=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        text_mas_context_length=-1,
        latent_only=False,
        sequential_info_only=False,
        max_new_tokens=args.max_new_tokens,
    )
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=ns)
    method = LatentMASMethod(
        model,
        latent_steps=0,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        args=ns,
    )

    all_rows: List[Dict] = []
    summaries: List[Dict] = []
    per_k_correct: Dict[int, np.ndarray] = {}

    for k in k_grid:
        rows = run_one_k(method, items, k=k, generate_bs=args.generate_bs)
        # Attach peak mem for this K
        mem = peak_gpu_mem_gb()
        for r in rows:
            r["peak_gpu_mem_gb"] = mem
        all_rows.extend(rows)
        # Align by example_id
        rows_sorted = sorted(rows, key=lambda r: r["example_id"])
        per_k_correct[k] = np.array([1.0 if r["correct"] else 0.0 for r in rows_sorted])
        summ = summarize_k(rows_sorted, seed=args.seed)
        summ["peak_gpu_mem_gb"] = mem
        summaries.append(summ)
        print(json.dumps(summ, ensure_ascii=False))

    # Paired comparisons vs max K
    k_ref = max(k_grid)
    paired = {}
    for k in k_grid:
        if k == k_ref:
            continue
        paired[str(k)] = paired_diff_ci(
            per_k_correct[k_ref], per_k_correct[k], n_boot=args.n_boot, seed=args.seed
        )

    suggestion = suggest_budgets(summaries, per_k_correct, seed=args.seed)

    # Persist
    csv_path = os.path.join(args.out_dir, "per_example.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else ["example_id"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    report = {
        "config": vars(args),
        "n_examples": len(items),
        "summaries": summaries,
        "paired_vs_max_k": paired,
        "budget_suggestion": suggestion,
        "primary_compute_metric": "mean_latency_sec",
        "secondary_compute_metric": "latent_forwards (unequal cost; KV grows)",
    }
    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Lightweight plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        ks = [s["k"] for s in summaries]
        acc = [s["accuracy"] for s in summaries]
        lat = [s["mean_latency_sec"] for s in summaries]
        fwd = [s["latent_forwards"] for s in summaries]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ks, acc, marker="o")
        axes[0].set_xlabel("per-agent K")
        axes[0].set_ylabel("accuracy")
        axes[0].set_title("Accuracy vs K")
        axes[1].plot(lat, acc, marker="o")
        for k, x, y in zip(ks, lat, acc):
            axes[1].annotate(f"K={k}\n(fwd={dict(zip(ks, fwd))[k]})", (x, y), fontsize=8)
        axes[1].set_xlabel("mean latency (sec/example) [primary]")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Accuracy vs latency")
        fig.tight_layout()
        plot_path = os.path.join(args.out_dir, "k_curve.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[gate1] wrote {plot_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[gate1] plot skipped: {e}")

    print(f"[gate1] wrote {csv_path}")
    print(f"[gate1] wrote {json_path}")
    print(json.dumps({"budget_suggestion": suggestion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
