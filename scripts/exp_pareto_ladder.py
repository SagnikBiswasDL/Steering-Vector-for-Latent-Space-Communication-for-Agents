#!/usr/bin/env python3
"""Memory–accuracy Pareto ladder over relay keep-budgets.

Build the upstream LatentMAS cache once per item, then compress with plain
eviction at each keep-budget B and decode the Judger. Arm "full" is the
uncompressed baseline (shared across the ladder).

Outputs:
  report.json  — per-budget accuracy / tokens / relay MB / positions
  rows.json    — per-item rows
  summary.csv  — one row per budget for plotting

Example:
  python scripts/exp_pareto_ladder.py --task gsm8k --n 60 \\
    --budgets 16,32,64,128 --judger_budget 768 --batch_size 20 \\
    --out_dir artifacts/exp2x2/pareto/gsm8k_s42
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from methods import default_agents  # noqa: E402
from models import ModelWrapper  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.relay_compress import RelayCompressor, kv_mb, num_positions  # noqa: E402
from utils import set_seed, auto_device  # noqa: E402


def _load_2x2():
    path = os.path.join(os.path.dirname(__file__), "exp_2x2_relay_asc.py")
    spec = importlib.util.spec_from_file_location("exp_2x2_relay_asc", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_x2 = _load_2x2()
load_task = _x2.load_task
make_ns = _x2.make_ns
build_upstream_cache = _x2.build_upstream_cache
decode_batch = _x2.decode_batch
graded = _x2.graded
reset_peak = _x2.reset_peak
peak_mem_mb = _x2.peak_mem_mb


def parse_budgets(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        b = int(tok)
        if b <= 0:
            raise ValueError(f"budget must be >0, got {b}")
        out.append(b)
    if not out:
        raise ValueError("empty --budgets")
    return sorted(set(out))


def bootstrap_ci(values, seed=42, n_boot=2000, alpha=0.05):
    """Mean and percentile CI for a 1-d array of floats."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        means[b] = rng.choice(x, size=x.size, replace=True).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def select_items(pool, n, seed, sample_items: bool):
    """Take n items. If sample_items, draw a seeded random subset from the full pool."""
    if n <= 0 or n >= len(pool):
        items = list(pool)
    elif sample_items:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), size=n, replace=False)
        # keep draw order (not sorted) so batch composition also varies by seed
        items = [pool[int(i)] for i in idx]
    else:
        items = list(pool[:n])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="gsm8k",
                    choices=["medqa", "gsm8k", "math", "aime2024", "aime_pooled"])
    ap.add_argument("--split", default="test", choices=["test", "train", "dev"])
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--judger_budget", type=int, default=768)
    ap.add_argument("--budgets", default="16,32,64,128",
                    help="comma-separated keep-budgets for plain eviction")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--importance", default="key_norm",
                    choices=["key_norm", "value_norm", "recency"])
    ap.add_argument("--with_none", action="store_true",
                    help="also decode Judger-only (no cache)")
    ap.add_argument("--sample_items", action="store_true",
                    help="seeded random subset of n from the full split (real multi-seed)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/exp2x2/pareto/run")
    # stubs so make_ns() from the 2x2 harness can read ASC fields when ASC is off
    ap.add_argument("--asc_vector", default=None)
    ap.add_argument("--asc_coef", type=float, default=40.0)
    ap.add_argument("--asc_layer", type=int, default=-1)
    ap.add_argument("--asc_apply_to", default="last")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)

    budgets = parse_budgets(args.budgets)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    pool = load_task(args.task, args.split)
    items = select_items(pool, args.n, args.seed, args.sample_items)
    n = len(items)
    print(
        f"[pareto] pool={len(pool)} selected={n} sample_items={bool(args.sample_items)} seed={args.seed}",
        flush=True,
    )
    # ASC off — ladder is memory×accuracy, not the 2×2 interaction
    ns = make_ns(args, run_asc=False)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    device = wrapper.device
    up_agents = [a for a in default_agents() if a.role != "judger"]

    compressors = {
        f"evict{b}": RelayCompressor(
            mode="evict", budget=b, sink=args.sink, importance=args.importance
        )
        for b in budgets
    }

    print(
        f"[pareto] task={args.task} n={n} k={args.k} judger_budget={args.judger_budget} "
        f"budgets={budgets} sink={args.sink} model={args.model_name}",
        flush=True,
    )

    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass

    # arm order: full, then each B ascending, optional none
    arm_specs = [("full", "full")] + [(f"B{b}", f"evict{b}") for b in budgets]
    if args.with_none:
        arm_specs.append(("none", "none"))

    rows: List[Dict] = []
    t0 = time.time()
    Bsz = max(1, int(args.batch_size))
    n_batches = (n + Bsz - 1) // Bsz
    print(f"[pareto] batch_size={Bsz} -> {n_batches} batches", flush=True)

    for bi in range(n_batches):
        batch = items[bi * Bsz:(bi + 1) * Bsz]
        bsz = len(batch)
        bidx = [it.get("idx", bi * Bsz + j) for j, it in enumerate(batch)]
        qs = [it["question"] for it in batch]
        golds = [it["gold"] for it in batch]

        real_caches, up_times = [], []
        cc_by_kind = {k: [] for k in compressors}
        comp_times = {k: [] for k in compressors}
        stats_by_kind = {"full": []}
        for k in compressors:
            stats_by_kind[k] = []

        for q in qs:
            t_up = time.perf_counter()
            rc = build_upstream_cache(wrapper, q, args.k, ns, up_agents)
            up_times.append(time.perf_counter() - t_up)
            real_caches.append(rc)
            stats_by_kind["full"].append({
                "positions_out": num_positions(rc),
                "mb_out": kv_mb(rc),
                "ratio": 1.0,
                "sink_retained": True,
            })
            for kind, comp in compressors.items():
                tc = time.perf_counter()
                cc, st = comp.compress(rc)
                comp_times[kind].append(time.perf_counter() - tc)
                cc_by_kind[kind].append(cc)
                stats_by_kind[kind].append(st.as_dict())

        jmsgs = [
            build_agent_message_sequential_latent_mas(
                role="judger", question=q, context="", method="latent_mas", args=ns
            )
            for q in qs
        ]
        _, jids_b, jmask_b, _ = wrapper.prepare_chat_batch(jmsgs, add_generation_prompt=True)

        cache_lists = {"full": real_caches}
        cache_lists.update(cc_by_kind)
        if args.with_none:
            cache_lists["none"] = [None] * bsz
            stats_by_kind["none"] = [
                {"positions_out": 0, "mb_out": 0.0, "ratio": 0.0, "sink_retained": True}
                for _ in range(bsz)
            ]

        for arm_name, kind in arm_specs:
            reset_peak(device)
            td = time.perf_counter()
            texts, ntoks, eoss = decode_batch(
                wrapper, jids_b, jmask_b, cache_lists[kind], False, args.judger_budget
            )
            dec_time = time.perf_counter() - td
            pmem = peak_mem_mb(device)
            per_item_dec = dec_time / max(1, bsz)
            for j in range(bsz):
                st = stats_by_kind[kind][j]
                ntok, text = int(ntoks[j]), texts[j]
                comp_s = comp_times[kind][j] if kind in comp_times else 0.0
                keep_b = None
                if kind.startswith("evict"):
                    keep_b = int(kind.replace("evict", ""))
                rows.append({
                    "idx": bidx[j],
                    "arm": arm_name,
                    "relay": kind,
                    "keep_budget": keep_b,
                    "correct": bool(graded(text, golds[j], args.task)),
                    "tokens": ntok,
                    "eos": bool(eoss[j]),
                    "decode_s": per_item_dec,
                    "e2e_s": up_times[j] + comp_s + per_item_dec,
                    "compress_s": comp_s,
                    "upstream_s": up_times[j],
                    "relay_positions": int(st["positions_out"]),
                    "relay_mb": float(st["mb_out"]),
                    "relay_ratio": float(st.get("ratio", 1.0)),
                    "peak_mem_mb": pmem,
                    "batch_size": bsz,
                })

        for rc in real_caches:
            del rc
        done = min((bi + 1) * Bsz, n)
        print(
            f"[pareto] batch {bi+1}/{n_batches} ({done}/{n}) "
            f"elapsed={time.time()-t0:.0f}s",
            flush=True,
        )

    # aggregate
    per_arm = {}
    for arm_name, kind in arm_specs:
        rs = [r for r in rows if r["arm"] == arm_name]
        toks = [r["tokens"] for r in rs]
        correct = [1.0 if r["correct"] else 0.0 for r in rs]
        acc_m, acc_lo, acc_hi = bootstrap_ci(correct, seed=args.seed)
        tok_m, tok_lo, tok_hi = bootstrap_ci(toks, seed=args.seed)
        per_arm[arm_name] = {
            "relay": kind,
            "keep_budget": rs[0]["keep_budget"] if rs else None,
            "n": len(rs),
            "acc": acc_m,
            "acc_ci95": [acc_lo, acc_hi],
            "mean_tokens": tok_m,
            "tokens_ci95": [tok_lo, tok_hi],
            "median_tokens": float(np.median(toks)) if toks else 0.0,
            "mean_decode_s": float(np.mean([r["decode_s"] for r in rs])) if rs else 0.0,
            "mean_e2e_s": float(np.mean([r["e2e_s"] for r in rs])) if rs else 0.0,
            "mean_relay_mb": float(np.mean([r["relay_mb"] for r in rs])) if rs else 0.0,
            "mean_relay_positions": float(np.mean([r["relay_positions"] for r in rs])) if rs else 0.0,
            "eos_rate": float(np.mean([r["eos"] for r in rs])) if rs else 0.0,
            "mean_peak_mem_mb": float(np.mean([r["peak_mem_mb"] for r in rs])) if rs else 0.0,
            "mean_compress_s": float(np.mean([r["compress_s"] for r in rs])) if rs else 0.0,
            "mean_upstream_s": float(np.mean([r["upstream_s"] for r in rs])) if rs else 0.0,
        }

    # vs full deltas (+ bootstrap on paired per-item gaps)
    full_by_idx = {r["idx"]: r for r in rows if r["arm"] == "full"}
    full_acc = per_arm.get("full", {}).get("acc", 0.0)
    full_mb = per_arm.get("full", {}).get("mean_relay_mb", 0.0)
    ladder = []
    for arm_name, _ in arm_specs:
        a = per_arm[arm_name]
        arm_rows = [r for r in rows if r["arm"] == arm_name]
        d_acc = []
        for r in arm_rows:
            fr = full_by_idx.get(r["idx"])
            if fr is None:
                continue
            d_acc.append(float(r["correct"]) - float(fr["correct"]))
        d_m, d_lo, d_hi = bootstrap_ci(d_acc, seed=args.seed) if d_acc else (0.0, 0.0, 0.0)
        ladder.append({
            "arm": arm_name,
            "keep_budget": a["keep_budget"],
            "acc": a["acc"],
            "acc_ci95_lo": a["acc_ci95"][0],
            "acc_ci95_hi": a["acc_ci95"][1],
            "d_acc_vs_full": d_m,
            "d_acc_ci95_lo": d_lo,
            "d_acc_ci95_hi": d_hi,
            "mean_tokens": a["mean_tokens"],
            "tokens_ci95_lo": a["tokens_ci95"][0],
            "tokens_ci95_hi": a["tokens_ci95"][1],
            "relay_mb": a["mean_relay_mb"],
            "relay_positions": a["mean_relay_positions"],
            "compress_ratio_vs_full": (full_mb / a["mean_relay_mb"]) if a["mean_relay_mb"] > 0 else None,
        })

    summary = {
        "config": {**vars(args), "budgets": budgets},
        "n": n,
        "arms": [a[0] for a in arm_specs],
        "per_arm": per_arm,
        "ladder": ladder,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(rows, f)

    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arm", "keep_budget", "acc", "acc_ci95_lo", "acc_ci95_hi",
                "d_acc_vs_full", "d_acc_ci95_lo", "d_acc_ci95_hi",
                "mean_tokens", "tokens_ci95_lo", "tokens_ci95_hi",
                "relay_mb", "relay_positions", "compress_ratio_vs_full",
            ],
        )
        w.writeheader()
        for row in ladder:
            w.writerow(row)

    print("\n=== PARETO LADDER ===", flush=True)
    print(f"{'arm':<8}{'B':<8}{'acc':<8}{'ci95':<18}{'dAcc':<9}{'tok':<8}{'MB':<10}{'pos':<8}{'ratio':<8}")
    for row in ladder:
        b = "-" if row["keep_budget"] is None else str(row["keep_budget"])
        ratio = "-" if row["compress_ratio_vs_full"] is None else f"{row['compress_ratio_vs_full']:.1f}x"
        ci = f"[{row['acc_ci95_lo']:.3f},{row['acc_ci95_hi']:.3f}]"
        print(
            f"{row['arm']:<8}{b:<8}{row['acc']:<8.3f}{ci:<18}"
            f"{row['d_acc_vs_full']:<+9.3f}"
            f"{row['mean_tokens']:<8.0f}{row['relay_mb']:<10.2f}"
            f"{row['relay_positions']:<8.0f}{ratio:<8}",
            flush=True,
        )
    print(f"[pareto] wrote {args.out_dir}/report.json (+ rows.json, summary.csv)", flush=True)
    print("PARETO_DONE", flush=True)


if __name__ == "__main__":
    main()
