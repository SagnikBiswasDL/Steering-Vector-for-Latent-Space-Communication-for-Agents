#!/usr/bin/env python3
"""Diagnostic: does the Judger actually USE the upstream latent cache, and is there
a token-budget regime where the latent agents matter?

For each item we build the real upstream cache (Planner->Critic->Refiner at K), then
decode the Judger under several cache interventions x Judger token budgets:

  conditions:
    real      - the item's own upstream cache
    shuffled  - another question's cache (cross-question; rolling derangement)
    zero      - the item's cache with all K/V zeroed (positions present, content gone)
    none      - no cache (judger-only baseline)

  budgets: Judger max_new_tokens in {64,128,256,512,1024,...}

Key comparisons (per budget):
  A(real) - A(shuffled)  -> does cache CONTENT matter? (readout usage)
  A(real) - A(none)      -> does the cache help at all vs judger-only?
  regime where real > none while unlimited real ~= none -> latent agents matter only
    when the Judger cannot re-derive with a long CoT.

NOTE: LatentMAS upstream rollout is deterministic, so "cross-run same-question" and
"correct-vs-incorrect rollout" caches collapse to `real`; they are intentionally not
included. Greedy decode (temp=0) is used so differences are causal, not sampling noise.

Example:
  python scripts/diag_cache_usage.py --model_name Qwen/Qwen3-14B --k 10 \
    --split test --n 60 --budgets 128,256,512,1024 \
    --conditions real,shuffled,zero,none --out_dir artifacts/diag/cache_usage_14b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_medqa, load_gsm8k  # noqa: E402
from methods import default_agents  # noqa: E402
from models import ModelWrapper, _past_length  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from utils import set_seed, auto_device, extract_gsm8k_answer, normalize_answer  # noqa: E402

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def load_task(task: str, split: str):
    if task == "medqa":
        return list(load_medqa(split=split))
    if task == "gsm8k":
        return list(load_gsm8k(split=split))
    raise ValueError(f"unknown task {task}")


def make_ns(args) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name, task=args.task, prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=max(args.budgets_list),
        text_mas_context_length=-1, temperature=0.0, top_p=1.0, seed=args.seed,
        seal=False, kvsteer=False, ces=False, capture_acts=None, planner_steps=None,
        critic_steps=None, refiner_steps=None, latent_steps=0, latent_space_realign=False,
    )


def _map_past(past, fn):
    """Apply fn to every tensor in a KV cache, returning a NEW cache of same type."""
    if past is None:
        return None
    if Cache is not None and isinstance(past, Cache):
        legacy = past.to_legacy_cache()
        new = tuple(tuple(fn(t) for t in layer) for layer in legacy)
        return past.__class__.from_legacy_cache(new)
    out = []
    for layer in past:
        if isinstance(layer, tuple):
            out.append(tuple(fn(t) for t in layer))
        elif torch.is_tensor(layer):
            out.append(fn(layer))
        else:
            out.append(layer)
    return tuple(out)


def clone_past(past):
    return _map_past(past, lambda t: t.clone())


def zero_past(past):
    return _map_past(past, lambda t: torch.zeros_like(t))


def to_cpu_past(past):
    return _map_past(past, lambda t: t.detach().to("cpu"))


def to_dev_past(past, device):
    return _map_past(past, lambda t: t.to(device))


def build_upstream_cache(wrapper, question: str, k: int, ns, agents):
    past = None
    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=question, context="", method="latent_mas", args=ns
        )
        _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        past = wrapper.generate_latent_batch(
            input_ids, attention_mask=attention_mask, latent_steps=int(k),
            past_key_values=past, role=agent.role,
        )
    return past


def judger_inputs(wrapper, question: str, ns):
    messages = build_agent_message_sequential_latent_mas(
        role="judger", question=question, context="", method="latent_mas", args=ns
    )
    _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
    return ids, mask


def decode_judger(wrapper, ids, mask, past, max_new_tokens):
    gens, _ = wrapper.generate_text_batch(
        ids, mask, max_new_tokens=int(max_new_tokens),
        temperature=0.0, top_p=1.0, past_key_values=past, role="judger",
    )
    toks = list(getattr(wrapper, "last_gen_token_counts", [0]))
    return gens[0], (toks[0] if toks else 0)


def graded(text: str, gold: str) -> bool:
    pred = normalize_answer(extract_gsm8k_answer(text))
    return bool(pred) and bool(gold) and pred == gold


def bootstrap_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def paired_ci(a, b, n_boot=2000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, lo, hi = bootstrap_ci(d, n_boot, seed)
    return {"mean_diff": m, "ci_lo": lo, "ci_hi": hi,
            "credible_positive": lo > 0, "credible_negative": hi < 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="medqa", choices=["medqa", "gsm8k"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="test", choices=["test", "train", "dev"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--budgets", default="128,256,512,1024")
    ap.add_argument("--conditions", default="real,shuffled,zero,none")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/diag/cache_usage")
    args = ap.parse_args()
    args.budgets_list = [int(b) for b in args.budgets.split(",") if b.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    items = load_task(args.task, args.split)
    if args.n > 0:
        items = items[: args.n]
    n = len(items)
    print(f"[diag] task={args.task} n={n} split={args.split} k={args.k} budgets={args.budgets_list} "
          f"conditions={conditions} model={args.model_name}", flush=True)

    ns = make_ns(args)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    agents = [a for a in default_agents() if a.role != "judger"]
    device = wrapper.device

    # Rolling derangement for `shuffled`: item i uses item (i-1)'s cache; item 0 is
    # deferred to the end and paired with the last item's cache. Caches are stashed
    # on CPU to bound GPU memory.
    rows: List[Dict] = []  # one row per (item, condition, budget)
    prev_cpu = None
    first_inputs = None

    t0 = time.time()
    for i, item in enumerate(items):
        q, gold = item["question"], item["gold"]
        cache = build_upstream_cache(wrapper, q, args.k, ns, agents)
        ids, mask = judger_inputs(wrapper, q, ns)
        plen = _past_length(cache)

        for cond in conditions:
            if cond == "shuffled" and prev_cpu is None:
                continue  # item 0 handled after the loop
            for T in args.budgets_list:
                if cond == "real":
                    past = clone_past(cache)
                elif cond == "zero":
                    past = zero_past(cache)
                elif cond == "none":
                    past = None
                elif cond == "shuffled":
                    past = to_dev_past(prev_cpu, device)  # already a fresh copy
                else:
                    raise ValueError(f"unknown condition {cond}")
                text, ntok = decode_judger(wrapper, ids, mask, past, T)
                rows.append({"idx": item.get("idx", i), "cond": cond, "budget": T,
                             "correct": graded(text, gold), "tokens": ntok,
                             "past_len": plen})
                del past

        # stash for shuffle + item-0 fixup
        cache_cpu = to_cpu_past(cache)
        if i == 0:
            first_inputs = (ids.clone(), mask.clone(), gold, item.get("idx", 0), plen)
        prev_cpu = cache_cpu
        del cache
        if (i + 1) % 5 == 0 or i == 0:
            acc_real = np.mean([r["correct"] for r in rows if r["cond"] == "real" and r["budget"] == args.budgets_list[-1]] or [0])
            print(f"[diag] {i+1}/{n} real@{args.budgets_list[-1]}={acc_real:.3f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    # item-0 shuffled arm, paired with the last item's cache
    if "shuffled" in conditions and first_inputs is not None and prev_cpu is not None:
        ids0, mask0, gold0, idx0, plen0 = first_inputs
        for T in args.budgets_list:
            past = to_dev_past(prev_cpu, device)
            text, ntok = decode_judger(wrapper, ids0, mask0, past, T)
            rows.append({"idx": idx0, "cond": "shuffled", "budget": T,
                         "correct": graded(text, gold0), "tokens": ntok, "past_len": plen0})
            del past

    # aggregate (paired diffs aligned by item idx, since the shuffled arm reorders item 0)
    def arm_map(cond, T):
        return {r["idx"]: (1.0 if r["correct"] else 0.0)
                for r in rows if r["cond"] == cond and r["budget"] == T}

    def toks_for(cond, T):
        return [r["tokens"] for r in rows if r["cond"] == cond and r["budget"] == T]

    summary = {"config": vars(args), "n": n, "budgets": args.budgets_list,
               "conditions": conditions, "by_budget": {}}
    for T in args.budgets_list:
        entry = {}
        maps = {cond: arm_map(cond, T) for cond in conditions}
        for cond in conditions:
            a = list(maps[cond].values())
            m, lo, hi = bootstrap_ci(a, seed=args.seed)
            toks = toks_for(cond, T)
            entry[cond] = {"acc": m, "ci": [lo, hi], "n": len(a),
                           "mean_tokens": float(np.mean(toks)) if toks else 0.0}

        def _paired(c1, c2):
            common = sorted(set(maps.get(c1, {})) & set(maps.get(c2, {})))
            if not common:
                return None
            return paired_ci([maps[c1][i] for i in common],
                             [maps[c2][i] for i in common], seed=args.seed)

        if "real" in maps and "shuffled" in maps:
            entry["real_minus_shuffled"] = _paired("real", "shuffled")
        if "real" in maps and "none" in maps:
            entry["real_minus_none"] = _paired("real", "none")
        summary["by_budget"][str(T)] = entry

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(rows, f)

    # compact stdout
    for T in args.budgets_list:
        e = summary["by_budget"][str(T)]
        parts = [f"{c}={e[c]['acc']:.3f}" for c in conditions if c in e]
        rms = e.get("real_minus_shuffled") or {}
        rmn = e.get("real_minus_none") or {}
        print(f"DIAG budget={T} " + " ".join(parts)
              + f" | real-shuffled={rms.get('mean_diff',float('nan')):.3f}"
                f"[{rms.get('ci_lo',float('nan')):.3f},{rms.get('ci_hi',float('nan')):.3f}]"
              + f" real-none={rmn.get('mean_diff',float('nan')):.3f}"
                f"[{rmn.get('ci_lo',float('nan')):.3f},{rmn.get('ci_hi',float('nan')):.3f}]",
              flush=True)
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
