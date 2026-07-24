#!/usr/bin/env python3
"""Learn a short synthetic KV scaffold that replaces the LatentMAS upstream agents.

We optimize m learnable KV slots (per layer K,V), frozen host model, so that the
Judger conditioned ONLY on these slots reproduces the FULL-LatentMAS Judger's behavior
(behavior distillation: teacher-force the full pipeline's greedy output under the
scaffold-conditioned Judger). The scaffold is question-INDEPENDENT (one fixed object).

Reuses the proven differentiable path: wrapper.teacher_force_nll(past=scaffold, ...)
backprops through attention over the scaffold KV to the slot parameters.

Pipeline: precompute teacher outputs on TRAIN -> train slots -> eval on TEST vs none.

Example:
  python scripts/train_synth_scaffold.py --model_name Qwen/Qwen3-14B \
    --m 64 --n_train 40 --steps 400 --lr 5e-2 --eval_n 40 --budget 1024 \
    --out_dir artifacts/ces/synth_scaffold_m64
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
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_medqa, load_gsm8k  # noqa: E402
from methods import default_agents  # noqa: E402
from methods.latent_mas import LatentMASMethod  # noqa: E402
from models import ModelWrapper, _past_length  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from utils import set_seed, auto_device, extract_gsm8k_answer, normalize_answer  # noqa: E402

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = None
    DynamicCache = None


def load_task(task, split):
    return list(load_medqa(split=split) if task == "medqa" else load_gsm8k(split=split))


def make_ns(args, budget):
    return SimpleNamespace(
        model_name=args.model_name, task=args.task, prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=budget,
        text_mas_context_length=-1, temperature=0.0, top_p=1.0, seed=args.seed,
        seal=False, kvsteer=False, ces=False, capture_acts=None, planner_steps=None,
        critic_steps=None, refiner_steps=None, latent_steps=args.k, latent_space_realign=False,
    )


def judger_inputs(wrapper, question, ns):
    messages = build_agent_message_sequential_latent_mas(
        role="judger", question=question, context="", method="latent_mas", args=ns)
    _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
    return ids, mask


def build_upstream_cache(wrapper, question, k, ns, agents):
    past = None
    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=question, context="", method="latent_mas", args=ns)
        _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
        past = wrapper.generate_latent_batch(ids, attention_mask=mask, latent_steps=int(k),
                                             past_key_values=past, role=agent.role)
    return past


def global_stats(wrapper, items, k, ns, agents):
    """Per-(layer,K/V,head,channel) mean/std over positions, pooled across real caches."""
    acc = None
    for it in items:
        cache = build_upstream_cache(wrapper, it["question"], k, ns, agents)
        legacy = cache.to_legacy_cache() if (Cache is not None and isinstance(cache, Cache)) else cache
        if acc is None:
            acc = [[{"s": torch.zeros(1, t.shape[1], 1, t.shape[3]),
                     "ss": torch.zeros(1, t.shape[1], 1, t.shape[3]), "cnt": 0}
                    for t in (K, V)] for (K, V) in legacy]
        for li, (K, V) in enumerate(legacy):
            for ti, t in enumerate((K, V)):
                tf = t.float().cpu()
                acc[li][ti]["s"] += tf.sum(dim=2, keepdim=True)
                acc[li][ti]["ss"] += (tf * tf).sum(dim=2, keepdim=True)
                acc[li][ti]["cnt"] += int(tf.shape[2])
        del cache
    stats = []
    for li in range(len(acc)):
        pair = []
        for ti in range(2):
            cnt = max(acc[li][ti]["cnt"], 1)
            mean = acc[li][ti]["s"] / cnt
            var = (acc[li][ti]["ss"] / cnt - (acc[li][ti]["s"] / cnt) ** 2).clamp_min(1e-8)
            pair.append((mean, var.sqrt()))
        stats.append(pair)
    return stats


class Scaffold(nn.Module):
    """m learnable KV slots per layer. Kept in fp32; cast to model dtype when used."""

    def __init__(self, stats, m, device, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.Ks = nn.ParameterList()
        self.Vs = nn.ParameterList()
        for (meanK, stdK), (meanV, stdV) in [(p[0], p[1]) for p in stats]:
            H, D = meanK.shape[1], meanK.shape[3]
            k0 = torch.randn(1, H, m, D, generator=g) * stdK + meanK
            v0 = torch.randn(1, H, m, D, generator=g) * stdV + meanV
            self.Ks.append(nn.Parameter(k0.to(device).float()))
            self.Vs.append(nn.Parameter(v0.to(device).float()))

    def cache(self, dtype):
        legacy = tuple((K.to(dtype), V.to(dtype)) for K, V in zip(self.Ks, self.Vs))
        return DynamicCache.from_legacy_cache(legacy)

    def save(self, path, meta):
        blob = {"K": [K.detach().cpu() for K in self.Ks],
                "V": [V.detach().cpu() for V in self.Vs], **meta}
        torch.save(blob, path)


def graded(text, gold):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="medqa", choices=["medqa", "gsm8k"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=64, help="# learnable scaffold slots")
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--stat_n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--teacher_max_tok", type=int, default=512, help="cap teacher target length")
    ap.add_argument("--eval_n", type=int, default=40)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/ces/synth_scaffold")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ns = make_ns(args, args.budget)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    for p in wrapper.model.parameters():
        p.requires_grad_(False)
    dtype = next(wrapper.model.parameters()).dtype
    up_agents = [a for a in default_agents() if a.role != "judger"]
    method = LatentMASMethod(wrapper, latent_steps=args.k, judger_max_new_tokens=args.budget,
                             temperature=0.0, top_p=1.0, generate_bs=1, args=ns)

    # --- teacher targets: full-LatentMAS greedy output on TRAIN questions ---
    train_items = load_task(args.task, "train")[: args.n_train]
    print(f"[synth] precomputing {len(train_items)} teacher outputs (full LatentMAS)...", flush=True)
    teachers = []
    t0 = time.time()
    for i, it in enumerate(train_items):
        out = method.run_batch([it])[0]
        j_ids, _ = judger_inputs(wrapper, it["question"], ns)
        tgt = wrapper.tokenizer(out.get("raw_prediction", "") or " ",
                                add_special_tokens=False, return_tensors="pt")["input_ids"]
        tgt = tgt[:, : args.teacher_max_tok].to(wrapper.device)
        if tgt.shape[1] >= 1:
            teachers.append((j_ids, tgt))
        if (i + 1) % 10 == 0:
            print(f"[synth] teacher {i+1}/{len(train_items)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[synth] {len(teachers)} teacher targets ready", flush=True)

    # --- init scaffold from global stats ---
    stats = global_stats(wrapper, load_task(args.task, "train")[: args.stat_n], args.k, ns, up_agents)
    scaffold = Scaffold(stats, args.m, wrapper.device, seed=args.seed)
    opt = torch.optim.Adam(list(scaffold.parameters()), lr=args.lr)

    # --- train: distill teacher outputs under the scaffold-conditioned Judger ---
    print(f"[synth] training m={args.m} slots, steps={args.steps}, lr={args.lr}", flush=True)
    hist = []
    for step in range(args.steps):
        j_ids, tgt = teachers[step % len(teachers)]
        opt.zero_grad(set_to_none=True)
        cache = scaffold.cache(dtype)
        loss = wrapper.teacher_force_nll(cache, j_ids, tgt, role="judger", steer_judger=False)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(scaffold.parameters(), args.grad_clip))
        opt.step()
        hist.append({"step": step, "loss": float(loss.detach()), "grad_norm": gnorm})
        if (step + 1) % 25 == 0 or step == 0:
            print(json.dumps(hist[-1]), flush=True)
    scaffold.save(os.path.join(args.out_dir, "scaffold.pt"),
                  {"m": args.m, "model_name": args.model_name, "task": args.task})

    # --- eval on TEST: learned scaffold vs none (+ report) ---
    def decode(past, ids, mask):
        gens, _ = wrapper.generate_text_batch(ids, mask, max_new_tokens=args.budget,
                                              temperature=0.0, top_p=1.0,
                                              past_key_values=past, role="judger")
        toks = list(getattr(wrapper, "last_gen_token_counts", [0]))
        return gens[0], (toks[0] if toks else 0)

    test_items = load_task(args.task, "test")[: args.eval_n]
    print(f"[synth] eval on {len(test_items)} test items (learned scaffold vs none)", flush=True)
    rows = {"scaffold": [], "none": []}
    for i, it in enumerate(test_items):
        ids, mask = judger_inputs(wrapper, it["question"], ns)
        with torch.no_grad():
            txt_s, tok_s = decode(scaffold.cache(dtype), ids, mask)
            txt_n, tok_n = decode(None, ids, mask)
        rows["scaffold"].append({"correct": graded(txt_s, it["gold"]), "tokens": tok_s})
        rows["none"].append({"correct": graded(txt_n, it["gold"]), "tokens": tok_n})
        if (i + 1) % 10 == 0:
            acc = np.mean([r["correct"] for r in rows["scaffold"]])
            print(f"[synth] eval {i+1}/{len(test_items)} scaffold_acc={acc:.3f}", flush=True)

    def summ(key):
        a = np.array([1.0 if r["correct"] else 0.0 for r in rows[key]])
        m, lo, hi = bootstrap_ci(a, seed=args.seed)
        return {"acc": m, "ci": [lo, hi], "mean_tokens": float(np.mean([r["tokens"] for r in rows[key]]))}

    report = {"config": vars(args), "m": args.m, "eval_n": len(test_items),
              "scaffold": summ("scaffold"), "none": summ("none"),
              "final_loss": hist[-1]["loss"] if hist else None,
              "loss_improved": (hist[-1]["loss"] < hist[0]["loss"]) if len(hist) > 1 else None}
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"SYNTHTRAIN m={args.m} scaffold_acc={report['scaffold']['acc']:.3f} "
          f"(tok={report['scaffold']['mean_tokens']:.0f}) none_acc={report['none']['acc']:.3f} "
          f"final_loss={report['final_loss']} improved={report['loss_improved']}", flush=True)
    print("SYNTHTRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
