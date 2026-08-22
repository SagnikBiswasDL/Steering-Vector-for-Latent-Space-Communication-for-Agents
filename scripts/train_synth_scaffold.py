#!/usr/bin/env python3
"""Learn a short synthetic KV scaffold that replaces LatentMAS upstream agents.

Optimizes m learnable KV slots (per layer K,V), frozen host model, so the Judger
conditioned ONLY on these slots matches full-LatentMAS / real-cache Judger behavior.
Scaffold is question-INDEPENDENT (one fixed object).

Objectives:
  nll     — teacher-force NLL on full-LatentMAS greedy Judger text
  kl      — KL(real-cache Judger || scaffold Judger) on those tokens
  nll+kl  — both (recommended for the shrink gamble)

Inits:
  gauss — noise matched to pooled per-channel stats (baseline)
  pool  — segment-mean pool of averaged real caches into m slots
  trunc — last-m positions of averaged real cache

Example (bigger gamble):
  python scripts/train_synth_scaffold.py --model_name Qwen/Qwen3-14B \\
    --m 128 --init pool --objective nll+kl --n_train 80 --steps 800 \\
    --lr 3e-2 --eval_n 40 --budget 1024 --out_dir artifacts/ces/synth_m128
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace
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
from seal.ces import kl_tokenwise  # noqa: E402
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


def _to_legacy(cache):
    if Cache is not None and isinstance(cache, Cache):
        return cache.to_legacy_cache()
    return cache


def _from_legacy(legacy):
    if DynamicCache is not None:
        return DynamicCache.from_legacy_cache(tuple(legacy))
    return tuple(legacy)


def global_stats(wrapper, items, k, ns, agents):
    """Per-(layer,K/V,head,channel) mean/std over positions, pooled across real caches."""
    acc = None
    for it in items:
        cache = build_upstream_cache(wrapper, it["question"], k, ns, agents)
        legacy = _to_legacy(cache)
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


def average_real_legacy(wrapper, items, k, ns, agents):
    """Mean real cache (pad/truncate to median length) for pool/trunc init."""
    caches = []
    for it in items:
        cache = build_upstream_cache(wrapper, it["question"], k, ns, agents)
        caches.append(_to_legacy(cache))
        del cache
    lengths = [int(c[0][0].shape[-2]) for c in caches]
    L = int(np.median(lengths))
    n_layers = len(caches[0])
    avg = []
    for li in range(n_layers):
        ks, vs = [], []
        for c in caches:
            K, V = c[li]
            Kf, Vf = K.float().cpu(), V.float().cpu()
            # truncate or pad to L along positions
            def fix(t):
                s = t.shape[-2]
                if s >= L:
                    return t[..., :L, :]
                pad = t[..., -1:, :].expand(*t.shape[:-2], L - s, t.shape[-1])
                return torch.cat([t, pad], dim=-2)
            ks.append(fix(Kf))
            vs.append(fix(Vf))
        avg.append((torch.stack(ks).mean(0), torch.stack(vs).mean(0)))
    return avg, L


def _pool_positions(t, m):
    """Segment-mean pool [1,H,S,D] -> [1,H,m,D]."""
    S = t.shape[-2]
    if m >= S:
        if m == S:
            return t.clone()
        pad = t[..., -1:, :].expand(*t.shape[:-2], m - S, t.shape[-1])
        return torch.cat([t, pad], dim=-2)
    # equal-width bins
    edges = torch.linspace(0, S, m + 1).long()
    chunks = []
    for i in range(m):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = min(a + 1, S)
        chunks.append(t[..., a:b, :].mean(dim=-2, keepdim=True))
    return torch.cat(chunks, dim=-2)


class Scaffold(nn.Module):
    """m learnable KV slots per layer. Kept in fp32; cast to model dtype when used."""

    def __init__(self, stats, m, device, seed=0, init="gauss", avg_legacy=None):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.Ks = nn.ParameterList()
        self.Vs = nn.ParameterList()
        self.m = int(m)
        for li, ((meanK, stdK), (meanV, stdV)) in enumerate(
                [(p[0], p[1]) for p in stats]):
            H, D = meanK.shape[1], meanK.shape[3]
            if init in ("pool", "trunc") and avg_legacy is not None:
                aK, aV = avg_legacy[li]
                if init == "pool":
                    k0 = _pool_positions(aK, m)
                    v0 = _pool_positions(aV, m)
                else:  # trunc: last m
                    k0 = aK[..., -m:, :].contiguous() if aK.shape[-2] >= m else _pool_positions(aK, m)
                    v0 = aV[..., -m:, :].contiguous() if aV.shape[-2] >= m else _pool_positions(aV, m)
            else:
                k0 = torch.randn(1, H, m, D, generator=g) * stdK + meanK
                v0 = torch.randn(1, H, m, D, generator=g) * stdV + meanV
            self.Ks.append(nn.Parameter(k0.to(device).float()))
            self.Vs.append(nn.Parameter(v0.to(device).float()))

    def cache(self, dtype):
        legacy = tuple((K.to(dtype), V.to(dtype)) for K, V in zip(self.Ks, self.Vs))
        return _from_legacy(legacy)

    def save(self, path, meta):
        blob = {"K": [K.detach().cpu() for K in self.Ks],
                "V": [V.detach().cpu() for V in self.Vs], **meta}
        torch.save(blob, path)

    def nbytes_fp16(self):
        n = sum(p.numel() for p in self.parameters())
        return int(n * 2)


def build_fixed_synth(stats, L, dtype, device, seed):
    g = torch.Generator().manual_seed(int(seed))
    legacy = []
    for pair in stats:
        tens = []
        for (mean, std) in pair:
            H, D = mean.shape[1], mean.shape[3]
            noise = torch.randn(1, H, L, D, generator=g)
            tens.append((noise * std + mean).to(dtype).to(device))
        legacy.append((tens[0], tens[1]))
    return _from_legacy(legacy)


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


def judger_target_logits(wrapper, past, j_ids, target_ids):
    full_ids = torch.cat([j_ids, target_ids], dim=1)
    attn = torch.ones_like(full_ids, device=full_ids.device)
    past_len = _past_length(past) if past is not None else 0
    if past_len > 0:
        past_mask = torch.ones((attn.shape[0], past_len), dtype=attn.dtype, device=attn.device)
        attn = torch.cat([past_mask, attn], dim=-1)
    out = wrapper.model(
        input_ids=full_ids, attention_mask=attn, past_key_values=past,
        use_cache=False, return_dict=True,
    )
    j_len = j_ids.shape[1]
    return out.logits[:, j_len - 1 : j_len - 1 + target_ids.shape[1], :]


def distill_kl(wrapper, past_teacher, past_student, j_ids, target_ids):
    with torch.no_grad():
        logits_t = judger_target_logits(wrapper, past_teacher, j_ids, target_ids)
    logits_s = judger_target_logits(wrapper, past_student, j_ids, target_ids)
    return kl_tokenwise(logits_t, logits_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="medqa", choices=["medqa", "gsm8k"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=128, help="# learnable scaffold slots")
    ap.add_argument("--init", default="pool", choices=["gauss", "pool", "trunc"])
    ap.add_argument("--objective", default="nll+kl", choices=["nll", "kl", "nll+kl"])
    ap.add_argument("--alpha_kl", type=float, default=1.0)
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--stat_n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--min_lr", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--teacher_max_tok", type=int, default=512)
    ap.add_argument("--eval_n", type=int, default=40)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/ces/synth_scaffold")
    ap.add_argument("--skip_fixed_eval", action="store_true",
                    help="skip full-length fixed synth arm at eval (saves time)")
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

    need_real = args.objective in ("kl", "nll+kl") or args.init in ("pool", "trunc")
    train_items = load_task(args.task, "train")[: args.n_train]
    print(f"[synth] precomputing {len(train_items)} teacher outputs "
          f"(full LatentMAS; need_real={need_real})...", flush=True)
    teachers = []
    t0 = time.time()
    for i, it in enumerate(train_items):
        out = method.run_batch([it])[0]
        j_ids, _ = judger_inputs(wrapper, it["question"], ns)
        tgt = wrapper.tokenizer(out.get("raw_prediction", "") or " ",
                                add_special_tokens=False, return_tensors="pt")["input_ids"]
        tgt = tgt[:, : args.teacher_max_tok].to(wrapper.device)
        real_cpu = None
        if need_real and args.objective in ("kl", "nll+kl"):
            # separate real-cache build (method.run already built one internally; rebuild cheap vs storing GPU)
            real = build_upstream_cache(wrapper, it["question"], args.k, ns, up_agents)
            legacy = _to_legacy(real)
            real_cpu = tuple((K.detach().cpu(), V.detach().cpu()) for K, V in legacy)
            del real
        if tgt.shape[1] >= 1:
            teachers.append({"j_ids": j_ids, "tgt": tgt, "real_cpu": real_cpu})
        if (i + 1) % 10 == 0:
            print(f"[synth] teacher {i+1}/{len(train_items)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[synth] {len(teachers)} teacher targets ready", flush=True)

    # --- stats + optional avg cache for init ---
    stat_items = load_task(args.task, "train")[: args.stat_n]
    print(f"[synth] computing global stats on {len(stat_items)} items...", flush=True)
    stats = global_stats(wrapper, stat_items, args.k, ns, up_agents)
    avg_legacy, full_L = None, None
    if args.init in ("pool", "trunc"):
        print(f"[synth] averaging real caches for init={args.init}...", flush=True)
        avg_legacy, full_L = average_real_legacy(wrapper, stat_items, args.k, ns, up_agents)
    elif not args.skip_fixed_eval:
        # still need a length for fixed synth eval
        _, full_L = average_real_legacy(wrapper, stat_items[: min(4, len(stat_items))],
                                        args.k, ns, up_agents)

    scaffold = Scaffold(stats, args.m, wrapper.device, seed=args.seed,
                        init=args.init, avg_legacy=avg_legacy)
    opt = torch.optim.Adam(list(scaffold.parameters()), lr=args.lr)

    def lr_at(step):
        # cosine decay lr -> min_lr
        if args.steps <= 1:
            return args.lr
        t = step / (args.steps - 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    print(f"[synth] train m={args.m} init={args.init} obj={args.objective} "
          f"steps={args.steps} lr={args.lr}->{args.min_lr} bytes_fp16={scaffold.nbytes_fp16()}",
          flush=True)
    hist = []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        sample = teachers[step % len(teachers)]
        j_ids, tgt = sample["j_ids"], sample["tgt"]
        opt.zero_grad(set_to_none=True)
        cache = scaffold.cache(dtype)
        loss = torch.zeros((), device=wrapper.device)
        parts = {}
        if args.objective in ("nll", "nll+kl"):
            nll = wrapper.teacher_force_nll(cache, j_ids, tgt, role="judger", steer_judger=False)
            loss = loss + nll
            parts["nll"] = float(nll.detach())
            cache = scaffold.cache(dtype)  # rebuild; previous may be mutated/consumed
        if args.objective in ("kl", "nll+kl") and sample["real_cpu"] is not None:
            real_dev = _from_legacy(
                tuple((K.to(wrapper.device, dtype=dtype), V.to(wrapper.device, dtype=dtype))
                      for K, V in sample["real_cpu"]))
            kd = distill_kl(wrapper, real_dev, cache, j_ids, tgt)
            loss = loss + float(args.alpha_kl) * kd
            parts["kl"] = float(kd.detach())
            del real_dev
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(scaffold.parameters(), args.grad_clip))
        opt.step()
        row = {"step": step, "loss": float(loss.detach()), "grad_norm": gnorm,
               "lr": lr_at(step), **parts}
        hist.append(row)
        if (step + 1) % 25 == 0 or step == 0:
            print(json.dumps(row), flush=True)

    scaffold.save(os.path.join(args.out_dir, "scaffold.pt"),
                  {"m": args.m, "model_name": args.model_name, "task": args.task,
                   "init": args.init, "objective": args.objective})
    with open(os.path.join(args.out_dir, "train_hist.json"), "w") as f:
        json.dump(hist, f)

    # --- eval: learned vs none vs real vs fixed full synth ---
    def decode(past, ids, mask):
        gens, _ = wrapper.generate_text_batch(ids, mask, max_new_tokens=args.budget,
                                              temperature=0.0, top_p=1.0,
                                              past_key_values=past, role="judger")
        toks = list(getattr(wrapper, "last_gen_token_counts", [0]))
        return gens[0], (toks[0] if toks else 0)

    fixed = None
    if not args.skip_fixed_eval and full_L is not None:
        fixed = build_fixed_synth(stats, full_L, dtype, wrapper.device, seed=args.seed)
        print(f"[synth] fixed synth length={full_L}", flush=True)

    test_items = load_task(args.task, "test")[: args.eval_n]
    arms = ["scaffold", "none", "real"] + ([] if fixed is None else ["fixed"])
    print(f"[synth] eval on {len(test_items)} test items arms={arms}", flush=True)
    rows = {a: [] for a in arms}
    for i, it in enumerate(test_items):
        ids, mask = judger_inputs(wrapper, it["question"], ns)
        with torch.no_grad():
            txt_s, tok_s = decode(scaffold.cache(dtype), ids, mask)
            rows["scaffold"].append({"correct": graded(txt_s, it["gold"]), "tokens": tok_s})
            txt_n, tok_n = decode(None, ids, mask)
            rows["none"].append({"correct": graded(txt_n, it["gold"]), "tokens": tok_n})
            real = build_upstream_cache(wrapper, it["question"], args.k, ns, up_agents)
            txt_r, tok_r = decode(real, ids, mask)
            rows["real"].append({"correct": graded(txt_r, it["gold"]), "tokens": tok_r})
            del real
            if fixed is not None:
                txt_f, tok_f = decode(fixed, ids, mask)
                rows["fixed"].append({"correct": graded(txt_f, it["gold"]), "tokens": tok_f})
        if (i + 1) % 10 == 0:
            acc = np.mean([r["correct"] for r in rows["scaffold"]])
            print(f"[synth] eval {i+1}/{len(test_items)} scaffold_acc={acc:.3f}", flush=True)

    def summ(key):
        a = np.array([1.0 if r["correct"] else 0.0 for r in rows[key]])
        m, lo, hi = bootstrap_ci(a, seed=args.seed)
        return {"acc": m, "ci": [lo, hi],
                "mean_tokens": float(np.mean([r["tokens"] for r in rows[key]]))}

    report = {
        "config": vars(args), "m": args.m, "eval_n": len(test_items),
        "scaffold_fp16_bytes": scaffold.nbytes_fp16(),
        "full_L": full_L,
        "final_loss": hist[-1]["loss"] if hist else None,
        "loss_improved": (hist[-1]["loss"] < hist[0]["loss"]) if len(hist) > 1 else None,
    }
    for a in arms:
        report[a] = summ(a)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    sc = report["scaffold"]
    msg = (f"SYNTHTRAIN m={args.m} init={args.init} obj={args.objective} "
           f"scaffold_acc={sc['acc']:.3f} (tok={sc['mean_tokens']:.0f}) "
           f"real={report['real']['acc']:.3f} none={report['none']['acc']:.3f}")
    if "fixed" in report:
        msg += f" fixed={report['fixed']['acc']:.3f}"
    msg += f" fp16_bytes={scaffold.nbytes_fp16()} final_loss={report['final_loss']}"
    print(msg, flush=True)
    print("SYNTHTRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
