#!/usr/bin/env python3
"""Scaffold mechanism sweep: WHAT property of the upstream latent cache creates the
content-independent "conciseness scaffold" effect on the Judger?

For each item we build the real upstream cache (P->C->R at K), then decode the Judger
with the cache replaced by several variants, at a fixed budget, and measure acc + tokens:

  real        - the item's own cache (upper baseline)
  none        - no cache (lower baseline; judger re-solves, long output)
  shuffled    - another question's real cache (cross-question; rolling derangement)
  gauss       - N(0,1) noise, same shapes (unmatched random control)
  matched     - Gaussian matched to real per-(head,channel) mean/std over positions
  repeat_last - the last latent position's K/V repeated across all positions (single vector)
  trunc{m}    - real cache truncated to its last m positions
  pca{r}      - per-(layer,head) rank-r reconstruction of the real cache over positions
  crosstask   - a cache from the OTHER task (if --donor_task set)

Decision tree (see docs):
  matched works              -> effect is coarse activation statistics -> universal synthetic scaffold
  only model-gen (real/shuffled/pca) works -> stay on manifold -> compress model caches
  repeat_last works          -> a single latent vector induces the concise mode
  need several positions/rank -> learn a small latent KV bank / subspace

Example:
  python scripts/diag_scaffold_sweep.py --model_name Qwen/Qwen3-4B --task medqa \
    --split test --n 40 --budgets 1024 \
    --variants real,none,shuffled,gauss,matched,repeat_last,trunc16,trunc4,pca8 \
    --out_dir artifacts/diag/4b/scaffold_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = None
    DynamicCache = None


def load_task(task, split):
    return list(load_medqa(split=split) if task == "medqa" else load_gsm8k(split=split))


def make_ns(args):
    return SimpleNamespace(
        model_name=args.model_name, task=args.task, prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=max(args.budgets_list),
        text_mas_context_length=-1, temperature=0.0, top_p=1.0, seed=args.seed,
        seal=False, kvsteer=False, ces=False, capture_acts=None, planner_steps=None,
        critic_steps=None, refiner_steps=None, latent_steps=0, latent_space_realign=False,
    )


# ---- cache utilities (operate per-tensor; each returns FRESH tensors, safe for decode) ----
def _map_past(past, fn):
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


def clone_past(p):
    return _map_past(p, lambda t: t.clone())


def to_cpu_past(p):
    return _map_past(p, lambda t: t.detach().to("cpu"))


def to_dev_past(p, dev):
    return _map_past(p, lambda t: t.to(dev))


def _gauss(t):
    return torch.randn_like(t)


def _matched(t):
    # match per-(head,channel) mean/std over the positions axis (dim=-2)
    mu = t.mean(dim=-2, keepdim=True)
    sd = t.std(dim=-2, keepdim=True)
    return torch.randn_like(t) * sd + mu


def _repeat_last(t):
    return t[..., -1:, :].expand_as(t).contiguous()


def _trunc(m):
    return lambda t: t[..., -int(m):, :].contiguous()


def _pca(r):
    def fn(t):
        # t: [1, H, S, D]; low-rank reconstruct over positions per head
        x = t[0].float()                        # [H, S, D]
        mu = x.mean(dim=1, keepdim=True)         # [H, 1, D]
        xc = x - mu
        try:
            U, Sig, Vh = torch.linalg.svd(xc, full_matrices=False)  # U[H,S,k] Sig[H,k] Vh[H,k,D]
        except Exception:
            return t.clone()
        rr = min(int(r), Sig.shape[-1])
        recon = (U[:, :, :rr] * Sig[:, None, :rr]) @ Vh[:, :rr, :] + mu  # [H,S,D]
        return recon.unsqueeze(0).to(t.dtype)
    return fn


def precompute_global_stats(wrapper, items, k, ns, agents, synth_len):
    """Pool per-(layer, K/V, head, channel) mean/std over positions across real caches."""
    acc = None
    lengths = []
    for it in items:
        cache = build_upstream_cache(wrapper, it["question"], k, ns, agents)
        legacy = cache.to_legacy_cache() if (Cache is not None and isinstance(cache, Cache)) else cache
        lengths.append(int(legacy[0][0].shape[-2]))
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
            var = (acc[li][ti]["ss"] / cnt - mean * mean).clamp_min(0)
            pair.append((mean, var.sqrt()))
        stats.append(pair)
    L = int(synth_len) if synth_len and synth_len > 0 else int(np.median(lengths))
    return stats, L


def build_synth_cache(stats, L, dtype, device, seed):
    g = torch.Generator().manual_seed(int(seed))
    legacy = []
    for pair in stats:
        tens = []
        for (mean, std) in pair:
            H, D = mean.shape[1], mean.shape[3]
            noise = torch.randn(1, H, L, D, generator=g)
            tens.append((noise * std + mean).to(dtype).to(device))
        legacy.append((tens[0], tens[1]))
    return DynamicCache.from_legacy_cache(tuple(legacy))


def parse_synth_spec(name, default_len, default_seed):
    """Parse a synth variant name for length/rank/seed: l<N>, r<N>, s<N> tokens.
    e.g. synthglobal -> (median,full,seed); synth_l64_r16 -> (64,16,seed); synth_s2 -> seed 2."""
    L, R, S = default_len, None, default_seed
    mL = re.search(r"l(\d+)", name)
    mR = re.search(r"r(\d+)", name)
    mS = re.search(r"s(\d+)", name)
    if mL:
        L = int(mL.group(1))
    if mR:
        R = int(mR.group(1))
    if mS:
        S = int(mS.group(1))
    return L, R, S


def build_variant(name, real_cache, prev_cache, donor_cache, synth_dict, device):
    if name == "real":
        return clone_past(real_cache)
    if name == "none":
        return None
    if name == "shuffled":
        return to_dev_past(prev_cache, device) if prev_cache is not None else None
    if name == "crosstask":
        return to_dev_past(donor_cache, device) if donor_cache is not None else None
    if name in synth_dict:
        return to_dev_past(synth_dict[name], device)
    if name == "gauss":
        return _map_past(real_cache, _gauss)
    if name == "matched":
        return _map_past(real_cache, _matched)
    if name == "repeat_last":
        return _map_past(real_cache, _repeat_last)
    m = re.fullmatch(r"trunc(\d+)", name)
    if m:
        return _map_past(real_cache, _trunc(int(m.group(1))))
    m = re.fullmatch(r"pca(\d+)", name)
    if m:
        return _map_past(real_cache, _pca(int(m.group(1))))
    raise ValueError(f"unknown variant {name}")


def build_upstream_cache(wrapper, question, k, ns, agents):
    past = None
    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=question, context="", method="latent_mas", args=ns)
        _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
        past = wrapper.generate_latent_batch(ids, attention_mask=mask, latent_steps=int(k),
                                             past_key_values=past, role=agent.role)
    return past


def judger_inputs(wrapper, question, ns):
    messages = build_agent_message_sequential_latent_mas(
        role="judger", question=question, context="", method="latent_mas", args=ns)
    _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
    return ids, mask


def decode(wrapper, ids, mask, past, T):
    gens, _ = wrapper.generate_text_batch(ids, mask, max_new_tokens=int(T),
                                          temperature=0.0, top_p=1.0,
                                          past_key_values=past, role="judger")
    toks = list(getattr(wrapper, "last_gen_token_counts", [0]))
    return gens[0], (toks[0] if toks else 0)


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


def paired_ci(a, b, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, lo, hi = bootstrap_ci(d, seed=seed)
    return {"mean_diff": m, "ci_lo": lo, "ci_hi": hi,
            "credible_positive": lo > 0, "credible_negative": hi < 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B")
    ap.add_argument("--task", default="medqa", choices=["medqa", "gsm8k"])
    ap.add_argument("--donor_task", default=None, choices=[None, "medqa", "gsm8k"],
                    help="task for the crosstask variant donor cache")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="test", choices=["test", "train", "dev"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--budgets", default="1024")
    ap.add_argument("--variants",
                    default="real,none,shuffled,gauss,matched,repeat_last,trunc16,trunc4,pca8")
    ap.add_argument("--stat_n", type=int, default=8, help="# real caches for global synth stats")
    ap.add_argument("--stat_split", default="train", choices=["test", "train", "dev"])
    ap.add_argument("--synth_len", type=int, default=0, help="0 = median real length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/diag/scaffold_sweep")
    args = ap.parse_args()
    args.budgets_list = [int(b) for b in args.budgets.split(",") if b.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    items = load_task(args.task, args.split)
    if args.n > 0:
        items = items[: args.n]
    n = len(items)
    print(f"[scaffold] task={args.task} n={n} k={args.k} budgets={args.budgets_list} "
          f"variants={variants} model={args.model_name}", flush=True)

    ns = make_ns(args)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    up_agents = [a for a in default_agents() if a.role != "judger"]
    device = wrapper.device

    # optional cross-task donor cache (one fixed donor from the other task)
    donor_cache = None
    if "crosstask" in variants and args.donor_task:
        d_items = load_task(args.donor_task, "test")
        donor_cache = to_cpu_past(build_upstream_cache(wrapper, d_items[0]["question"], args.k, ns, up_agents))
        print(f"[scaffold] built crosstask donor from {args.donor_task}", flush=True)

    # GLOBAL synthetic scaffolds (question-independent; from real-cache stats). Any variant
    # named synth* is built here at its parsed length/rank/seed.
    synth_dict = {}
    synth_variants = [v for v in variants if v.startswith("synth")]
    if synth_variants:
        stat_items = load_task(args.task, args.stat_split)[: args.stat_n]
        stats, med_len = precompute_global_stats(wrapper, stat_items, args.k, ns, up_agents, args.synth_len)
        dtype = next(wrapper.model.parameters()).dtype
        default_len = int(args.synth_len) if args.synth_len and args.synth_len > 0 else med_len
        for v in synth_variants:
            L, R, S = parse_synth_spec(v, default_len, args.seed)
            sc = build_synth_cache(stats, L, dtype, device, S)
            if R:
                sc = _map_past(sc, _pca(R))
            synth_dict[v] = to_cpu_past(sc)
            print(f"[scaffold] built {v}: len={L} rank={R or 'full'} seed={S} "
                  f"(median real len={med_len}, from {len(stat_items)} {args.stat_split} caches)", flush=True)

    rows: List[Dict] = []
    prev_cpu = None
    first_inputs = None
    t0 = time.time()
    for i, item in enumerate(items):
        q, gold = item["question"], item["gold"]
        real_cache = build_upstream_cache(wrapper, q, args.k, ns, up_agents)
        ids, mask = judger_inputs(wrapper, q, ns)
        for T in args.budgets_list:
            for v in variants:
                if v == "shuffled" and prev_cpu is None:
                    continue  # item 0 fixed up after the loop
                past = build_variant(v, real_cache, prev_cpu, donor_cache, synth_dict, device)
                text, ntok = decode(wrapper, ids, mask, past, T)
                rows.append({"idx": item.get("idx", i), "variant": v, "budget": T,
                             "correct": graded(text, gold), "tokens": ntok})
                del past
        cache_cpu = to_cpu_past(real_cache)
        if i == 0:
            first_inputs = (ids.clone(), mask.clone(), gold, item.get("idx", 0))
        prev_cpu = cache_cpu
        del real_cache
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[scaffold] {i+1}/{n} elapsed={time.time()-t0:.0f}s", flush=True)

    if "shuffled" in variants and first_inputs is not None and prev_cpu is not None:
        ids0, mask0, gold0, idx0 = first_inputs
        for T in args.budgets_list:
            past = to_dev_past(prev_cpu, device)
            text, ntok = decode(wrapper, ids0, mask0, past, T)
            rows.append({"idx": idx0, "variant": "shuffled", "budget": T,
                         "correct": graded(text, gold0), "tokens": ntok})
            del past

    # aggregate (align by idx for paired diffs vs real / none)
    def vmap(v, T):
        return {r["idx"]: (1.0 if r["correct"] else 0.0)
                for r in rows if r["variant"] == v and r["budget"] == T}

    summary = {"config": vars(args), "n": n, "by_budget": {}}
    for T in args.budgets_list:
        maps = {v: vmap(v, T) for v in variants}
        entry = {}
        for v in variants:
            a = list(maps[v].values())
            m, lo, hi = bootstrap_ci(a, seed=args.seed)
            toks = [r["tokens"] for r in rows if r["variant"] == v and r["budget"] == T]
            entry[v] = {"acc": m, "ci": [lo, hi], "n": len(a),
                        "mean_tokens": float(np.mean(toks)) if toks else 0.0}
        for v in variants:
            if v in ("real", "none"):
                continue
            for base in ("real", "none"):
                common = sorted(set(maps.get(v, {})) & set(maps.get(base, {})))
                if common:
                    entry[v][f"minus_{base}"] = paired_ci(
                        [maps[v][i] for i in common], [maps[base][i] for i in common], seed=args.seed)
        summary["by_budget"][str(T)] = entry

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(rows, f)

    for T in args.budgets_list:
        e = summary["by_budget"][str(T)]
        real_acc = e.get("real", {}).get("acc", float("nan"))
        none_acc = e.get("none", {}).get("acc", float("nan"))
        print(f"SCAFFOLD budget={T} real={real_acc:.3f} none={none_acc:.3f}", flush=True)
        for v in variants:
            if v in ("real", "none"):
                continue
            ev = e[v]
            mr = ev.get("minus_real", {})
            print(f"SCAFFOLD   {v:<12} acc={ev['acc']:.3f} tok={ev['mean_tokens']:.0f} "
                  f"minus_real={mr.get('mean_diff', float('nan')):+.3f}"
                  f"[{mr.get('ci_lo', float('nan')):+.3f},{mr.get('ci_hi', float('nan')):+.3f}]",
                  flush=True)
    print("SCAFFOLD_DONE", flush=True)


if __name__ == "__main__":
    main()
