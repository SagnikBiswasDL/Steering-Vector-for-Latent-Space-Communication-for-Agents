#!/usr/bin/env python3
"""2x2 experiment: {Full relay, H-OBF compressed relay} x {ASC off, ASC on}.

System claim under test
-----------------------
    Can a compressed inter-agent relay cut KV memory while Judger-only concision
    steering (ASC/SEAL) removes any downstream verbosity tax, at iso-accuracy?

Arms (paired, same items, greedy decode):
    A  full relay  + no ASC      (baseline)
    B  full relay  + ASC         (concision on an uncompressed relay)
    C  H-OBF relay + no ASC      (compression alone -- may inflate Judger tokens)
    D  H-OBF relay + ASC         (compression + concision -- the target config)
  (optional diagnostic, --with_evict:)
    E  plain-H relay + no ASC     (eviction only, no low-rank backfill)
    F  plain-H relay + ASC

Primary analysis (per item, then paired-bootstrap CIs) on
{final tokens, judger decode time, end-to-end latency, accuracy}:
    ASC effect under Full   = B - A
    ASC effect under H-OBF  = D - C
    Interaction             = (D - C) - (B - A)
    Compression tax (noASC) = C - A
The headline story is: C increases Judger tokens vs A, ASC removes it (D ~ B ~ A
tokens) while accuracy stays ~ Full and end-to-end latency + relay MB drop.

Upstream (Planner/Critic/Refiner) is ALWAYS unsteered; ASC applies at the Judger
decoding only (seal_agents=judger).

ASC vector: pass --asc_vector <path>. On the pod this is the SEAL residual vector
(exec - reflection), which functionally reduces Judger tokens ~17-39% at retained
accuracy; a dedicated verbose->concise ASC vector for Qwen3 is a drop-in swap.
If --asc_vector is omitted, only the no-ASC arms (A, C[, E]) run.

Modes:
  --smoke        : gate mode (default n=12) -- prints PASS/FAIL wiring checks and
                   the decision-tree branch, does NOT trust the numbers.
  (no flag)      : paired run (default n=100).

Example (pod):
  python scripts/exp_2x2_relay_asc.py --model_name Qwen/Qwen3-14B --task gsm8k \
    --k 40 --n 12 --smoke --judger_budget 512 \
    --relay_budget 32 --sink 4 --rank 8 \
    --asc_vector artifacts/seal_vectors/qwen3-14b/gsm8k.pt --asc_coef 40 \
    --out_dir artifacts/exp2x2/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import (  # noqa: E402
    load_medqa, load_gsm8k, load_math, load_aime2024, load_aime_pooled,
)
from methods import default_agents  # noqa: E402
from models import ModelWrapper  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.relay_compress import RelayCompressor, to_legacy, from_legacy, kv_mb, num_positions  # noqa: E402
from utils import (set_seed, auto_device, extract_gsm8k_answer, normalize_answer,  # noqa: E402
                   extract_boxed_answer, normalize_math_answer)


# --------------------------------------------------------------------------- #
# pipeline helpers (mirrors scripts/diag_scaffold_sweep.py -- validated)
# --------------------------------------------------------------------------- #
HARD_TASKS = ("math", "aime2024", "aime_pooled")


def load_task(task, split):
    if task == "medqa":
        return list(load_medqa(split=split))
    if task == "math":
        return list(load_math(split=split))
    if task == "aime2024":
        # AIME HF splits are 'train' (the contest set); ignore caller split.
        return list(load_aime2024(split="train"))
    if task == "aime_pooled":
        return list(load_aime_pooled())
    return list(load_gsm8k(split=split))


def make_ns(args, run_asc):
    return SimpleNamespace(
        model_name=args.model_name, task=args.task, prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=args.judger_budget,
        text_mas_context_length=-1, temperature=0.0, top_p=1.0, seed=args.seed,
        # ASC = SEAL residual at the Judger only
        seal=bool(run_asc), seal_vector=args.asc_vector, seal_coef=args.asc_coef,
        seal_layer=args.asc_layer, seal_apply_to=args.asc_apply_to, seal_agents="judger",
        kvsteer=False, ces=False, capture_acts=None,
        planner_steps=None, critic_steps=None, refiner_steps=None,
        latent_steps=0, latent_space_realign=False,
    )


def deep_clone(past):
    """Fresh copy so an in-place-growing decode never corrupts a shared cache."""
    if past is None:
        return None
    return from_legacy([(k.clone(), v.clone()) for (k, v) in to_legacy(past)], like=past)


def to_dev(past, device):
    if past is None:
        return None
    return from_legacy([(k.to(device), v.to(device)) for (k, v) in to_legacy(past)], like=past)


def _pad_caches_left(caches, device):
    """Stack a list of per-item caches into one batched cache, left-padding the
    position axis to the max length. Returns (batched_cache, past_mask[B,Pmax], Pmax).
    Uniform-length caches (compressed arms) incur no padding -> past_mask all ones.
    Padded positions are masked out, so HF derives correct (left-padding-safe)
    position_ids from the combined attention mask.
    """
    legacies = [to_legacy(c) for c in caches]
    B = len(legacies)
    n_layers = len(legacies[0])
    lens = [lg[0][0].shape[-2] for lg in legacies]
    Pmax = max(lens)
    past_mask = torch.zeros(B, Pmax, dtype=torch.long, device=device)
    for b, ln in enumerate(lens):
        past_mask[b, Pmax - ln:] = 1
    layers = []
    for li in range(n_layers):
        Ks, Vs = [], []
        for b in range(B):
            k, v = legacies[b][li]
            k = k.to(device)
            v = v.to(device)
            ln = k.shape[-2]
            if ln < Pmax:
                pk = torch.zeros(1, k.shape[1], Pmax - ln, k.shape[3], dtype=k.dtype, device=device)
                pv = torch.zeros(1, v.shape[1], Pmax - ln, v.shape[3], dtype=v.dtype, device=device)
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            Ks.append(k)
            Vs.append(v)
        layers.append((torch.cat(Ks, 0), torch.cat(Vs, 0)))
    return from_legacy(layers), past_mask, Pmax


def decode_batch(wrapper, judger_ids, judger_mask, caches, asc_on, budget):
    """Batched Judger decode over a list of per-item caches. ASC (SEAL residual)
    is toggled judger-only via the persistent hook. Returns (texts, ntoks, eoss).
    caches may be all-None (Judger-only / arm N)."""
    device = wrapper.device
    jids = judger_ids.to(device)
    jmask = judger_mask.to(device)
    if all(c is None for c in caches):
        past, full_mask, cache_position = None, jmask, None
    else:
        if any(c is None for c in caches):
            raise ValueError("decode_batch: mixed None/non-None caches not supported")
        past, past_mask, Pmax = _pad_caches_left(caches, device)
        full_mask = torch.cat([past_mask, jmask], dim=1)
        cache_position = torch.arange(Pmax, Pmax + jids.shape[1], device=device)
    seal = getattr(wrapper, "seal", None)
    use = False
    if asc_on and seal is not None:
        seal.set_active_role("judger")
        if seal.has_effect_for("judger"):
            seal.enable()
            use = True
    try:
        gen_kwargs = dict(
            input_ids=jids, attention_mask=full_mask, past_key_values=past,
            max_new_tokens=int(budget), do_sample=False,
            pad_token_id=wrapper.tokenizer.pad_token_id,
            return_dict_in_generate=True, output_scores=False,
        )
        if cache_position is not None:
            gen_kwargs["cache_position"] = cache_position
        out = wrapper.model.generate(**gen_kwargs)
    finally:
        if use:
            seal.disable()
    seqs = out.sequences
    gen_start = jids.shape[1]
    eos_id = wrapper.tokenizer.eos_token_id
    pad_id = wrapper.tokenizer.pad_token_id
    texts, ntoks, eoss = [], [], []
    for i in range(seqs.shape[0]):
        gen = seqs[i, gen_start:]
        texts.append(wrapper.tokenizer.decode(gen, skip_special_tokens=True).strip())
        cnt, eos = 0, False
        for tok in gen.tolist():
            if eos_id is not None and tok == eos_id:
                cnt += 1
                eos = True
                break
            if pad_id is not None and pad_id != eos_id and tok == pad_id:
                break
            cnt += 1
        ntoks.append(cnt)
        eoss.append(eos)
    return texts, ntoks, eoss


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


def set_asc(wrapper, on: bool):
    """Toggle Judger-only ASC on the shared wrapper (no-op if no seal vector)."""
    if getattr(wrapper, "seal", None) is None:
        return False
    wrapper.seal_active_roles = {"judger"} if on else set()
    return on


def graded(text, gold, task="gsm8k"):
    if task == "math":
        pred = normalize_math_answer(extract_boxed_answer(text) or text)
    elif task in ("aime2024", "aime_pooled"):
        # AIME answers are integers; prefer \boxed{}, fall back to gsm8k extractor.
        pred = normalize_answer(extract_boxed_answer(text) or extract_gsm8k_answer(text))
    else:
        pred = normalize_answer(extract_gsm8k_answer(text))
    return bool(pred) and bool(gold) and pred == gold


def peak_mem_mb(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return 0.0


def reset_peak(device):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def bootstrap_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def paired_diff(a_map, b_map, seed=0):
    """b - a over shared idx; returns mean diff + 95% CI."""
    common = sorted(set(a_map) & set(b_map))
    d = np.asarray([b_map[i] - a_map[i] for i in common], float)
    m, lo, hi = bootstrap_ci(d, seed=seed)
    return {"mean_diff": m, "ci_lo": lo, "ci_hi": hi, "n": len(common),
            "credible_positive": lo > 0, "credible_negative": hi < 0}


def flips(a_map, b_map):
    common = sorted(set(a_map) & set(b_map))
    gained = sum(1 for i in common if b_map[i] > a_map[i])
    lost = sum(1 for i in common if b_map[i] < a_map[i])
    return {"n": len(common), "gained": gained, "lost": lost, "net": gained - lost}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="gsm8k",
                    choices=["medqa", "gsm8k", "math", "aime2024", "aime_pooled"])
    ap.add_argument("--split", default="test", choices=["test", "train", "dev"])
    ap.add_argument("--k", type=int, default=40, help="latent steps per upstream agent")
    ap.add_argument("--n", type=int, default=0, help="0 -> 12 (smoke) or 100 (paired)")
    ap.add_argument("--judger_budget", type=int, default=512, help="Judger max_new_tokens")
    # H-OBF compression
    ap.add_argument("--relay_budget", type=int, default=32, help="kept prompt KV positions/layer")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8, help="OBF low-rank backfill positions")
    ap.add_argument("--importance", default="key_norm",
                    choices=["key_norm", "value_norm", "recency"])
    ap.add_argument("--with_evict", action="store_true",
                    help="also run plain-H (eviction only) arms E/F")
    ap.add_argument("--with_shuffled", action="store_true",
                    help="also run a shuffled-cache arm S (wrong-question real cache, no ASC) "
                         "-- the content-independence control (real vs shuffled)")
    ap.add_argument("--with_crosstask", action="store_true",
                    help="also run a cross-task arm X (a DIFFERENT task's cache, no ASC) "
                         "-- the content-specificity ladder (instance vs task-level content)")
    ap.add_argument("--with_none", action="store_true",
                    help="also run arm N = Judger-only (no relay cache, no ASC) — hard-task "
                         "budget-artifact control")
    ap.add_argument("--donor_task", default=None,
                    choices=["medqa", "gsm8k", "math", "aime2024", "aime_pooled"],
                    help="task to draw cross-task donor caches from (for arm X)")
    # ASC (SEAL residual at Judger)
    ap.add_argument("--asc_vector", default=None, help="path to SEAL/ASC vector (.pt)")
    ap.add_argument("--asc_coef", type=float, default=40.0)
    ap.add_argument("--asc_layer", type=int, default=-1, help="-1 -> use vector's stored layer")
    ap.add_argument("--asc_apply_to", default="last", choices=["last", "all"])
    ap.add_argument("--smoke", action="store_true", help="gate mode + PASS/FAIL checks")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="items decoded per batch (GPU utilization). Upstream built per-item.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/exp2x2/run")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required (this runs Qwen3-14B on the pod).", file=sys.stderr)
        sys.exit(2)
    if args.n <= 0:
        args.n = 12 if args.smoke else 100

    run_asc = bool(args.asc_vector)
    if not run_asc:
        print("[2x2] no --asc_vector -> running no-ASC arms only (A, C[, E]).", flush=True)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    items = load_task(args.task, args.split)[: args.n]
    n = len(items)
    ns = make_ns(args, run_asc)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    device = wrapper.device
    up_agents = [a for a in default_agents() if a.role != "judger"]
    set_asc(wrapper, False)  # start with ASC off

    compressors = {"obf": RelayCompressor(mode="obf", budget=args.relay_budget, sink=args.sink,
                                          rank=args.rank, importance=args.importance)}
    if args.with_evict:
        compressors["evict"] = RelayCompressor(mode="evict", budget=args.relay_budget,
                                               sink=args.sink, importance=args.importance)

    # arm table: (name, relay-kind, asc-on)
    arms = [("A", "full", False)]
    if run_asc:
        arms.append(("B", "full", True))
    arms.append(("C", "obf", False))
    if run_asc:
        arms.append(("D", "obf", True))
    if args.with_evict:
        arms.append(("E", "evict", False))
        if run_asc:
            arms.append(("F", "evict", True))
    if args.with_shuffled:
        arms.append(("S", "shuffled", False))
    if args.with_crosstask:
        arms.append(("X", "crosstask", False))
    if args.with_none:
        arms.append(("N", "none", False))

    print(f"[2x2] task={args.task} n={n} k={args.k} judger_budget={args.judger_budget} "
          f"arms={[a[0] for a in arms]} compress=({','.join(c.summary() for c in compressors.values())}) "
          f"asc={'on' if run_asc else 'off'} coef={args.asc_coef} model={args.model_name}", flush=True)

    # cross-task donor cache pool (built once from a DIFFERENT task; kept on CPU)
    donor_pool = None
    if args.with_crosstask:
        if args.donor_task:
            dt = args.donor_task
        elif args.task in ("math", "aime2024", "aime_pooled"):
            dt = "medqa"  # domain jump (med → hard-math) is the paper's decisive cell
        elif args.task == "gsm8k":
            dt = "medqa"
        else:
            dt = "gsm8k"
        donor_ns = make_ns(args, run_asc)
        donor_ns.task = dt
        d_items = load_task(dt, "test")[: max(args.batch_size, 8)]
        donor_pool = []
        for it in d_items:
            dc = build_upstream_cache(wrapper, it["question"], args.k, donor_ns, up_agents)
            donor_pool.append(to_dev(dc, "cpu"))
        print(f"[2x2] built {len(donor_pool)} cross-task donor caches from task={dt}", flush=True)

    # left-pad batched prompts so the batched Judger decode is left-padding-safe
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass

    rows: List[Dict] = []
    t0 = time.time()
    B = max(1, int(args.batch_size))
    n_batches = (n + B - 1) // B
    print(f"[2x2] batch_size={B} -> {n_batches} batches", flush=True)
    for bi in range(n_batches):
        batch = items[bi * B:(bi + 1) * B]
        bsz = len(batch)
        bidx = [it.get("idx", bi * B + j) for j, it in enumerate(batch)]
        qs = [it["question"] for it in batch]
        golds = [it["gold"] for it in batch]

        # --- upstream (unsteered) + compression, per item (clean, unpadded) ---
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
            stats_by_kind["full"].append({"positions_out": num_positions(rc),
                                          "mb_out": kv_mb(rc), "ratio": 1.0,
                                          "sink_retained": True})
            for kind, comp in compressors.items():
                tc = time.perf_counter()
                cc, st = comp.compress(rc)
                comp_times[kind].append(time.perf_counter() - tc)
                cc_by_kind[kind].append(cc)
                stats_by_kind[kind].append(st.as_dict())
        assert getattr(wrapper, "seal", None) is None or not wrapper.seal._enabled, \
            "upstream must be unsteered"

        # batched judger prompts (left-padded)
        jmsgs = [build_agent_message_sequential_latent_mas(
            role="judger", question=q, context="", method="latent_mas", args=ns) for q in qs]
        _, jids_b, jmask_b, _ = wrapper.prepare_chat_batch(jmsgs, add_generation_prompt=True)

        cache_lists = {"full": real_caches}
        cache_lists.update(cc_by_kind)
        # shuffled = wrong-question real cache (within-batch derangement by 1)
        if args.with_shuffled:
            if bsz > 1:
                cache_lists["shuffled"] = real_caches[-1:] + real_caches[:-1]
                stats_by_kind["shuffled"] = stats_by_kind["full"][-1:] + stats_by_kind["full"][:-1]
            else:
                cache_lists["shuffled"] = real_caches
                stats_by_kind["shuffled"] = stats_by_kind["full"]
        # cross-task = a different task's cache (cycled from the donor pool)
        if args.with_crosstask and donor_pool:
            xs = [to_dev(donor_pool[j % len(donor_pool)], device) for j in range(bsz)]
            cache_lists["crosstask"] = xs
            stats_by_kind["crosstask"] = [
                {"positions_out": num_positions(c), "mb_out": kv_mb(c),
                 "ratio": 1.0, "sink_retained": True} for c in xs]
        if args.with_none:
            cache_lists["none"] = [None] * bsz
            stats_by_kind["none"] = [
                {"positions_out": 0, "mb_out": 0.0, "ratio": 0.0, "sink_retained": True}
                for _ in range(bsz)]

        # --- run each arm as a single batched decode ---
        for name, kind, asc_on in arms:
            reset_peak(device)
            td = time.perf_counter()
            texts, ntoks, eoss = decode_batch(
                wrapper, jids_b, jmask_b, cache_lists[kind], asc_on, args.judger_budget)
            dec_time = time.perf_counter() - td
            pmem = peak_mem_mb(device)
            per_item_dec = dec_time / max(1, bsz)
            for j in range(bsz):
                st = stats_by_kind[kind][j]
                ntok, text = int(ntoks[j]), texts[j]
                comp_s = comp_times[kind][j] if kind in comp_times else 0.0
                rows.append({
                    "idx": bidx[j], "arm": name, "relay": kind, "asc": bool(asc_on),
                    "correct": bool(graded(text, golds[j], args.task)), "tokens": ntok,
                    "eos": bool(eoss[j]),
                    "decode_s": per_item_dec,
                    "tok_per_s": (ntok / per_item_dec if per_item_dec > 0 else 0.0),
                    "e2e_s": up_times[j] + comp_s + per_item_dec,
                    "compress_s": comp_s, "upstream_s": up_times[j],
                    "relay_positions": int(st["positions_out"]),
                    "relay_mb": float(st["mb_out"]), "relay_ratio": float(st.get("ratio", 1.0)),
                    "sink_retained": bool(st.get("sink_retained", True)),
                    "peak_mem_mb": pmem, "batch_decode_s": dec_time, "batch_size": bsz,
                    "text_len_chars": len(text),
                })
        for rc in real_caches:
            del rc
        done = min((bi + 1) * B, n)
        print(f"[2x2] batch {bi+1}/{n_batches} ({done}/{n}) elapsed={time.time()-t0:.0f}s", flush=True)

    # --------------------- aggregate ---------------------
    arm_names = [a[0] for a in arms]

    def col(arm, key):
        return {r["idx"]: r[key] for r in rows if r["arm"] == arm}

    def num(arm, key):
        return {i: (1.0 if v is True else (0.0 if v is False else float(v)))
                for i, v in col(arm, key).items()}

    per_arm = {}
    for name, kind, asc_on in arms:
        rs = [r for r in rows if r["arm"] == name]
        acc = np.mean([r["correct"] for r in rs]) if rs else 0.0
        toks = [r["tokens"] for r in rs]
        per_arm[name] = {
            "relay": kind, "asc": bool(asc_on), "n": len(rs),
            "acc": float(acc),
            "mean_tokens": float(np.mean(toks)) if toks else 0.0,
            "median_tokens": float(np.median(toks)) if toks else 0.0,
            "mean_decode_s": float(np.mean([r["decode_s"] for r in rs])) if rs else 0.0,
            "mean_e2e_s": float(np.mean([r["e2e_s"] for r in rs])) if rs else 0.0,
            "mean_tok_per_s": float(np.mean([r["tok_per_s"] for r in rs])) if rs else 0.0,
            "mean_relay_mb": float(np.mean([r["relay_mb"] for r in rs])) if rs else 0.0,
            "mean_relay_positions": float(np.mean([r["relay_positions"] for r in rs])) if rs else 0.0,
            "eos_rate": float(np.mean([r["eos"] for r in rs])) if rs else 0.0,
            "mean_peak_mem_mb": float(np.mean([r["peak_mem_mb"] for r in rs])) if rs else 0.0,
        }

    # 2x2 interaction on the primary metrics (needs A & C; B & D if ASC ran)
    interaction = {}
    metrics = ["tokens", "decode_s", "e2e_s", "correct"]
    have = set(arm_names)
    for mkey in metrics:
        label = "acc" if mkey == "correct" else mkey
        blk = {}
        if {"A", "C"} <= have:
            blk["compression_tax_noASC (C-A)"] = paired_diff(num("A", mkey), num("C", mkey), seed=args.seed)
        if {"A", "B"} <= have:
            blk["ASC_effect_full (B-A)"] = paired_diff(num("A", mkey), num("B", mkey), seed=args.seed)
        if {"C", "D"} <= have:
            blk["ASC_effect_obf (D-C)"] = paired_diff(num("C", mkey), num("D", mkey), seed=args.seed)
        if {"A", "B", "C", "D"} <= have:
            # interaction = (D-C) - (B-A), computed per-item then bootstrapped
            a, b, c, d = num("A", mkey), num("B", mkey), num("C", mkey), num("D", mkey)
            common = sorted(set(a) & set(b) & set(c) & set(d))
            inter = np.asarray([(d[i] - c[i]) - (b[i] - a[i]) for i in common], float)
            m, lo, hi = bootstrap_ci(inter, seed=args.seed)
            blk["interaction ((D-C)-(B-A))"] = {"mean_diff": m, "ci_lo": lo, "ci_hi": hi,
                                                "n": len(common)}
        if {"A", "D"} <= have:
            blk["combined_vs_full (D-A)"] = paired_diff(num("A", mkey), num("D", mkey), seed=args.seed)
        if {"A", "S"} <= have:
            # content-independence crux: real (A) vs shuffled wrong-question cache (S)
            blk["shuffled_vs_full (S-A)"] = paired_diff(num("A", mkey), num("S", mkey), seed=args.seed)
        if {"A", "X"} <= have:
            # content-specificity ladder: real (A) vs cross-task cache (X)
            blk["crosstask_vs_full (X-A)"] = paired_diff(num("A", mkey), num("X", mkey), seed=args.seed)
        if {"A", "N"} <= have:
            # budget-artifact control: real (A) vs Judger-only (N)
            blk["none_vs_full (N-A)"] = paired_diff(num("A", mkey), num("N", mkey), seed=args.seed)
        interaction[label] = blk

    correctness_flips = {}
    if {"A", "C"} <= have:
        correctness_flips["C_vs_A (compression)"] = flips(num("A", "correct"), num("C", "correct"))
    if {"C", "D"} <= have:
        correctness_flips["D_vs_C (ASC|obf)"] = flips(num("C", "correct"), num("D", "correct"))
    if {"A", "D"} <= have:
        correctness_flips["D_vs_A (combined)"] = flips(num("A", "correct"), num("D", "correct"))

    summary = {
        "config": vars(args), "n": n, "arms": arm_names, "run_asc": run_asc,
        "compressors": {k: c.summary() for k, c in compressors.items()},
        "per_arm": per_arm, "interaction": interaction,
        "correctness_flips": correctness_flips,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(rows, f)

    # --------------------- print ---------------------
    print("\n=== PER-ARM ===", flush=True)
    hdr = f"{'arm':<4}{'relay':<7}{'asc':<5}{'acc':<7}{'tok':<7}{'dec_s':<8}{'e2e_s':<8}{'relayMB':<9}{'pos':<6}{'eos':<6}"
    print(hdr)
    for name in arm_names:
        a = per_arm[name]
        print(f"{name:<4}{a['relay']:<7}{('on' if a['asc'] else 'off'):<5}"
              f"{a['acc']:<7.3f}{a['mean_tokens']:<7.0f}{a['mean_decode_s']:<8.2f}"
              f"{a['mean_e2e_s']:<8.2f}{a['mean_relay_mb']:<9.2f}{a['mean_relay_positions']:<6.0f}"
              f"{a['eos_rate']:<6.2f}", flush=True)

    print("\n=== 2x2 INTERACTION (paired, 95% CI) ===", flush=True)
    for label, blk in interaction.items():
        print(f"[{label}]", flush=True)
        for k, d in blk.items():
            print(f"  {k:<28} {d['mean_diff']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] (n={d['n']})",
                  flush=True)

    if args.smoke:
        print("\n=== SMOKE GATE ===", flush=True)
        obf = per_arm.get("C", {})
        full = per_arm.get("A", {})
        checks = []
        smaller = obf.get("mean_relay_mb", 1e9) < full.get("mean_relay_mb", 0) and \
            obf.get("mean_relay_positions", 1e9) < full.get("mean_relay_positions", 0)
        checks.append(("H-OBF relay genuinely smaller (MB & positions)", smaller))
        sink_ok = all(r["sink_retained"] for r in rows if r["relay"] == "obf")
        checks.append(("sink positions retained after compression", sink_ok))
        valid = all(r["text_len_chars"] > 0 for r in rows) and \
            all(r["tokens"] > 0 for r in rows)
        checks.append(("no empty/invalid Judger output", valid))
        asc_ok = None
        if run_asc:
            b_tok = per_arm.get("B", {}).get("median_tokens", 1e9)
            a_tok = per_arm.get("A", {}).get("median_tokens", 0)
            d_tok = per_arm.get("D", {}).get("median_tokens", 1e9)
            c_tok = per_arm.get("C", {}).get("median_tokens", 0)
            asc_ok = (b_tok < a_tok) or (d_tok < c_tok)
            checks.append(("ASC shortens median output (B<A or D<C)", asc_ok))
        upstream_ok = getattr(wrapper, "seal", None) is None or not wrapper.seal._enabled
        checks.append(("upstream P/C/R unsteered (ASC judger-only)", upstream_ok))

        for desc, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {desc}", flush=True)

        # decision-tree branch
        if not smaller or not valid or not sink_ok:
            branch = "FIX compression wiring; stop GPU runs (interventions not functional)"
        elif run_asc and asc_ok is False:
            branch = "Tiny ASC-strength sweep (ASC did not shorten output)"
        else:
            branch = "Both pass -> proceed to 100-example paired 2x2"
        print(f"\n  NEXT: {branch}", flush=True)

    print(f"\n[2x2] wrote {os.path.join(args.out_dir, 'report.json')}", flush=True)
    print("EXP2X2_DONE", flush=True)


if __name__ == "__main__":
    main()
