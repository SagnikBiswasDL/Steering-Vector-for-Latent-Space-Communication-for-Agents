#!/usr/bin/env python3
"""Diagnostic: is the ANSWER decodable from the upstream latent states?

The prior probe (RESULTS.md 9.1) asked "will the final output be correct?" and found
near-chance upstream AUC. That does NOT prove the cache lacks answer-relevant info: a
latent state could strongly encode "option C" without encoding whether C is correct.

Here we probe the MedQA gold answer choice (4-way) from each agent's per-layer final
latent residual, with BOTH a linear and a small MLP probe (5-fold CV). Interpretation:
  linear strong           -> answer info linearly available upstream
  MLP strong, linear weak -> info present but distributed/nonlinear
  both weak               -> upstream agents genuinely don't encode the answer

Raw features are saved (features.npz) so K/V-pooled or model-prediction labels can be
probed later without re-running the model.

Example:
  python scripts/diag_answer_probes.py --model_name Qwen/Qwen3-14B --k 10 \
    --split train --n 200 --out_dir artifacts/diag/answer_probes_14b
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

from data import load_medqa  # noqa: E402
from methods import default_agents  # noqa: E402
from models import ModelWrapper  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.hooks import _find_decoder_layers  # noqa: E402
from utils import set_seed, auto_device  # noqa: E402

CHOICE_TO_IDX = {"a": 0, "b": 1, "c": 2, "d": 3}


class MultiLayerRecorder:
    """Capture the last-token residual output of every decoder layer (latest forward)."""

    def __init__(self, model):
        self.layers = _find_decoder_layers(model)
        self.buf: List[torch.Tensor] = [None] * len(self.layers)
        self.handles = []

    def _mk(self, i):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            self.buf[i] = hs[:, -1, :].detach().float().to("cpu")
        return hook

    def register(self):
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._mk(i)))

    def snapshot(self) -> np.ndarray:
        # [n_layers, d] for batch size 1
        return np.stack([b[0].numpy() if b is not None else np.zeros(1) for b in self.buf])

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def make_ns(args) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name, task="medqa", prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=256,
        text_mas_context_length=-1, temperature=0.0, top_p=1.0, seed=args.seed,
        seal=False, kvsteer=False, ces=False, capture_acts=None, planner_steps=None,
        critic_steps=None, refiner_steps=None, latent_steps=0, latent_space_realign=False,
    )


def auc_ovr(scores: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Macro one-vs-rest AUC via the rank (Mann-Whitney) statistic."""
    aucs = []
    for c in range(n_classes):
        s = scores[:, c]
        pos = y == c
        neg = ~pos
        npos, nneg = int(pos.sum()), int(neg.sum())
        if npos == 0 or nneg == 0:
            continue
        order = np.argsort(s)
        ranks = np.empty(len(s), float)
        ranks[order] = np.arange(1, len(s) + 1)
        auc = (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def train_probe(X: np.ndarray, y: np.ndarray, *, kind: str, n_classes: int = 4,
                folds: int = 5, steps: int = 300, seed: int = 0):
    """k-fold linear or MLP probe; returns mean held-out accuracy and macro AUC."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    idx = rng.permutation(n)
    fold_sizes = [n // folds + (1 if i < n % folds else 0) for i in range(folds)]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    accs, aucs = [], []
    start = 0
    for fs in fold_sizes:
        te = idx[start:start + fs]
        tr = np.setdiff1d(idx, te)
        start += fs
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device=dev)
        Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device=dev)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=dev)
        if kind == "linear":
            net = nn.Linear(d, n_classes).to(dev)
        else:
            net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_classes)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
        lossf = nn.CrossEntropyLoss()
        net.train()
        for _ in range(steps):
            opt.zero_grad()
            loss = lossf(net(Xtr), ytr)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            logits = net(Xte)
            probs = torch.softmax(logits, -1).cpu().numpy()
            pred = probs.argmax(1)
        accs.append(float((pred == y[te]).mean()))
        aucs.append(auc_ovr(probs, y[te], n_classes))
    return float(np.mean(accs)), float(np.nanmean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="train", choices=["test", "train", "dev"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layer_stride", type=int, default=2, help="probe every Nth layer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/diag/answer_probes")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    items = list(load_medqa(split=args.split))
    if args.n > 0:
        items = items[: args.n]
    # keep only items with a clean a/b/c/d gold
    items = [it for it in items if str(it.get("gold", "")).strip().lower() in CHOICE_TO_IDX]
    n = len(items)
    print(f"[probe] n={n} split={args.split} k={args.k} model={args.model_name}", flush=True)

    ns = make_ns(args)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    up_agents = [a for a in default_agents() if a.role != "judger"]
    roles = [a.role for a in up_agents]
    rec = MultiLayerRecorder(wrapper.model)
    rec.register()

    feats: Dict[str, List[np.ndarray]] = {r: [] for r in roles}
    labels: List[int] = []
    t0 = time.time()
    for i, item in enumerate(items):
        q = item["question"]
        labels.append(CHOICE_TO_IDX[str(item["gold"]).strip().lower()])
        past = None
        for agent in up_agents:
            messages = build_agent_message_sequential_latent_mas(
                role=agent.role, question=q, context="", method="latent_mas", args=ns
            )
            _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
            past = wrapper.generate_latent_batch(
                ids, attention_mask=mask, latent_steps=int(args.k),
                past_key_values=past, role=agent.role,
            )
            feats[agent.role].append(rec.snapshot())  # [n_layers, d]
        del past
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[probe] captured {i+1}/{n} elapsed={time.time()-t0:.0f}s", flush=True)
    rec.remove()

    y = np.array(labels)
    n_layers = feats[roles[0]][0].shape[0]
    layers = list(range(0, n_layers, args.layer_stride))
    # stack: per role -> [n, n_layers, d]
    stacks = {r: np.stack(feats[r]).astype(np.float16) for r in roles}
    np.savez_compressed(os.path.join(args.out_dir, "features.npz"),
                        y=y, layers=np.array(layers), roles=np.array(roles, dtype=object),
                        **{f"X_{r}": stacks[r] for r in roles})

    print(f"[probe] class balance: {np.bincount(y, minlength=4).tolist()} (chance acc={1/4:.2f})", flush=True)
    report = {"config": vars(args), "n": n, "n_layers": n_layers,
              "class_balance": np.bincount(y, minlength=4).tolist(), "results": {}}
    for r in roles:
        Xr = stacks[r].astype(np.float32)  # [n, L, d]
        report["results"][r] = {}
        for L in layers:
            XL = Xr[:, L, :]
            lin_acc, lin_auc = train_probe(XL, y, kind="linear", seed=args.seed)
            mlp_acc, mlp_auc = train_probe(XL, y, kind="mlp", seed=args.seed)
            report["results"][r][str(L)] = {
                "linear_acc": lin_acc, "linear_auc": lin_auc,
                "mlp_acc": mlp_acc, "mlp_auc": mlp_auc,
            }
            print(f"PROBE role={r} layer={L} lin_acc={lin_acc:.3f} lin_auc={lin_auc:.3f} "
                  f"mlp_acc={mlp_acc:.3f} mlp_auc={mlp_auc:.3f}", flush=True)

    # best-layer summary per role
    best = {}
    for r in roles:
        rr = report["results"][r]
        bl = max(rr, key=lambda L: rr[L]["mlp_acc"])
        best[r] = {"best_layer": int(bl), **rr[bl]}
    report["best_per_role"] = best
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    for r in roles:
        b = best[r]
        print(f"PROBE_BEST role={r} layer={b['best_layer']} lin_acc={b['linear_acc']:.3f} "
              f"mlp_acc={b['mlp_acc']:.3f} (chance {1/4:.2f})", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
