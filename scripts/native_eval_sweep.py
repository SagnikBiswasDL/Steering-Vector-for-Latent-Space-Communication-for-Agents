"""Single-load evaluation sweep for native per-agent steering vs Judger-only SEAL.

Loads the model ONCE and evaluates many steering configurations by reconfiguring
a per-role SealSteerer between runs (much cheaper than re-launching run.py per
config). For each configuration it reports downstream accuracy and Judger
mean_output_tokens on the *test* set, so we can plot the accuracy-token Pareto
frontier and answer: does steering the (fixed-runtime) upstream agents with their
native correctness-contrastive vectors beat Judger-only SEAL?

Groups evaluated (each steered with the corresponding native vector(s)):
  control, planner, critic, refiner, judger, upstream3 (P+C+R), all.
Optionally a `judger_generic` group using an existing generic SEAL vector, and
`upstream3+judger` combos, for direct comparison.

Example:
  python scripts/native_eval_sweep.py \
      --model_name Qwen/Qwen3-14B --task gsm8k --split test --n 300 \
      --native_dir artifacts/seal_vectors/qwen3-14b/native_gsm8k \
      --coefs 40 80 --groups control planner critic refiner judger upstream3 all \
      --out_csv artifacts/sweeps/native_gsm8k.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (  # noqa: E402
    load_gsm8k, load_arc_challenge, load_medqa, load_arc_easy,
)
from methods.latent_mas import LatentMASMethod  # noqa: E402
from models import ModelWrapper  # noqa: E402
from seal.hooks import SealSteerer  # noqa: E402
from utils import auto_device, set_seed  # noqa: E402

GROUPS = {
    "control": [],
    "planner": ["planner"],
    "critic": ["critic"],
    "refiner": ["refiner"],
    "judger": ["judger"],
    "upstream3": ["planner", "critic", "refiner"],
    "all": ["planner", "critic", "refiner", "judger"],
    "upstream3+judger": ["planner", "critic", "refiner", "judger"],
}
ROLES = ["planner", "critic", "refiner", "judger"]


def load_dataset(task, split, n, offset=0):
    if task == "gsm8k":
        it = load_gsm8k(split=split)
    elif task == "arc_challenge":
        it = load_arc_challenge(split=split)
    elif task == "arc_easy":
        it = load_arc_easy(split=split)
    elif task == "medqa":
        # load_medqa ignores `split` (single local file); we carve a held-out
        # slice via `offset` so eval items are disjoint from the capture set.
        it = load_medqa(split=split)
    else:
        raise ValueError(f"unsupported task: {task}")
    out = []
    for i, ex in enumerate(it):
        if i < offset:
            continue
        out.append(ex)
        if n > 0 and len(out) >= n:
            break
    return out


def load_native_vectors(native_dir, roles):
    vecs, layer = {}, None
    for r in roles:
        # match "{role}_native_layer{L}.pt"
        cand = [f for f in os.listdir(native_dir) if f.startswith(f"{r}_native_layer") and f.endswith(".pt")]
        if not cand:
            print(f"[sweep] WARNING: no native vector file for role={r} in {native_dir}")
            continue
        blob = torch.load(os.path.join(native_dir, cand[0]), map_location="cpu")
        v = blob.get("unit_vector")
        if v is None:
            v = blob["vector"]; v = v / v.norm().clamp_min(1e-8)
        vecs[r] = v.float()
        li = int(blob["layer_index"])
        layer = li if layer is None else layer
        if li != layer:
            raise ValueError(f"inconsistent layer indices across native vectors ({li} vs {layer})")
    return vecs, layer


def evaluate_config(method, model, dataset, generate_bs, roles, coef, seed,
                    role_vectors, layer_index, override=None):
    set_seed(seed)
    steerer = model.seal
    # Reset then configure this run's per-role vectors/coefs.
    rv = dict(role_vectors)
    if override:
        rv.update(override)
    steerer.role_vectors = {r: v.float() for r, v in rv.items()}
    steerer.role_coefs = {r: float(coef) for r in roles}
    steerer.unit_vector = None
    steerer.coef = 0.0
    steerer.layer_index = int(layer_index)
    steerer.disable()
    model.seal_active_roles = set(roles)

    n_correct, total_tokens, n = 0, 0, 0
    t0 = time.time()
    for i in range(0, len(dataset), generate_bs):
        batch = dataset[i:i + generate_bs]
        results = method.run_batch(batch)
        for res in results:
            n += 1
            n_correct += 1 if res.get("correct") else 0
            total_tokens += int(res.get("output_tokens", 0))
    dt = time.time() - t0
    acc = n_correct / n if n else 0.0
    mean_tokens = total_tokens / n if n else 0.0
    return {"n": n, "accuracy": acc, "correct": n_correct,
            "mean_output_tokens": mean_tokens, "sec": round(dt, 1),
            "sec_per_sample": round(dt / n, 3) if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--task", type=str, default="gsm8k")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--data_offset", type=int, default=0,
                    help="Skip the first N dataset items (used for MedQA to hold out a disjoint eval slice).")
    ap.add_argument("--native_dir", type=str, required=True)
    ap.add_argument("--judger_generic", type=str, default=None,
                    help="Optional generic SEAL vector (.pt) for a judger_generic baseline group.")
    ap.add_argument("--coefs", type=float, nargs="+", default=[40, 80])
    ap.add_argument("--groups", type=str, nargs="+",
                    default=["control", "planner", "critic", "refiner", "judger", "upstream3", "all"])
    ap.add_argument("--latent_steps", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--generate_bs", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", type=str, default="sequential")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--device2", type=str, default="cuda:1")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    # Namespace fields the ModelWrapper / method expect.
    args.method = "latent_mas"
    args.use_vllm = False
    args.latent_space_realign = False
    args.think = False
    args.seal = False
    args.enable_prefix_caching = False
    args.use_second_HF_model = False
    args.capture_acts = None

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)

    role_vectors, layer_index = load_native_vectors(args.native_dir, ROLES)
    print(f"[sweep] loaded native vectors for {sorted(role_vectors)} at layer {layer_index}")

    # Install a persistent per-role steerer; reconfigured per config.
    steerer = SealSteerer(unit_vector=None, layer_index=int(layer_index), coef=0.0,
                          apply_to="last", role_vectors=role_vectors)
    steerer.register(model.model)
    steerer.disable()
    model.seal = steerer
    model.seal_active_roles = set()

    generic_vec = None
    if args.judger_generic:
        gblob = torch.load(args.judger_generic, map_location="cpu")
        gv = gblob.get("unit_vector")
        if gv is None:
            gv = gblob["vector"]; gv = gv / gv.norm().clamp_min(1e-8)
        generic_vec = gv.float()

    method = LatentMASMethod(
        model, latent_steps=args.latent_steps, judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, generate_bs=args.generate_bs, args=args,
    )

    dataset = load_dataset(args.task, args.split, args.n, offset=args.data_offset)
    print(f"[sweep] {len(dataset)} eval items on {args.task}/{args.split} (offset={args.data_offset})")

    # Build config list.
    configs = []
    if "control" in args.groups:
        configs.append(("control", [], 0.0, None))
    for g in args.groups:
        if g == "control":
            continue
        if g == "judger_generic":
            continue
        roles = GROUPS[g]
        for c in args.coefs:
            configs.append((g, roles, c, None))
    if "judger_generic" in args.groups and generic_vec is not None:
        for c in args.coefs:
            configs.append(("judger_generic", ["judger"], c, {"judger": generic_vec}))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fieldnames = ["group", "roles", "coef", "n", "accuracy", "correct",
                  "mean_output_tokens", "sec", "sec_per_sample"]
    rows = []
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (g, roles, coef, override) in configs:
            res = evaluate_config(method, model, dataset, args.generate_bs, roles, coef,
                                  args.seed, role_vectors, layer_index, override=override)
            row = {"group": g, "roles": "|".join(roles), "coef": coef, **res}
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"[sweep] {g:>16} coef={coef:>5} -> acc={res['accuracy']*100:.1f}% "
                  f"tokens={res['mean_output_tokens']:.1f} ({res['sec']}s)", flush=True)

    with open(args.out_csv.replace(".csv", ".json"), "w") as f:
        json.dump({"meta": {"task": args.task, "split": args.split, "n": args.n,
                            "model": args.model_name, "layer": layer_index,
                            "seed": args.seed, "coefs": args.coefs},
                   "rows": rows}, f, indent=2)
    print(f"[sweep] done -> {args.out_csv}")


if __name__ == "__main__":
    main()
