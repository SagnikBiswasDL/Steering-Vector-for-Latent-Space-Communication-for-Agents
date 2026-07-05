"""Build native, in-pipeline correctness-contrastive steering vectors per agent.

Reads one or more activation caches produced by
`run.py --capture_acts ...` (each holds per-agent layer-L latent activations
captured *inside the full pipeline* plus the run's final correctness) and, for
each agent, computes

    v_agent = mean(acts | final answer correct) - mean(acts | incorrect)

saved as a standard SEAL artifact (unit_vector, vector, raw_norm, layer_index)
so `SealSteerer.from_artifact(...)` / `run.py --seal_vector ...` can load it.

Example:
  python scripts/build_native_vectors.py \
      --cache artifacts/capture/gsm8k_train_s42.pt \
      --out_dir artifacts/seal_vectors/qwen3-14b/native_gsm8k
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seal.capture import build_contrastive_vector, correctness_probe_auc  # noqa: E402

ROLES = ["planner", "critic", "refiner", "judger"]


def load_and_merge(cache_paths):
    """Merge caches into per-role (acts, correct) with aligned correctness.

    Each cache stores per-role activation rows plus the row->global-run index
    map, so we can attach the correct label to each activation row.
    """
    per_role_acts = {r: [] for r in ROLES}
    per_role_corr = {r: [] for r in ROLES}
    meta = None
    for path in cache_paths:
        blob = torch.load(path, map_location="cpu")
        if meta is None:
            meta = {k: blob.get(k) for k in ("layer_index", "model_name", "task", "split")}
        correct = blob["correct"].bool()
        acts = blob["acts"]
        row_index = blob.get("acts_row_index", None)
        for r in ROLES:
            a = acts.get(r, torch.empty(0))
            if a.numel() == 0:
                continue
            if row_index is not None and r in row_index and len(row_index[r]) == a.shape[0]:
                idx = torch.tensor(row_index[r], dtype=torch.long)
                c = correct[idx]
            else:
                # Fallback: assume rows align 1:1 with runs (capture always fires).
                c = correct[: a.shape[0]]
            per_role_acts[r].append(a.float())
            per_role_corr[r].append(c)
    merged = {}
    for r in ROLES:
        if per_role_acts[r]:
            merged[r] = (torch.cat(per_role_acts[r], 0), torch.cat(per_role_corr[r], 0))
    return merged, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, nargs="+", required=True,
                    help="One or more capture caches (.pt) to merge.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--roles", type=str, default=",".join(ROLES))
    ap.add_argument("--probe_folds", type=int, default=5)
    args = ap.parse_args()

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    merged, meta = load_and_merge(args.cache)
    layer_index = int(meta.get("layer_index", 28))
    os.makedirs(args.out_dir, exist_ok=True)

    report = {"meta": meta, "roles": {}}
    for r in roles:
        if r not in merged:
            print(f"[build] no activations for role={r}; skipping")
            continue
        acts, correct = merged[r]
        n_c = int(correct.sum()); n_i = int((~correct).sum())
        if n_c == 0 or n_i == 0:
            print(f"[build] role={r}: need both classes (correct={n_c}, incorrect={n_i}); skipping vector")
            report["roles"][r] = {"n": int(acts.shape[0]), "n_correct": n_c, "n_incorrect": n_i,
                                  "note": "degenerate (single class)"}
            continue
        res = build_contrastive_vector(acts, correct, normalize=True)
        probe = correctness_probe_auc(acts, correct, k=args.probe_folds)
        blob = {
            "vector": res["vector"],
            "unit_vector": res["unit_vector"],
            "raw_norm": res["raw_norm"],
            "layer_index": layer_index,
            "role": r,
            "kind": "native_correct_contrastive",
            "n": int(acts.shape[0]),
            "n_correct": res["n_correct"],
            "n_incorrect": res["n_incorrect"],
            "model_name": meta.get("model_name"),
            "task": meta.get("task"),
            "split": meta.get("split"),
            "probe_auc": probe,
        }
        out_path = os.path.join(args.out_dir, f"{r}_native_layer{layer_index}.pt")
        torch.save(blob, out_path)
        report["roles"][r] = {
            "path": out_path,
            "n": int(acts.shape[0]),
            "n_correct": res["n_correct"],
            "n_incorrect": res["n_incorrect"],
            "raw_norm": float(res["raw_norm"]),
            "probe_auc": probe,
        }
        print(f"[build] {r}: n={acts.shape[0]} (c={n_c}/i={n_i}) raw_norm={float(res['raw_norm']):.4f} "
              f"probe_auc={probe['auc_mean']:.3f}+/-{probe['auc_std']:.3f} -> {out_path}")

    report_path = os.path.join(args.out_dir, "build_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[build] report -> {report_path}")


if __name__ == "__main__":
    main()
