"""Error analysis over captured pipeline runs: where do wrong answers originate?

Uses the activation caches from `run.py --capture_acts ...` to answer, with
runtime evidence, *where* the pipeline's failures are decided:

  1. Per-agent correctness probe (AUC): how linearly separable is final
     correctness from each agent's in-pipeline layer-L latent state? A high AUC
     that appears early (Planner) means the outcome is largely determined by the
     plan; a jump only at the Judger means failures are late (decode-side). This
     motivates which agent's native vector should matter.

  2. Judger token budget vs correctness: do wrong answers over-/under-generate?

  3. Answer-format taxonomy: of the wrong answers, how many produced no
     parseable \\boxed{} answer (a decode/format failure) vs a parsed-but-wrong
     value (a reasoning failure).

Outputs a JSON summary and a bar plot of per-agent probe AUC.

Example:
  python scripts/analyze_pipeline_errors.py \
      --cache artifacts/capture/gsm8k_train_s42.pt \
      --out_json artifacts/analysis/errors_gsm8k.json \
      --out_plot artifacts/plots/probe_auc_gsm8k.png
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seal.capture import correctness_probe_auc  # noqa: E402
from scripts.build_native_vectors import load_and_merge, ROLES  # noqa: E402


def make_plot(auc_by_role, out_plot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    roles = [r for r in ROLES if r in auc_by_role]
    means = [auc_by_role[r]["auc_mean"] for r in roles]
    stds = [auc_by_role[r]["auc_std"] for r in roles]
    x = np.arange(len(roles))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylabel("correctness probe AUC (layer-L latent state)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Where is the outcome decided? Per-agent correctness separability")
    for i, m in enumerate(means):
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)), exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    print(f"[errors] plot -> {out_plot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, nargs="+", required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_plot", type=str, default=None)
    ap.add_argument("--probe_folds", type=int, default=5)
    args = ap.parse_args()

    # Per-role activations for the probe.
    merged, meta = load_and_merge(args.cache)
    auc_by_role = {}
    for r in ROLES:
        if r not in merged:
            continue
        acts, correct = merged[r]
        if int(correct.sum()) == 0 or int((~correct).sum()) == 0:
            continue
        auc_by_role[r] = correctness_probe_auc(acts, correct, k=args.probe_folds)
        print(f"[errors] probe {r}: AUC={auc_by_role[r]['auc_mean']:.3f}+/-{auc_by_role[r]['auc_std']:.3f}")

    # Run-level stats from raw caches (correctness, tokens, answer format).
    all_correct, all_tokens, all_pred, all_gold = [], [], [], []
    for path in args.cache:
        blob = torch.load(path, map_location="cpu")
        all_correct += blob["correct"].bool().tolist()
        all_tokens += blob["output_tokens"].long().tolist()
        all_pred += list(blob.get("prediction", []))
        all_gold += list(blob.get("gold", []))

    n = len(all_correct)
    n_correct = sum(all_correct)
    n_wrong = n - n_correct

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    tok_correct = [t for t, c in zip(all_tokens, all_correct) if c]
    tok_wrong = [t for t, c in zip(all_tokens, all_correct) if not c]

    # Format taxonomy: an empty parsed prediction == no/unparseable boxed answer.
    parse_fail = sum(1 for p, c in zip(all_pred, all_correct) if (not c) and (p is None or str(p).strip() == ""))
    wrong_value = n_wrong - parse_fail

    summary = {
        "meta": meta,
        "n": n,
        "accuracy": (n_correct / n) if n else 0.0,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "probe_auc": auc_by_role,
        "judger_tokens": {
            "mean_correct": _mean(tok_correct),
            "mean_wrong": _mean(tok_wrong),
            "mean_all": _mean(all_tokens),
        },
        "error_taxonomy": {
            "parse_fail": parse_fail,
            "parse_fail_frac_of_wrong": (parse_fail / n_wrong) if n_wrong else 0.0,
            "wrong_value": wrong_value,
            "wrong_value_frac_of_wrong": (wrong_value / n_wrong) if n_wrong else 0.0,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[errors] json -> {args.out_json}")
    print(json.dumps(summary, indent=2))

    if args.out_plot and auc_by_role:
        make_plot(auc_by_role, args.out_plot)


if __name__ == "__main__":
    main()
