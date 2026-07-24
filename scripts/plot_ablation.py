"""Two-panel Pareto plots for the LatentMAS agent-ablation sweep.

Reads the CSV from ablation_sweep.py and plots, for each agent-chain config:
  - left:  accuracy vs Judger output tokens (efficiency of the text emitter)
  - right: accuracy vs latency per example (total system cost)
Point size encodes the number of upstream latent agents. This visualizes
whether the upstream agents buy accuracy that justifies their token/compute cost.

Example:
  python scripts/plot_ablation.py --csv artifacts/sweeps/ablation_gsm8k.csv \
      --out_plot artifacts/plots/ablation_gsm8k.png --title "GSM8K n=300"
"""

import argparse
import csv
import os


def read_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            acc = float(r["accuracy"])
            rows.append({
                "config": r["config"],
                "accuracy": acc * 100.0 if acc <= 1.0 else acc,
                "tokens": float(r["mean_output_tokens"]),
                "latency": float(r["sec_per_sample"]),
                "n_upstream": int(r["n_upstream"]),
                "latent_forwards": int(r["latent_forwards"]),
            })
    return rows


def _scatter(ax, rows, xkey, xlabel):
    for r in rows:
        size = 90 + 120 * r["n_upstream"]
        ax.scatter(r[xkey], r["accuracy"], s=size, alpha=0.8, zorder=3)
        ax.annotate(r["config"], (r[xkey], r["accuracy"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("accuracy (%)")
    ax.grid(True, ls=":", alpha=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_plot", type=str, required=True)
    ap.add_argument("--title", type=str, default="LatentMAS agent ablation")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_rows(args.csv)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axl, rows, "tokens", "Judger mean output tokens (lower = cheaper)")
    axl.set_title("Accuracy vs Judger tokens")
    _scatter(axr, rows, "latency", "latency per example (s, lower = cheaper)")
    axr.set_title("Accuracy vs total latency")

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_plot)), exist_ok=True)
    fig.savefig(args.out_plot, dpi=150)
    print(f"[plot-ablation] plot -> {args.out_plot}")


if __name__ == "__main__":
    main()
