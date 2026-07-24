"""Accuracy-token Pareto plot for the native-vector steering sweep.

Reads the CSV from `native_eval_sweep.py` and plots accuracy (y) vs Judger mean
output tokens (x). Each steering group is a colored series over its coef sweep;
control is a star. The Pareto frontier (max accuracy at min tokens, i.e.
up-and-left) is outlined so we can see whether upstream steering pushes the
frontier beyond Judger-only SEAL.

Example:
  python scripts/plot_pareto.py \
      --csv artifacts/sweeps/native_gsm8k.csv \
      --out_plot artifacts/plots/pareto_gsm8k.png --title "GSM8K (n=300)"
"""

import argparse
import csv
import os


def read_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "group": r["group"],
                "coef": float(r["coef"]),
                "accuracy": float(r["accuracy"]) * (100.0 if float(r["accuracy"]) <= 1.0 else 1.0),
                "tokens": float(r["mean_output_tokens"]),
            })
    return rows


def pareto_front(points):
    """points: list of (tokens, acc). Frontier = min tokens & max acc (up-left)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best_acc = [], -1e9
    for tok, acc in pts:
        if acc > best_acc:
            front.append((tok, acc))
            best_acc = acc
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_plot", type=str, required=True)
    ap.add_argument("--title", type=str, default="Accuracy-token frontier")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_rows(args.csv)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    all_points = []
    for gi, (g, rs) in enumerate(sorted(groups.items())):
        rs = sorted(rs, key=lambda x: x["coef"])
        xs = [r["tokens"] for r in rs]
        ys = [r["accuracy"] for r in rs]
        all_points += list(zip(xs, ys))
        if g == "control":
            ax.scatter(xs, ys, marker="*", s=320, color="black", zorder=5, label="control")
            for r in rs:
                ax.annotate("control", (r["tokens"], r["accuracy"]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8)
            continue
        color = cmap(gi % 10)
        ax.plot(xs, ys, "-o", color=color, label=g, alpha=0.85)
        for r in rs:
            ax.annotate(f"{int(r['coef'])}", (r["tokens"], r["accuracy"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)

    front = pareto_front(all_points)
    if front:
        fx = [p[0] for p in front]
        fy = [p[1] for p in front]
        ax.plot(fx, fy, "--", color="gray", lw=1.5, zorder=1, label="Pareto frontier")

    ax.set_xlabel("Judger mean output tokens (lower = cheaper)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(args.title)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_plot)), exist_ok=True)
    fig.savefig(args.out_plot, dpi=150)
    print(f"[pareto] plot -> {args.out_plot}")


if __name__ == "__main__":
    main()
