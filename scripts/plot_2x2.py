#!/usr/bin/env python3
"""Plot the {Full, H-OBF} x {ASC off, on} 2x2 as a 4-panel summary figure."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

report = sys.argv[1] if len(sys.argv) > 1 else "artifacts/exp2x2/paired/report.json"
out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/exp2x2/paired/plot_2x2.png"

with open(report) as f:
    R = json.load(f)
pa = R["per_arm"]

# arms A=full/off B=full/on C=obf/off D=obf/on
relays = ["full", "obf"]
ascs = [False, True]
def get(relay, asc, key):
    for name, a in pa.items():
        if a["relay"] == relay and a["asc"] == asc:
            return a[key]
    return float("nan")

panels = [
    ("Accuracy", "acc", 1.0, False),
    ("Mean Judger tokens", "mean_tokens", None, False),
    ("Relay KV (MB)", "mean_relay_mb", None, True),
    ("Judger decode time (s)", "mean_decode_s", None, False),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
x = np.arange(len(relays))
w = 0.36
colors = {False: "#8c9eb2", True: "#2b8cbe"}
for ax, (title, key, ymax, logy) in zip(axes, panels):
    for j, asc in enumerate(ascs):
        vals = [get(r, asc, key) for r in relays]
        bars = ax.bar(x + (j - 0.5) * w, vals, w,
                      label=("ASC on" if asc else "ASC off"), color=colors[asc])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Full relay", "H-OBF relay"])
    if logy:
        ax.set_yscale("log")
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(loc="lower right", fontsize=10)
cfg = R["config"]
fig.suptitle(
    f"Relay compression (H-OBF) x Judger concision (ASC) - "
    f"{cfg['task'].upper()} {cfg['model_name'].split('/')[-1]} "
    f"n={R['n']} k={cfg['k']} budget={cfg['judger_budget']} "
    f"| H-OBF: {get('obf',False,'mean_relay_positions'):.0f} vs "
    f"{get('full',False,'mean_relay_positions'):.0f} pos "
    f"({get('full',False,'mean_relay_mb')/get('obf',False,'mean_relay_mb'):.0f}x smaller)",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140)
print("wrote", out)
