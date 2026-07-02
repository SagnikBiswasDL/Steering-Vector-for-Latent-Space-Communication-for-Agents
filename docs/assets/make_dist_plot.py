"""Regenerate the per-agent thought-distribution plot locally from measured
fractions (GSM8K, Qwen3-14B, isolation, n=50). Numbers from
artifacts/analysis/agent_thoughts_gsm8k.json produced on the pod.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

roles = ["Planner", "Critic", "Refiner", "Judger"]
execf = [0.875, 0.878, 0.888, 0.882]
reflf = [0.114, 0.087, 0.103, 0.102]
transf = [0.011, 0.036, 0.009, 0.017]

x = np.arange(len(roles))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x, execf, label="execution")
ax.bar(x, reflf, bottom=execf, label="reflection")
ax.bar(x, transf, bottom=[e + r for e, r in zip(execf, reflf)], label="transition")
ax.set_xticks(x)
ax.set_xticklabels(roles)
ax.set_ylabel("fraction of reasoning steps")
ax.set_ylim(0, 1.08)
ax.set_title("Thought-type distribution per LatentMAS agent (GSM8K, Qwen3-14B)")
ax.legend(loc="lower right")
for i in range(len(roles)):
    ne = reflf[i] + transf[i]
    ax.text(i, 1.01, f"non-exec {ne:.0%}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_thoughts_gsm8k.png")
fig.savefig(out, dpi=150)
print("saved", out)
