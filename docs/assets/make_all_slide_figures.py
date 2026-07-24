"""Generate presentation figures for activation-steering slides.

All numbers are from documented results (RESULTS.md, GATE1_STATUS.md, KV findings).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(OUT), "figs")
os.makedirs(FIG, exist_ok=True)

# Clean, presentation-friendly style
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name: str) -> None:
    for d in (OUT, FIG):
        path = os.path.join(d, name)
        fig.savefig(path)
        print("saved", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. LatentMAS architecture schematic
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("LatentMAS sequential pipeline (shared KV; only Judger emits text)")

    boxes = [
        (0.3, 1.4, "Question", "#e8e8e8"),
        (1.9, 1.4, "Planner\nK latent", "#9ecae1"),
        (3.6, 1.4, "Critic\nK latent", "#9ecae1"),
        (5.3, 1.4, "Refiner\nK latent", "#9ecae1"),
        (7.0, 1.4, "Judger\ntext decode", "#fc9272"),
        (8.7, 1.4, "Answer", "#e8e8e8"),
    ]
    for x, y, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                1.4,
                1.3,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor=color,
                edgecolor="#333",
                linewidth=1.2,
            )
        )
        ax.text(x + 0.7, y + 0.65, label, ha="center", va="center", fontsize=10, fontweight="bold")

    for x in (1.7, 3.4, 5.1, 6.8, 8.5):
        ax.annotate(
            "",
            xy=(x + 0.2, 2.05),
            xytext=(x - 0.05, 2.05),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
        )

    # KV growing bar
    ax.add_patch(
        FancyBboxPatch(
            (1.9, 0.35),
            6.5,
            0.7,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#fff7bc",
            edgecolor="#b8860b",
            linewidth=1.2,
        )
    )
    ax.text(
        5.15,
        0.7,
        "shared KV cache grows (working memory)  ·  Planner → Critic → Refiner → Judger",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.text(7.7, 2.95, "SEAL / KV steer\nhooks here", ha="center", fontsize=8, color="#b30000")
    ax.annotate(
        "",
        xy=(7.7, 2.7),
        xytext=(7.7, 2.95),
        arrowprops=dict(arrowstyle="->", color="#b30000", lw=1.2),
    )
    save(fig, "arch_latentmas.png")


# ---------------------------------------------------------------------------
# 2. SEAL mechanism schematic
# ---------------------------------------------------------------------------
def fig_seal_mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: thought geometry cartoon
    ax = axes[0]
    rng = np.random.default_rng(0)
    exec_pts = rng.normal([0.0, 0.0], 0.35, size=(40, 2))
    refl_pts = rng.normal([2.2, 0.8], 0.35, size=(18, 2))
    ax.scatter(exec_pts[:, 0], exec_pts[:, 1], c="#2ca02c", s=28, alpha=0.75, label="execution")
    ax.scatter(refl_pts[:, 0], refl_pts[:, 1], c="#d62728", s=28, alpha=0.75, label="refl+trans")
    m_e = exec_pts.mean(0)
    m_r = refl_pts.mean(0)
    v = m_e - m_r
    ax.annotate(
        "",
        xy=m_e,
        xytext=m_r,
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.5),
    )
    ax.text(1.1, -0.9, r"$v = \mu_{exec} - \mu_{refl\cup trans}$", color="#1f77b4", fontsize=10)
    ax.set_title("Offline: contrastive vector at layer L")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Right: residual hook
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Online: residual-stream hook (last token)")
    layers = ["…", "L−1", "L", "L+1", "…"]
    for i, lab in enumerate(layers):
        y = 4.5 - i * 0.7
        ax.add_patch(
            FancyBboxPatch(
                (1.5, y),
                4.0,
                0.55,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#deebf7" if lab == "L" else "#f0f0f0",
                edgecolor="#333",
            )
        )
        ax.text(3.5, y + 0.27, f"decoder layer {lab}", ha="center", va="center", fontsize=9)
    ax.annotate(
        "",
        xy=(7.2, 3.15),
        xytext=(5.6, 3.15),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=2),
    )
    ax.text(7.4, 3.15, r"$h_{:,-1,:}\ +=\ \alpha\, v$", va="center", fontsize=11, color="#d62728")
    ax.text(5.0, 0.6, r"$\alpha>0$ nudges away from reflection $\Rightarrow$ shorter CoT", ha="center", fontsize=9)
    save(fig, "seal_mechanism.png")


# ---------------------------------------------------------------------------
# 3. Judger dose-response (tokens + accuracy)
# ---------------------------------------------------------------------------
def fig_dose_response():
    coef = np.array([0, 20, 40, 60, 80])
    acc = np.array([93.3, 94.2, 95.0, 94.2, 94.2])
    tok = np.array([621.3, 561.8, 514.6, 436.9, 379.4])

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    color_t = "#1f77b4"
    color_a = "#d62728"
    ax1.plot(coef, tok, "o-", color=color_t, lw=2, markersize=8, label="mean output tokens")
    ax1.set_xlabel("SEAL coefficient α (Judger, layer 28)")
    ax1.set_ylabel("mean Judger output tokens", color=color_t)
    ax1.tick_params(axis="y", labelcolor=color_t)
    ax1.set_xticks(coef)

    ax2 = ax1.twinx()
    ax2.plot(coef, acc, "s--", color=color_a, lw=2, markersize=7, label="accuracy")
    ax2.set_ylabel("accuracy (%)", color=color_a)
    ax2.tick_params(axis="y", labelcolor=color_a)
    ax2.set_ylim(90, 97)
    ax2.spines["top"].set_visible(False)

    ax1.set_title("GSM8K Judger SEAL dose–response (n=120, Qwen3-14B)")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right", frameon=False)
    ax1.annotate("−39% tokens\nacc held", xy=(80, 379), xytext=(55, 480),
                 arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9)
    save(fig, "dose_response_gsm8k.png")


# ---------------------------------------------------------------------------
# 4. Per-agent token bars
# ---------------------------------------------------------------------------
def fig_per_agent():
    agents = ["Judger", "Planner", "Critic", "Refiner"]
    c40 = [515, 580, 635, 623]
    c80 = [379, 593, 615, 608]
    control = 621

    x = np.arange(len(agents))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(control, color="#666", ls="--", lw=1.2, label=f"control ({control})")
    b1 = ax.bar(x - w / 2, c40, w, label="coef 40", color="#9ecae1")
    b2 = ax.bar(x + w / 2, c80, w, label="coef 80", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("mean Judger output tokens")
    ax.set_title("Per-agent SEAL (GSM8K): only Judger shrinks tokens")
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{int(h)}", ha="center", va="bottom", fontsize=8)
    save(fig, "per_agent_tokens_gsm8k.png")


# ---------------------------------------------------------------------------
# 5. Probe AUC grouped bars
# ---------------------------------------------------------------------------
def fig_probe_auc():
    agents = ["Planner", "Critic", "Refiner", "Judger"]
    gsm = [0.54, 0.62, 0.59, 0.69]
    arc = [0.57, 0.51, 0.46, 0.73]
    med = [0.57, 0.63, 0.64, 0.80]
    x = np.arange(len(agents))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(x - w, gsm, w, label="GSM8K", color="#6baed6")
    ax.bar(x, arc, w, label="ARC-C", color="#fd8d3c")
    ax.bar(x + w, med, w, label="MedQA", color="#74c476")
    ax.axhline(0.5, color="#999", ls=":", lw=1.2, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("correctness-probe AUC")
    ax.set_ylim(0.4, 0.9)
    ax.set_title("Where is final correctness linearly encoded? (layer-28 activations)")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    save(fig, "probe_auc_all.png")


# ---------------------------------------------------------------------------
# 6. Gate-1 MedQA dual panel
# ---------------------------------------------------------------------------
def fig_gate1_medqa():
    K = np.array([0, 5, 10, 20, 40])
    acc = np.array([0.78, 0.77, 0.83, 0.79, 0.79])
    acc_lo = np.array([0.69, 0.68, 0.76, 0.71, 0.71])
    acc_hi = np.array([0.86, 0.85, 0.90, 0.87, 0.86])
    lat = np.array([40.6, 26.2, 22.9, 24.9, 25.8])
    tok = np.array([1731, 1097, 944, 992, 967])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax = axes[0]
    ax.errorbar(
        K,
        acc * 100,
        yerr=[(acc - acc_lo) * 100, (acc_hi - acc) * 100],
        fmt="o-",
        color="#3182bd",
        lw=2,
        capsize=3,
        label="accuracy ± CI",
    )
    ax.set_xlabel("latent steps K per upstream agent")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("MedQA accuracy vs K")
    ax.set_xticks(K)
    ax.axvline(10, color="#e6550d", ls="--", alpha=0.7, label="best K=10")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(K, lat, "o-", color="#e6550d", lw=2, label="latency (s)")
    ax.set_xlabel("latent steps K per upstream agent")
    ax.set_ylabel("mean latency (s / example)", color="#e6550d")
    ax.tick_params(axis="y", labelcolor="#e6550d")
    ax2 = ax.twinx()
    ax2.plot(K, tok, "s--", color="#3182bd", lw=2, label="Judger tokens")
    ax2.set_ylabel("mean Judger output tokens", color="#3182bd")
    ax2.tick_params(axis="y", labelcolor="#3182bd")
    ax2.spines["top"].set_visible(False)
    ax.set_title("Latency & Judger tokens vs K")
    ax.set_xticks(K)
    ax.axvline(10, color="#e6550d", ls="--", alpha=0.5)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Gate 1: MedQA n=100 @ max_new_tokens=4096 — smaller K is slower", y=1.02)
    save(fig, "gate1_medqa_kcurve.png")


# ---------------------------------------------------------------------------
# 7. Gate-1 AIME dual panel
# ---------------------------------------------------------------------------
def fig_gate1_aime():
    K = np.array([0, 10, 40])
    acc = np.array([0.53, 0.67, 0.50]) * 100
    lat = np.array([178, 146, 159])
    tok = np.array([7237, 5927, 6401])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(K, acc, "o-", color="#3182bd", lw=2, markersize=8)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("AIME 2024 accuracy vs K (n=30)")
    axes[0].set_xticks(K)
    axes[0].set_ylim(40, 75)

    ax = axes[1]
    ax.plot(K, lat, "o-", color="#e6550d", lw=2, markersize=8, label="latency (s)")
    ax.set_xlabel("K")
    ax.set_ylabel("latency (s)", color="#e6550d")
    ax.tick_params(axis="y", labelcolor="#e6550d")
    ax2 = ax.twinx()
    ax2.plot(K, tok, "s--", color="#3182bd", lw=2, markersize=7, label="Judger tokens")
    ax2.set_ylabel("Judger tokens", color="#3182bd")
    ax2.tick_params(axis="y", labelcolor="#3182bd")
    ax2.spines["top"].set_visible(False)
    ax.set_title("AIME 2024 latency & tokens")
    ax.set_xticks(K)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, frameon=False, fontsize=8)
    fig.suptitle("Gate 1 probe @ 8192: K=10 best; K=0 slower despite fewer latent forwards", y=1.02)
    save(fig, "gate1_aime_kcurve.png")


# ---------------------------------------------------------------------------
# 8. Cross-task transfer bars
# ---------------------------------------------------------------------------
def fig_transfer():
    tasks = ["GSM8K\ncoef80", "ARC-C\ncoef40", "MedQA\ncoef60"]
    dtok = [-38.9, -25.2, -12.2]
    dacc = [0.9, 0.0, -5.0]
    x = np.arange(len(tasks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, dtok, w, label="Δ tokens (%)", color="#3182bd")
    ax.bar(x + w / 2, dacc, w, label="Δ accuracy (pts)", color="#e6550d")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("change vs control")
    ax.set_title("Cross-task transfer of GSM8K SEAL vector (Judger)")
    ax.legend(frameon=False)
    save(fig, "transfer_bars.png")


# ---------------------------------------------------------------------------
# 9. Thought distribution (regenerate)
# ---------------------------------------------------------------------------
def fig_thoughts():
    roles = ["Planner", "Critic", "Refiner", "Judger"]
    execf = [0.875, 0.878, 0.888, 0.882]
    reflf = [0.114, 0.087, 0.103, 0.102]
    transf = [0.011, 0.036, 0.009, 0.017]
    x = np.arange(len(roles))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x, execf, label="execution", color="#74c476")
    ax.bar(x, reflf, bottom=execf, label="reflection", color="#fc9272")
    ax.bar(
        x,
        transf,
        bottom=[e + r for e, r in zip(execf, reflf)],
        label="transition",
        color="#fdae6b",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(roles)
    ax.set_ylabel("fraction of reasoning steps")
    ax.set_ylim(0, 1.08)
    ax.set_title("Thought-type mix is nearly uniform across roles (GSM8K isolation)")
    ax.legend(loc="lower right", frameon=False)
    for i in range(len(roles)):
        ne = reflf[i] + transf[i]
        ax.text(i, 1.01, f"non-exec {ne:.0%}", ha="center", va="bottom", fontsize=8)
    save(fig, "agent_thoughts_gsm8k.png")


# ---------------------------------------------------------------------------
# 10. Synthesis localization cartoon
# ---------------------------------------------------------------------------
def fig_localization():
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Converging evidence: intervention leverage at the Judger")

    items = [
        (0.4, "SEAL residual\nJudger: −39% tok", "#9e9ac8"),
        (2.6, "Native upstream\nnull / no tok win", "#fcbba1"),
        (4.8, "KV handoff\nno gain / crash", "#fcbba1"),
        (7.0, "KV judger_token\n−33% tok", "#9e9ac8"),
        (9.0, "K-curve:\nJudger dominates\nwall-clock", "#9e9ac8"),
    ]
    for x, lab, c in items:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.85, 1.2),
                1.7,
                1.8,
                boxstyle="round,pad=0.04,rounding_size=0.1",
                facecolor=c,
                edgecolor="#333",
            )
        )
        ax.text(x, 2.1, lab, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(5, 0.5, "purple = Judger-side win · peach = upstream / handoff null", ha="center", fontsize=9)
    save(fig, "synthesis_localization.png")


if __name__ == "__main__":
    fig_architecture()
    fig_seal_mechanism()
    fig_dose_response()
    fig_per_agent()
    fig_probe_auc()
    fig_gate1_medqa()
    fig_gate1_aime()
    fig_transfer()
    fig_thoughts()
    fig_localization()
    print("done")
