# Latent Scaffold — Compute Results & Next Directions

Updated: 2026-07-21. Focused record of the "scaffold" line (can the upstream latent agents
be replaced by a synthetic KV scaffold?) so we can pick the next direction. Full context:
[`DIAG_READOUT_FINDINGS.md`](DIAG_READOUT_FINDINGS.md), [`EXPERIMENTS_UNIFIED_RESULTS.md`](EXPERIMENTS_UNIFIED_RESULTS.md).

## The question

LatentMAS's benefit is **content-independent** (a wrong-question cache = the right one) and
mostly a **conciseness/token-budget** effect. So: what minimal object reproduces it, and can
we replace the 3 upstream agents (Planner/Critic/Refiner = **33 latent forwards**) with a
cheap synthetic scaffold?

## Setup

- LatentMAS, K=10, greedy (temp=0). Judger decodes with its cache replaced by a variant.
- Metric: MedQA accuracy + mean Judger tokens, at a fixed Judger token budget.
- Cache variants: `real` (full pipeline), `none` (judger-only), `shuffled` (wrong-question
  real cache), `gauss` (N(0,1)), `matched` (per-channel mean/std noise, full length),
  `repeat_last` (one latent vector ×all positions), `trunc{m}` (last m real positions),
  `pca{r}` (rank-r reconstruction), `synthglobal` (ONE fixed synthetic cache built from the
  averaged per-channel stats of 8 train caches — question-independent, zero upstream forwards).
- Script: `scripts/diag_scaffold_sweep.py`. Artifacts: `artifacts/diag/{4b,14b}/`.

## Results

### 4B MedQA, budget 1024, n=40 (real=0.450, none=0.150) — nothing synthetic works

| variant | acc | verdict |
|---|---|---|
| real / shuffled | 0.450 / 0.450 | reproduce (content-independent) |
| gauss / matched | 0.125 / 0.050 | fail |
| repeat_last | 0.075 | fail |
| trunc4/16/32/64/256 | 0.075–0.175 | fail (all lengths) |
| pca8/16/32/64 | 0.000–0.050 | fail (all ranks) |

→ On 4B, only a full real/shuffled cache works; every synthetic/compressed form collapses
to ≈none. (4B is brittle: low-rank/matched caches are off-manifold and degenerate output.)

### 14B MedQA, budget 1024, n=40 (real=0.675, none=0.350) — the FLIP

| variant | acc | tokens | vs real |
|---|---|---|---|
| real | 0.675 | — | — |
| shuffled | 0.625 | 712 | −0.05 (n.s.) |
| **matched** (per-item stat noise) | 0.600 | 712 | −0.075 (n.s.) |
| **pca32** (rank-32) | 0.650 | 711 | −0.025 (n.s.) |
| **synthglobal** (fixed, question-independent) | **0.600** | **701** | **−0.075 (n.s.)** |
| repeat_last | 0.225 | 957 | −0.45 |
| trunc64 | 0.400 | 490 | −0.275 |
| none | 0.350 | (long) | −0.325 |

→ On the capable model, the scaffold is **synthesizable from coarse statistics at full
length**. `matched`, `pca32`, and — crucially — `synthglobal` (a single fixed,
question-independent synthetic cache, **0 upstream forwards**) all reproduce real accuracy
(within noise) and conciseness, and beat judger-only by +0.25. `repeat_last` (single vector)
and `trunc64` (short) still fail → the scaffold needs **full length + full-rank-ish coarse
structure**, but not real content.

### Hardening (DONE) — holds, with a persistent ~6pt gap to real

MedQA 14B, **n=80**, budget 1024: real 0.600, **synthglobal 0.537**, none 0.338
(synth−real = −0.062 [−0.150, +0.025], n.s.; recovers ~76% of the real−none gap).

GSM8K 14B, n=60 (second task ✓):
- budget 512: real 0.567 / **synth 0.567** / none 0.467 (full recovery)
- budget 1024: real 0.933 / **synth 0.883** / none 0.800 (~62% recovery)

### Robustness (multi-seed) — NOT a lucky cache

14B MedQA b1024 n=40, 3 independent synth seeds: **0.55 / 0.60 / 0.65** (all ≫ none 0.35).

### Shrinking grid (14B MedQA b1024 n=40; real=0.675, none=0.350)

Length (`artifacts/diag/14b/grid_len/`):

| length | acc | note |
|---|---|---|
| full (~1023) | 0.60 | baseline synth |
| 256 | 0.300 | anomalous dip (single-seed noise?) |
| 64 | 0.575 | works — ~70% recovery |
| 32 | 0.575 | works |

Rank at full length (`artifacts/diag/14b/grid_rank/`):

| rank | acc |
|---|---|
| 64 | 0.550 |
| 32 | 0.525 |
| 16 | 0.450 |
| 8 | 0.450 |
| len64+rank16 | 0.475 |

→ Short (32–64) and low-rank (32–64) scaffolds both recover most of the benefit; degrades
gracefully. The l256 dip means the length sweep is noisy at single-seed (needs multi-seed).
A **learned** short scaffold is motivated to (a) close the ~6pt gap to real and (b) give a
reliably short/low-rank scaffold.

### Learned short scaffold (m=64) — trains, but does NOT beat the fixed full-length synth

14B MedQA, b1024, m=64 learnable KV slots distilled to full-LatentMAS Judger behavior
(`scripts/train_synth_scaffold.py`, `artifacts/ces/synth_scaffold_m64/`):

| variant | acc | tokens |
|---|---|---|
| real (n=80) | 0.600 | — |
| fixed synthglobal (full length) | 0.537–0.600 | ~700 |
| **learned m=64 slots** | **0.500** | 802 |
| none | 0.350 | (long) |

The differentiable-cache training path **works** (final loss 0.08, grad flows to slots).
The learned 64-slot scaffold recovers ~60% of the real−none gap but **underperforms the
fixed full-length synthetic scaffold** — i.e., compressing to a few learned slots loses
accuracy rather than closing the gap to real. m=32/m=16 not run (queued chain stalled; would
likely be worse). Net: the compelling "16–64-slot scaffold matches full LatentMAS" outcome
did not materialize; the strongest result remains the **fixed full-length synthetic scaffold
≈ real with zero upstream forwards** (saves compute, not KV memory).

## The positive, stated carefully

On **Qwen3-14B / MedQA**, at a constrained Judger budget, the three upstream latent agents
(33 forwards) can be replaced by **one precomputed, question-independent synthetic KV
scaffold** (Gaussian matched to average real-cache per-channel stats, full length) with **no
measurable accuracy loss** (0.600 vs 0.675, n.s.) and the same conciseness — at **zero
upstream compute**. Caveats: n=40, 1 budget, 1 task, 1 seed (hardening in flight); the effect
is largest in the constrained-budget regime (at very large budgets judger-only catches up).

## Compute framing

- Full LatentMAS upstream: **33 latent forwards** (3 agents × (K+1), K=10) + a ~1023-pos KV cache.
- `synthglobal`: **0 upstream forwards**; scaffold precomputed once. KV length still ~1023
  (so it saves *compute*, not KV *memory* — yet).

## Next directions (pick one)

**A. Harden + write the systems/mechanism paper (low risk, in progress).**
Finish n=80 + GSM8K (running), add ≥1 more seed and a hard task (AIME), measure wall-clock /
latent-forward savings, and an accuracy–token Pareto (real vs synthglobal vs none). Deliver:
*"LatentMAS's upstream agents are replaceable by a fixed synthetic KV scaffold."*

**B. Shrink the scaffold (higher-upside systems win).**
`pca32` works but `trunc64` fails → the scaffold is low-rank but needs length. Try: (i) a
**low-rank** synthetic scaffold (store rank-32 factors, not full KV) to cut memory; (ii)
**learn** a short scaffold (m≪1023 slots) end-to-end (now with a tailwind: low rank suffices)
to also cut KV memory; (iii) find the min length that still works (sweep synth_len 64→1023).
Goal: save upstream compute AND KV memory.

**C. Mechanistic depth (paper strength).**
Why does a statistical scaffold induce the concise mode? Probe the Judger's attention over
the scaffold (is it an attention-sink / effective-context-length effect?), and characterize
which layers' stats matter (ablate per-layer synth vs real).

**D. Generalize (breadth).**
Other model sizes (where's the 4B→14B robustness threshold? 8B?), other families (Llama),
other tasks (AIME, ARC), hierarchical LatentMAS. Establishes how general the phenomenon is.

**Recommendation:** A to bank the result, then B for the higher-ceiling systems story
(shrinking the scaffold is the difference between "skip the agents" and "skip the agents AND
the KV"). C strengthens whichever we write.
