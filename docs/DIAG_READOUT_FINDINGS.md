# Latent Readout Diagnostics — Findings

Updated: 2026-07-21. Tests the "steer the reader, not the memory" hypothesis by asking
(1) does the upstream latent cache contain answer info, and (2) does the Judger use it.
Scripts: `scripts/diag_cache_usage.py`, `scripts/diag_answer_probes.py`,
`scripts/run_diagnostics.sh`. Models: **Qwen3-4B and Qwen3-14B** (findings replicate on
both), MedQA, K=10, greedy (temp=0). Artifacts: `artifacts/diag/{4b,14b}/`.

## Headline

**The LatentMAS upstream latent channel is not an information channel; it is a
content-independent conciseness scaffold, and most of its apparent benefit is a
Judger token-budget artifact. Confirmed on both 4B and 14B (14B = the model where
LatentMAS was validated to help).**

- Content is ~irrelevant: a wrong-question cache helps ~as much as the correct one.
- Answer info is barely decodable upstream (near chance).
- The real benefit vs judger-only shrinks to non-significant once the Judger has room
  to re-derive the answer itself.

This falsifies the readout-bottleneck premise (info present but unread): there is little
answer content to read.

## 1. Does the Judger USE the cache? (cache-usage, test, n=40–60)

Per item: build the real upstream cache (P->C->R at K=10), then decode the Judger under
{real, shuffled(cross-question), zero, none} x Judger token budget. Greedy. idx-aligned
paired bootstrap CIs.

| budget | real | shuffled | none | real−shuffled | real−none | tokens real/shuf/none |
|---|---|---|---|---|---|---|
| 1024 | 0.450 | 0.450 | 0.150 | 0.00 [−.15,.15] | +0.30 [.15,.45] | 822 / 797 / 983 |
| 2048 | 0.650 | 0.575 | 0.550 | +0.075 [.00,.15] | +0.10 [.00,.22] | 1000 / 934 / 1476 |
| 4096 | 0.675 | 0.600 | 0.625 | +0.075 [−.025,.175] | +0.05 [−.05,.15] | 1062 / 997 / 1750 |

(n=60 run at budget 1024 also had zero=0.217 @ 961 tok ~= none, i.e. zeroed content
behaves like no cache; only non-zero content gives the conciseness effect.)

**Reads:**
1. **Token-budget artifact.** `real − none` collapses +0.30 → +0.10 → +0.05 as budget
   grows. At 4096 judger-only (0.625) ≈ full LatentMAS (0.675) — with room to think the
   Judger re-solves the task. Explains the prior role-ablation (judger-only ≈ full).
2. **Content ~irrelevant.** `real − shuffled` = 0.00 → +0.075 → +0.075 with CIs
   touching/crossing 0; a wrong-question cache is ~as useful as the correct one.
3. **Conciseness scaffold.** Any non-zero latent prefix makes the Judger conclude sooner
   (~1000–1060 tok) vs judger-only (~1476–1750 tok), at comparable accuracy — a ~40%
   token reduction that is content-independent (shuffled does it too).

**14B replication (test, n=40)** — same picture on the stronger model:

| budget | real | shuffled | none | real−shuffled | real−none | tokens real/shuf/none |
|---|---|---|---|---|---|---|
| 1024 | 0.675 | 0.625 | 0.350 | +0.05 | +0.325 | 722 / 712 / 916 |
| 2048 | 0.775 | 0.775 | 0.700 | 0.00 | +0.075 | 882 / 867 / 1329 |
| 4096 | 0.800 | 0.775 | 0.750 | +0.025 | +0.05 | 892 / 878 / 1603 |

real ≈ shuffled at all budgets; real−none collapses +0.325 → +0.075 → +0.05 (none 0.75 ≈
real 0.80 at 4096); conciseness scaffold ~890 tok vs 1603 for judger-only (~45% fewer).
14B answer probes also near chance (planner/critic/refiner MLP 0.32–0.33, AUC ~0.50).

**GSM8K 14B (test, n=60)** — second task, open-ended numeric (not multiple-choice):

| budget | real | shuffled | none | real−shuffled | real−none | tokens real/shuf/none |
|---|---|---|---|---|---|---|
| 256 | 0.233 | 0.200 | 0.250 | +0.03 | −0.02 | 254 / 255 / 256 |
| 512 | 0.567 | 0.550 | 0.467 | +0.02 | +0.10 | 444 / 451 / 477 |
| 1024 | 0.933 | 0.933 | 0.800 | 0.00 | +0.13 | 542 / 560 / 668 |

Same signature: **real ≈ shuffled** at every budget (content irrelevant). Note at 1024
the real/shuffled runs are *not* token-starved (finish at ~542–560 < 1024) yet still beat
judger-only by +0.13 — and none is also not capped (668) — so here the scaffold gives a
genuine, **content-independent accuracy boost** (a wrong problem's cache helps as much),
with the usual conciseness signature (542/560 tok vs 668).

## 2. Does the cache CONTAIN the answer? (answer probes, train, n=200)

Per-agent per-layer final latent residual -> linear & MLP probe for the gold 4-way
answer (chance 0.25). Both probes weak:

| agent | best layer | linear acc | MLP acc | AUC |
|---|---|---|---|---|
| planner | 32 | 0.325 | 0.333 | ~0.50 |
| critic | 14 | 0.258 | 0.317 | ~0.50 |
| refiner | 22 | 0.267 | 0.283 | ~0.49 |

MLP ≈ linear and both near chance -> the gold answer is barely encoded upstream (linear
or nonlinear). Raw features saved (`artifacts/diag/4b/answer_probes/features.npz`) for
probing K/V-pooled or model-prediction labels later.

## 3. Decision-tree outcome

Per the "steer the reader" plan's tree:
- probes weak (answer not decodable) AND shuffling barely matters -> **NOT** a clean
  readout bottleneck (which needs info-present-but-unused). Closest to "cache contains
  ~no usable answer info; benefit is a length/scaffolding effect."
- Therefore head-wise readout gates aimed at **accuracy** are not motivated: little
  content to amplify, and the Judger already re-solves when unconstrained.

## 4. Surviving threads

1. **Weak content signal:** `real − shuffled ≈ +0.075` at budget ≥ 2048 (underpowered,
   n=40, CIs include 0). Only manifests with room to reason. Worth a larger-n / multi-seed
   / 14B test before fully dismissing "content matters a little."
2. **Efficiency reframing (most promising positive):** the latent agents act as a
   content-independent conciseness regularizer (~40% fewer Judger tokens at iso-accuracy
   vs judger-only at 4096). A real mechanistic finding; distinct from SEAL (which steers
   the Judger's own residual). Could be quantified across tasks/models and paired with a
   token–accuracy frontier.

## 4b. Scaffold mechanism sweep — WHAT reproduces the effect? (4B MedQA, budget 1024, n=40)

Replace the Judger's cache with variants; measure acc (real=0.450, none=0.150 baselines).
Script: `scripts/diag_scaffold_sweep.py`; artifacts `artifacts/diag/4b/scaffold_sweep/`.

| variant | acc | vs real | note |
|---|---|---|---|
| real | 0.450 | — | upper baseline |
| **shuffled** (wrong-question real cache) | **0.450** | +0.00 | reproduces (content-independent) |
| none | 0.150 | −0.30 | lower baseline |
| gauss (N(0,1)) | 0.125 | −0.325 | fails |
| **matched** (per-channel mean/std noise, full length) | 0.050 | −0.40 | **fails** → not coarse statistics |
| **repeat_last** (one latent vector ×all positions) | 0.075 | −0.375 | **fails** → not a single vector |
| trunc16 / trunc4 (last m real positions) | 0.075 / 0.125 | −0.375 / −0.325 | fails → needs many positions |
| pca8 (rank-8 recon) | 0.050 (17 tok) | −0.40 | fails → not low-rank |

**Result:** only a **genuine, full-length, on-manifold model-generated cache** reproduces
the scaffold (real AND shuffled = 0.45). Distribution-matched noise, a single repeated
vector, truncation to 4–16 positions, and rank-8 PCA all collapse to ≈none or worse. So
the effect is **content-independent but structurally demanding** (full-length, high-rank,
on-manifold) — it is NOT a coarse-statistics or low-dimensional object.

**Threshold sweep (same setup; `artifacts/diag/4b/scaffold_thresh/`):**

| variant | acc | vs real | note |
|---|---|---|---|
| pca16 / pca32 / pca64 | 0.025 / 0.000 / 0.000 | −0.43 / −0.45 / −0.45 | low-rank fails at every rank (degenerate at low r) |
| trunc32 / trunc64 / trunc256 | 0.175 / 0.175 / 0.075 | ≈ none | truncation fails at every length up to 256 |

**Combined conclusion:** the scaffold effect is **content-independent but not compressible**.
Across 12 variants, only a full on-manifold model cache (real or shuffled) reproduces it;
matched statistics, a single repeated vector, low-rank PCA (≤64), and truncation (≤256 of
~1000 positions) all collapse to ≈none. The effect lives in the full, high-rank,
full-length structure of a genuine model-generated cache.

**14B mechanism sweep (test, n=40; real=0.675, none=0.350) — FLIPS the 4B conclusion:**

| variant | acc | vs real | 4B |
|---|---|---|---|
| shuffled | 0.625 | −0.05 (n.s.) | reproduces |
| **matched** (per-channel mean/std noise, full length) | 0.600 | −0.075 (n.s.) | **failed on 4B, works on 14B** |
| **pca32** (rank-32 recon) | 0.650 | −0.025 (n.s.) | **failed on 4B, works on 14B** |
| repeat_last | 0.225 | −0.45 | fails |
| trunc64 | 0.400 | −0.275 | fails (length matters) |

**Model-size effect:** 4B is brittle to off-manifold caches (matched/PCA degenerate);
14B is robust enough that **coarse per-channel statistics at full length reproduce the
scaffold**. On the capable model the effect is NOT the fine semantic structure — it's
distribution-level statistics over a long context (single vector and truncation still fail).

**Decision-tree impact (revised):** on 14B this is the "**distribution-matched random cache
works → build a fixed universal synthetic scaffold**" branch.

### POSITIVE: a fixed universal synthetic scaffold works (14B MedQA, budget 1024, n=40)

`synthglobal` = one fixed synthetic KV cache, question-independent, built from the averaged
per-(layer,head,channel) mean/std of 8 **train** caches (len 1023), Gaussian-sampled — **no
upstream forward passes at all**. Compared to real (full LatentMAS, 33 latent fwds) and none.

| variant | acc | tokens | vs real |
|---|---|---|---|
| real | 0.675 | — | — |
| shuffled | 0.625 | 712 | −0.05 (n.s.) |
| matched (per-item stat noise) | 0.600 | 712 | −0.075 (n.s.) |
| **synthglobal** (fixed, question-independent) | **0.600** | **701** | **−0.075 (n.s.)** |
| none (judger-only) | 0.350 | (long) | −0.325 |

**Result:** a precomputed synthetic scaffold reproduces the full pipeline's accuracy
(0.600 vs 0.675, CI includes 0) and conciseness (701 tok), and beats judger-only by +0.25,
with **zero upstream compute**. So on 14B the 3 latent agents can be replaced by a fixed
statistical scaffold at the constrained-budget operating point. (n=40, budget 1024, one
task/seed — hardening in progress: larger n, budgets {1024,2048,4096}, GSM8K.)
Script: `scripts/diag_scaffold_sweep.py --variants ...,synthglobal`; artifacts
`artifacts/diag/14b/synth_scaffold/`.

## 5. Caveats / next

- 4B only; MedQA only; single seed; greedy. The content question (real vs shuffled) on
  **14B** (the system where LatentMAS was validated to help) is the key generalization.
- Larger n + multi-seed to resolve the +0.075 content hint at budget ≥ 2048.
- The token-budget crossover (none catches up by 4096) should be shown on ≥1 more task.
- If pursuing a mechanism: given content is ~irrelevant, an accuracy win likely requires
  first making the channel carry/commit answer-relevant info (e.g., a learned bridge/
  aggregation token trained to be decodable), not steering the readout of content that
  isn't there.
