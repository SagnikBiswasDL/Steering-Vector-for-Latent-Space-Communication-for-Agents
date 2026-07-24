# CES Latent-Agent Steering — Findings (boost-at-best-K)

Updated: 2026-07-19

## Verdict: **NEGATIVE for improvement** (well-controlled)

A learned CES (contrastive energy) steering vector applied to the LatentMAS **latent
agents** (Planner / Critic / Refiner, during their recurrent latent steps) **does not
improve** MedQA accuracy on Qwen3-4B at the best operating point (`K=10`). The
intervention can *degrade* the pipeline (a large negative all-token perturbation drives
accuracy to 0 by making the Judger degenerate), but there is **no direction or strength
that lifts accuracy above the unsteered baseline**. This matches the prior linear-probe
result (correctness is barely encoded in upstream latents) and the training-free
negative in [`RESULTS.md`](RESULTS.md) §9.

This is the answer to the research question *"can steering the latent agents meaningfully
alter (improve) performance?"* → **No, on this setup. It can only disrupt.**

---

## Setup

- Model: **Qwen3-4B** (hidden 2560, 36 layers), HF backend, greedy (`temp=0`), seed 42.
- Method: LatentMAS, `K=10` per upstream agent (the best unsteered point from Gate 1).
- Steering: one residual vector `v` at **layer 20**, added during **latent steps only**
  (`steer_phase=latent_only`) on `{planner, critic, refiner}`; Judger not steered.
- Objective: CES ranking loss `softplus(E(y+) − E(y−))`, `y+`=gold letter,
  `y-`=model's mined wrong prediction (distractor fallback). `E` = length-normalized
  Judger NLL under the steered upstream KV (`teacher_force_nll`).
- Training pairs: **single-K correctness mining** on the MedQA **train** split
  `[100,220)` (`scripts/mine_budget_pairs.py --single_k 10`), 120 items.
- Evaluation: `--mode boost` with causal controls (α=0 sanity, α<0 negate, random-v),
  paired bootstrap CIs; go/no-go on the **dev** split so **test `[0,100)` stays untouched**.
- Two injection sites tested: `apply_to=last` (last latent token only) and
  `apply_to=all` (every latent token position).

Code: `scripts/mine_budget_pairs.py` (`--single_k`), `scripts/train_ces_claim_b.py`
(`--apply_to`, `--grad_clip`, `--max_v_norm`), `scripts/eval_ces_heldout.py`
(`--mode boost --control`), `scripts/coef_sweep.py`, drivers
`scripts/run_boost_experiment.sh`, `scripts/run_all_tokens.sh`.

---

## Results

### 1. Dev go/no-go (last-token), n=80, unsteered = 0.650

| Config | unsteered | steered α>0 | negate α<0 | decision |
|---|---|---|---|---|
| LR 1e-2, no clip | 0.650 | 0.662 | 0.650 | NULL, paired CI [−0.075, +0.10] |
| LR 5e-3, grad-clip 1.0 | 0.650 | 0.662 | 0.650 | NULL (same) |

Both configs land on the same +1/80 flip. Grad clipping fixed the *optimization*
(`v_norm` stabilized ~4.95 instead of exploding; raw grads had spiked to 1e4–1e5) but
did **not** change the outcome.

### 2. Coef sweep — last-token vector, dev n=40, ‖v‖≈4.95, unsteered = 0.650

| α | acc | Δ | 95% CI |
|---|---|---|---|
| +4 | 0.600 | −0.05 | [−0.15, +0.05] |
| +8 | 0.650 | 0.00 | [−0.125, +0.125] |
| +16 | 0.600 | −0.05 | [−0.15, +0.05] |
| −8 | 0.600 | −0.05 | [−0.15, +0.05] |

Inert even at α=16 (delta norm ~80). No dose-response, no polarity.

### 3. Coef sweep — all-tokens vector, dev n=40, ‖v‖≈4.99, unsteered = 0.650

| α | acc | Δ | 95% CI | mean tokens |
|---|---|---|---|---|
| +1 | 0.600 | −0.05 | [−0.15, +0.05] | 962 |
| +2 | 0.675 | +0.025 | [−0.075, +0.15] | 1018 |
| +4 | 0.625 | −0.025 | [−0.125, +0.075] | 969 |
| +8 | 0.600 | −0.05 | [−0.15, +0.05] | 1005 |
| **−8** | **0.000** | **−0.65** | **[−0.775, −0.50]** | **1946** |

All-tokens training was **well-conditioned** (grad norms ~15, vs 1e4–1e5 for
last-token). Positive strengths stay within noise of baseline; the **negative
direction at α=−8 collapses accuracy to 0** — and the mean output nearly doubles
(1946 ≈ the 2048 cap), i.e. the Judger **degenerates / rambles** rather than steering
to a specific wrong answer.

---

## Interpretation

1. **Last-token steering is inert** — the last latent token's residual at layer 20 has
   negligible causal leverage over the Judger (whose KV is dominated by hundreds of
   prefill tokens vs 10 latent tokens/agent).
2. **All-tokens steering has real leverage but only destructive** — perturbing every
   latent position *can* break the pipeline (α=−8 → 0%), proving the channel matters,
   yet **no positive direction improves accuracy**.
3. **Mechanism**: consistent with the linear probe in `RESULTS.md` §9.1 — correctness
   is barely linearly encoded in upstream latents (AUC 0.46–0.64) but is readable at the
   Judger (0.69–0.80). CES can therefore find *disruptive* directions but not an
   *accuracy-improving* one; the signal to steer toward simply isn't in the upstream
   latent stream.

---

## Relation to the rest of the project

- **Judger-side SEAL (positive):** steering the text-emitting Judger cuts output tokens
  −17% to −39% at retained accuracy (`RESULTS.md`) — a real efficiency win.
- **Training-free upstream steering (negative):** native diff-of-means vectors on
  P/C/R don't beat Judger-only SEAL (`RESULTS.md` §9).
- **Learned CES upstream steering (this memo, negative):** the *stronger* test —
  optimized end-to-end through the differentiable KV — also fails to improve, and adds
  the new fact that the only reachable effect is destructive.

Coherent paper framing: *activation steering helps the text-emitting Judger's
efficiency, but the non-text latent inter-agent channel is not a usable lever for
improving reasoning accuracy — it can only disrupt — with a probe explaining why.*

---

## Limitations

- Qwen3-4B only; MedQA only; layer 20 only; single seed (42).
- Boost dev/coef-sweep on the **dev** split (n=40–80); the frozen **test** `[0,100)`
  was never consumed (go/no-go was NULL, so held-out was correctly skipped).
- Dev baseline (~0.65) is lower than the Gate-1 **test** accuracy (0.83, n=100) — small
  harder subset; the *within-config* steered-vs-unsteered comparison is the valid signal.

## Suggested next steps (if continuing)

1. **14B confirmation** (layer 28) to rule out model-scale — likely same verdict.
2. **Second task** (GSM8K) for generality.
3. If still pursuing a positive: the probe implicates a *non-linear / late* correctness
   signal, so an improving intervention would likely need to act on the **KV the Judger
   reads** (not the upstream latent loops) — but static handoff steering was already
   falsified in `docs/KV_STEERING_FINDINGS.md`.
