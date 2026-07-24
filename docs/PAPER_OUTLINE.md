# Paper Outline — What Latent Multi-Agent Communication Actually Does

Updated: 2026-07-21. A concrete paper skeleton grounded in the experiments recorded in
[`EXPERIMENTS_UNIFIED_RESULTS.md`](EXPERIMENTS_UNIFIED_RESULTS.md),
[`DIAG_READOUT_FINDINGS.md`](DIAG_READOUT_FINDINGS.md), and
[`RESULTS.md`](RESULTS.md).

## Working title

**"Steer the Reader, Not the Memory — and the Memory Barely Speaks: What Latent
Inter-Agent Communication Actually Does in Multi-Agent LLMs."**

(Alt: "Latent Multi-Agent Communication is a Conciseness Scaffold, Not an Information
Channel.")

## Thesis

In LatentMAS-style systems, the upstream agents' latent KV cache is **not a
task-information channel**. The final (text-emitting) agent does not read its content:
a wrong problem's cache helps as much as the correct one, the target answer is not
decodable from the latent states, and every attempt to steer the latent content (three
independent methods) fails to improve accuracy. What the channel *does* provide is a
**content-independent scaffold** that makes the decoder more concise (~40% fewer tokens)
and, under limited budgets, more accurate. The usable control surface is the decoder,
not the latent channel.

## Claims & evidence (all in-repo)

1. **Steering latent content does not improve accuracy** — three independent methods:
   - training-free diff-of-means residual (native, in-pipeline): within noise, hurts
     MedQA (`RESULTS.md` §9).
   - KV-cache steering of the handoff: falsified; content columns saturate/corrupt
     (`KV_STEERING_FINDINGS.md`).
   - learned CES vector (optimized end-to-end through the differentiable KV): inert
     (last-token) or destructive-only (all-token α<0 → 0), across strengths/signs/2 configs
     (`CES_LATENT_STEERING_FINDINGS.md`).
2. **The decoder ignores the cache's content** — causal cache-swap: real ≈ shuffled
   (wrong-question) at every budget, on **MedQA (4B+14B) and GSM8K (14B)**
   (`DIAG_READOUT_FINDINGS.md`).
3. **The answer is barely encoded upstream** — per-agent/layer linear+MLP probes for the
   gold answer are near chance (MedQA 4B & 14B); the correctness probe is also near
   chance upstream vs decodable at the decoder (0.69–0.80) (`RESULTS.md` §9.1).
4. **What the channel does: content-independent conciseness / scaffolding** — any
   non-zero latent prefix cuts decoder tokens ~40% at iso-accuracy; zeroed cache ≈ none.
   On MedQA the accuracy edge over decoder-only is largely a token-budget artifact
   (collapses by 4096); on GSM8K it is a genuine but content-independent +0.13 boost.
5. **Where steering DOES work: the decoder** — SEAL residual & KV `judger_token` steering
   cut decoder tokens −17 to −39% at retained accuracy, and transfer across tasks
   (`RESULTS.md` §3–4).

## Framing vs related work

- Activation steering (CAA, ActAdd, ITI, RepE, SEAL, ASC) steers *token generation*; we
  study steering a *non-text latent inter-agent channel* and show it is not steerable for
  reasoning quality — a negative that maps the limits of steering to the text boundary.
- KV-cache steering (arXiv:2507.08799): we corroborate that leverage is at aggregation
  positions, not dense content, and extend it to the multi-agent latent handoff.
- LatentMAS (arXiv:2511.20639): we provide a mechanistic account of *why* it helps —
  conciseness scaffolding + budget effects — rather than latent information transfer.

## Contributions

1. A controlled, causal cache-intervention protocol (real/shuffled/zero/none × token
   budget) that separates *content* from *presence* effects in latent MAS.
2. Evidence, across 2 models and 2 tasks and 3 steering methods, that latent inter-agent
   content is not used and not steerable for accuracy.
3. A reframing of latent MAS as a conciseness scaffold, with an accuracy–token analysis.
4. A practical takeaway: put steering/efficiency effort at the decoder.

## Solid vs. reviewer-risk

- Solid: multi-method convergence, causal cache swaps, cross-model + cross-task
  replication, probes, controls (α-sign, random-v, zero, shuffled), paired CIs.
- Risk: headline is a negative/mechanistic result; n=40–60 single-seed for diagnostics;
  MedQA + GSM8K only; the +0.075 real−shuffled hint is underpowered (not claimed).

## Remaining experiments to harden (in priority order)

1. Multi-seed + larger n on the cache-swap (tighten real vs shuffled; currently n=40–60).
2. A hard task (AIME) where the decoder cannot re-solve — strongest token-budget crossover.
3. Accuracy–token Pareto: {decoder-only, full, shuffled} across budgets, both models —
   packages the conciseness/efficiency positive cleanly.
4. Optional: does the scaffold generalize to other latent-MAS variants / hierarchical
   prompt.

## Honest venue read

Workshop / ACL-Findings tier as-is (rigorous mechanistic negative + a clean efficiency
reframing). Main-track would need either a *positive* mechanism that makes the latent
channel carry usable info (e.g., a learned bridge/aggregation token trained to be
decodable — see `LATENT_STEERING_RESEARCH_HANDOFF.md` §6) or a substantially broader,
more surprising characterization.
