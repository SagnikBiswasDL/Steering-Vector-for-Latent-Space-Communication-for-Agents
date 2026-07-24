# Gate 1 status (K-budget CES)

Updated: 2026-07-15 (final)

## Verdict: **STOP K-budget CES** (Gate 1 no-go)

Both non-truncated curves are done. There is **no operating point** where a smaller `K` loses accuracy *and* saves ≥15% wall-clock vs a larger `K`. Latency is dominated by Judger CoT length, which **falls** as `K` rises (better latent context → shorter text). Cutting `K` makes the system *slower*, not faster.

Gate 2 (grad smoke) already passed; do **not** scale CES training for Option 1. Pivot candidates: Judger ASC / token compression, or other axes.

---

## Gate 2 (passed)

Qwen3-4B, `K_small=5`, layer 20, latent-only:
- only `v` trainable; nonzero grads; zero-vector allclose to baseline
- artifact: `artifacts/ces/gate2_smoke.json`

---

## Gate 1 probe A — AIME 2024 @ `max_new_tokens=2048` (**invalid**)

All runs hit the token cap; do not use for selection. Artifacts: `artifacts/gate1/aime2024_probe/`

---

## Gate 1 primary — MedQA n=100 @ 4096 (valid)

Qwen3-14B, `generate_bs=1`, seed 42. Zero failures / no-answer.

| K | Acc (CI) | Latency (s) | Output tokens | Latent forwards |
|---|----------|-------------|---------------|-----------------|
| 0 | 0.78 [0.69, 0.86] | **40.6** | 1731 | 3 |
| 5 | 0.77 [0.68, 0.85] | 26.2 | 1097 | 18 |
| **10** | **0.83** [0.76, 0.90] | **22.9** | **944** | 33 |
| 20 | 0.79 [0.71, 0.87] | 24.9 | 992 | 63 |
| 40 | 0.79 [0.71, 0.86] | 25.8 | 967 | 123 |

- Best accuracy **and** best latency at **K=10** (not K=40).
- Paired vs max-K: no credible positive gap favoring larger K.
- K=10 vs K=5: +6 pts paired (CI [0.02, 0.11], credible) but latency goes the *wrong* way for “recover accuracy at smaller K with compute savings” (−14% when moving small→full means full is faster).
- Script `selected=false`.

Artifacts: `artifacts/gate1/medqa_n100/`

---

## Gate 1 probe — AIME 2024 @ 8192 (valid; n=30)

| K | Acc (CI) | Latency (s) | Output tokens | Latent forwards |
|---|----------|-------------|---------------|-----------------|
| 0 | 0.53 [0.37, 0.70] | **178** | 7237 | 3 |
| **10** | **0.67** [0.50, 0.83] | **146** | **5927** | 33 |
| 40 | 0.50 [0.33, 0.67] | 159 | 6401 | 123 |

- Credible paired gap: K=10 − K=0 ≈ **+13 pts** (CI [0.03, 0.27]).
- But K=0 is **~21% slower** than K=10 (Judger tokens 7237 vs 5927). No latency win from shrinking K.
- K=40 underperforms K=10 on accuracy (wide CIs).
- Script `selected=false`.

Artifacts: `artifacts/gate1/aime2024_tok8192/`

---

## Decision rule check

| Criterion | MedQA | AIME@8192 |
|-----------|-------|-----------|
| Credible accuracy(K_full) > accuracy(K_small) | Weak / wrong direction for “full=large K” | Yes for K_full=10, K_small=0 |
| ≥15% wall-clock reduction at K_small | **No** (K_small slower) | **No** (K_small slower) |
| Proceed to pair mining / CES train | **No** | **No** |

**Root cause for the compute claim:** wall-clock is Judger-token–dominated. Extra latent steps often *shorten* Judger generation enough to more than pay for themselves. Primary metric therefore does not support “smaller K = cheaper.”

---

## Next (pivot)

1. Document Option-1 stop in notes / advisor slides if needed.
2. Do **not** start CES pair mining for K-budget recovery.
3. Preferred pivots: Judger-side ASC / SEAL-style CoT compression (existing strength), or reframe compute in pure latent-forward / KV-byte terms *without* claiming wall-clock wins from cutting K.
4. Pods idle — safe to stop both GPUs when ready.
