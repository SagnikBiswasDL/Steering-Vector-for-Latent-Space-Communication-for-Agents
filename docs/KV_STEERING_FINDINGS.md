# KV Cache Steering on LatentMAS — Findings

Adaptation of *KV Cache Steering for Controlling Frozen LLMs* (arXiv:2507.08799)
to LatentMAS. One-shot edit `K += c_k·S_k`, `V += c_v·S_v` applied at all layers
to the key/value cache the Judger consumes, then normal decoding (no per-step
hook). Steering direction `S` built paper-faithfully from 200 GSM8K-train
contrastive pairs (CoT-rich vs answer-only ICL), value-only (`c_k = 0`).

- Model: `Qwen/Qwen3-14B` · Task: GSM8K test, n=150 · Backend: HF, 1x H200
- Baseline (control): **94.7% accuracy, 587.7 Judger output tokens**
- Vector artifact: `artifacts/kv_steer_vectors/qwen3-14b/gsm8k_cot.pt`
  (40 layers x 8 KV heads x 128 dim; mean value-direction norm ~12.3)
- Raw results: `artifacts/sweeps/kv_*.csv`; plot: `artifacts/plots/kv_gsm8k.png`

## Application surfaces tested
- `handoff_last` — the single last column of the latent handoff.
- `handoff_all` — every column of the latent handoff (Planner/Critic/Refiner KV).
- `judger_token` — the Judger's own final prompt token (paper's target; control arm).

## Results (n=150, GSM8K)

handoff_last (steer 1 of ~130 handoff columns)

| c_v | acc | tokens |
|---|---|---|
| +8 | 91.3% | 590 |
| +12 | 92.0% | 610 |
| -8 | 93.3% | 604 |

handoff_all (steer the whole handoff)

| c_v | acc | tokens |
|---|---|---|
| 0.05 | 92.0% | 569 |
| 0.1 | 94.0% | 584 |
| 0.2 | 92.7% | 503 |
| -0.1 | 93.3% | 636 |
| 1.0 | 72.0% | 894 |

judger_token (steer the Judger's own aggregation token)

| c_v | acc | tokens | Δtokens |
|---|---|---|---|
| +8 | 94.0% | 440 | **-25%** |
| +12 | 93.3% | 392 | **-33%** |
| -8 | 93.3% | 622 | +6% |

## Statistical read
At n=150 and ~94% accuracy the noise band is ~+/-3-4 points. Every accuracy
number except the `handoff_all cv=1` crash is within noise of control. What is
clearly real: the `judger_token` token reductions (-25% / -33%), the
`handoff_all cv=1` collapse, and the sign consistency everywhere (`+c_v` reduces
tokens, `-c_v` increases them).

## Conclusions
1. **Falsified:** steering the latent handoff does not make the upstream agents
   useful. No regime helps — tiny `c_v` is neutral (within noise), meaningful
   `c_v` is destructive (`cv=1` collapses accuracy 94.7% -> 72% and inflates
   tokens).
2. **Mechanism (generalizes):** steering *all* handoff columns shifts the
   Judger's attention output by the full `c_v·S_v` per layer (attention weights
   sum to 1), so it saturates into oversteering almost immediately; a single
   dense column is ~1/130 of attention, so it is too weak to matter. The KV cache
   has two functionally different position types — dense information-bearing
   positions (latent thoughts) that steering *corrupts*, and aggregation/routing
   positions (the last prompt token) that steering *rides*.
3. **Validated:** cache steering works at the paper's intended surface
   (`judger_token`): a free, one-shot, zero-latency **-25% to -33% output tokens
   at statistically-unchanged accuracy**. The extraction produces a real,
   sign-consistent reasoning-verbosity axis in KV space.

## Implication
Second independent method (after residual-stream SEAL on the latent agents) to
find that leverage is concentrated at the text-emitting Judger, not the upstream
latent agents. Motivates the agent-ablation study: do Planner/Critic/Refiner
provide causal value at all?

## Caveats
One task/seed/backbone; direction extracted from text CoT but applied to latent
KV (modality gap); repo's batched decode uses right padding (pre-existing,
identical across arms); `c_k = 0` throughout.
