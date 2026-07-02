# Results: SEAL Steering of LatentMAS for Token Efficiency

Detailed results for steering LatentMAS with a SEAL reasoning-calibration vector,
including the per-agent analysis requested by our advisor. For method/architecture
see [`EXPERIMENT_OVERVIEW.md`](EXPERIMENT_OVERVIEW.md); for the plan see
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

**TL;DR.** On a **verified-faithful LatentMAS fork**, adding a training-free SEAL
steering vector to the **Judger's** decoding cuts output tokens **~17–39% on GSM8K
with accuracy retained or improved** (and ~2× faster decoding). The vector
**transfers** (GSM8K → ARC-Challenge: −25% tokens at identical accuracy). A
per-agent study shows the effect is **specific to the Judger**: steering the
latent sub-agents (Planner/Critic/Refiner) does not reduce tokens. We show this is
**structural** (only the Judger emits text) rather than distributional (all four
agents have near-identical thought-type mixes). Latent-agent steering instead acts
as an **accuracy lever**, and role-native vectors help there.

---

## 1. Setup

| | |
|---|---|
| Backbone | Qwen3-14B (40 layers, hidden 5120) |
| System | LatentMAS, sequential (Planner → Critic → Refiner → Judger), 40 latent steps |
| Backend | HuggingFace (torch 2.8/cu128, transformers 4.57.1, vLLM 0.11), 1× H200 |
| Sampling | temperature 0.6, top-p 0.95, seed 42 |
| Eval size | n = 120 (GSM8K/ARC), n = 100 (MedQA) |
| SEAL vector | `v = mean(exec) − mean(reflection+transition)`, layer 28, from ~50 GSM8K-train traces |
| Metric | accuracy + mean output tokens (Judger), counted to EOS |

---

## 2. Fork fidelity (two independent proofs)

**Byte-for-byte:** the baseline is a verbatim fork of
[`Gen-Verse/LatentMAS`](https://github.com/Gen-Verse/LatentMAS) `@9a9e4d3`; all 24
upstream files' git blob SHAs match exactly (0 mismatches) — the authors' code
unmodified, not a re-implementation. **Behavioral:** it reproduces the paper's
accuracy on Qwen3-14B (GSM8K 92.7 vs 95.2, ARC-C 94.7 vs 95.6, MedQA 79.3 vs 80.7;
within noise for a subset/single-seed run). SEAL is layered on a separate branch so
the baseline stays provably intact.

---

## 3. Judger steering: dose-response (GSM8K, n=120)

Control = no steering (621 tokens, 93.3%).

| coef | accuracy | mean tokens | Δ tokens | sec/sample |
|---|---|---|---|---|
| 0 | 93.3% | 621.3 | — | 6.08 |
| 20 | 94.2% | 561.8 | −9.6% | 6.16 |
| 40 | **95.0%** | 514.6 | −17.2% | 4.17 |
| 60 | 94.2% | 436.9 | −29.7% | 4.55 |
| 80 | 94.2% | 379.4 | **−38.9%** | 3.16 |

Monotonic token reduction; accuracy holds or improves throughout; ~2× faster at
coef 80.

---

## 4. Cross-task transfer (GSM8K vector, Judger steering)

| Task | control | steered | Δ tokens | Δ acc |
|---|---|---|---|---|
| GSM8K | 621 tok / 93.3% | coef80: 379 / 94.2% | −38.9% | +0.9 |
| ARC-Challenge | 498 tok / 95.8% | coef40: 372 / 95.8% | −25.2% | 0.0 |
| ARC-Challenge | " | coef80: 310 / 93.3% | −37.7% | −2.5 |
| MedQA | 1003 tok / 78.0% | coef60: 880 / 73.0% | −12.2% | −5.0 |

The vector transfers cleanly to ARC-Challenge (−25% at iso-accuracy). MedQA is
**task-sensitive**: coef60 cuts tokens but costs 5 points of accuracy, indicating
medical reasoning tolerates less reflection-suppression and needs a gentler coef.

---

## 5. Per-agent steering: only the Judger matters (GSM8K, n=120)

Same GSM8K vector, applied to one agent at a time (control 621 tok / 93.3%):

| Steered agent | coef 40 | coef 80 |
|---|---|---|
| **Judger** | 515 tok / 95.0% | **379 tok / 94.2%** |
| Planner | 580 / 95.0% | 593 / 94.2% |
| Critic | 635 / 94.2% | 615 / 95.8% |
| Refiner | 623 / 91.7% | 608 / 95.0% |
| All agents (coef 60) | 455 / 93.3% | — |

Only Judger steering reduces tokens (−17 to −39%). Steering the latent sub-agents
barely changes length (they emit no text) and can slightly hurt (Refiner coef40).
Steering **all** agents (455 tok) is no better than Judger-only at the same coef
(437 tok / 94.2%) — the Judger is the locus.

---

## 6. Why (thought-type distribution) — it's structural, not distributional

We generated each role's reasoning and classified every step
(`scripts/analyze_agent_thoughts.py`, GSM8K, n=50, isolation).

![Per-agent thought-type distribution](assets/agent_thoughts_gsm8k.png)

| Role | execution | reflection | transition | non-exec |
|---|---|---|---|---|
| Planner | 87.5% | 11.4% | 1.1% | 12.5% |
| Critic | 87.8% | 8.7% | 3.6% | 12.2% |
| Refiner | 88.8% | 10.3% | 0.9% | 11.2% |
| Judger | 88.2% | 10.2% | 1.7% | 11.8% |

All four roles have **near-identical** thought-type mixes (~88% execution). So the
Judger is **not** unusually reflective — the reason SEAL only helps there is
**structural**: the Judger is the sole text-emitter, so it is the only place a
token count exists to shrink; the sub-agents run a fixed 40-step latent loop and
emit nothing. (Caveat: this is the isolation proxy — each role prompted alone; the
in-pipeline Critic, which consumes prior latent KV, may reflect more.)

---

## 7. New steering strategy for a latent agent (Phase C, GSM8K, n=120)

Since latent-agent steering can't reduce tokens, we tested whether it is instead an
**accuracy lever**, and whether a **role-native** vector (extracted from the
Critic's own reasoning) beats the generic GSM8K vector on the Critic:

| Critic vector | coef 40 | coef 80 |
|---|---|---|
| generic (GSM8K) | 635 tok / 94.2% | 615 / 95.8% |
| **critic-native** | 617 / **95.8%** | 615 / 94.2% |

Steering the Critic doesn't move tokens (~615–635, ≈ control), but a critic-native
vector improves accuracy at coef 40 (95.8% vs 94.2% generic; control 93.3%). This
supports the plan's thesis: **latent sub-agents need a different steering objective
(quality/correctness) than the token-length objective that works for the Judger.**

---

## 8. Synthesis

1. SEAL on the Judger is a clean **token-efficiency** win (−17–39%, accuracy held,
   ~2× faster), and the vector **transfers** across tasks.
2. The effect is **localized to the text-emitting agent** — shown three ways:
   per-agent steering, all-agents ≈ Judger-only, and a uniform thought
   distribution. This is a structural property of latent MAS.
3. Latent-agent steering is an **accuracy/quality lever**, not a length lever, and
   benefits from **role-native** vectors — a concrete direction for a novel
   contribution beyond "SEAL + LatentMAS."

## 9. Limitations

- n = 100–120, single seed; a larger n + multiple seeds are needed for tight CIs.
- Single layer (28) and one backbone (Qwen3-14B).
- Thought-distribution uses an isolation proxy (roles prompted independently), not
  in-pipeline latent-debug decoding.
- MedQA shows the token/accuracy trade-off is task- and coef-dependent.

## 10. Next steps

- Firm up headline with n≈500, 3 seeds, and a full accuracy–token Pareto frontier.
- Gentler coef sweep for MedQA; layer sweep for the Judger.
- Develop the latent-agent **quality** steering (correct-vs-incorrect contrastive
  vectors; role-native divisions) into the paper's novel method.
