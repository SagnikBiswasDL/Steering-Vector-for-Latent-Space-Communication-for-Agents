# Unified Experiment Results — Steering LatentMAS

Updated: 2026-07-21 (rev 3: readout diagnostics now cross-model (4B+14B) and cross-task
(MedQA+GSM8K); refreshed state-of-play + progress plan). Single-file record of every
steering + diagnostic experiment run on this project, with exact configs and numbers.
Companion docs: [`DIAG_READOUT_FINDINGS.md`](DIAG_READOUT_FINDINGS.md) (diagnostics),
[`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) (write-up skeleton),
[`LATENT_STEERING_RESEARCH_HANDOFF.md`](LATENT_STEERING_RESEARCH_HANDOFF.md) (mechanism ideas).

## Common setup

- **System:** LatentMAS, sequential Planner → Critic → Refiner → Judger. Upstream 3 agents
  reason in latent space (`K` = `latent_steps` forward passes each, no text); only the
  Judger decodes. Total latent forwards ≈ `3·(K+1)`.
- **Backbones:** Qwen3-14B (hidden 5120, 40 layers) for §1–§6, §8; Qwen3-4B (hidden 2560,
  36 layers) for §7 (learned CES). Vectors do not transfer across sizes.
- **Backend/env:** HuggingFace, torch 2.8+cu128, transformers 4.57.1, vLLM 0.11, 1× H200.
- **Fork fidelity:** byte-for-byte fork of `Gen-Verse/LatentMAS@9a9e4d3` (24/24 file SHAs
  match); reproduces paper accuracy. Steering layered as additions.
- Unless noted: SEAL/native at layer 28 (14B); CES at layer 20 (4B). Sampling: temp 0.6 /
  top-p 0.95 for §1–§6,§8 (seed 42); greedy (temp 0) for §7.

## Master summary

| § | Experiment | Model / Task | Headline result | Verdict |
|---|---|---|---|---|
| 1 | Unsteered K-curve (Gate 1) | 14B / MedQA, AIME | best at **K=10** (0.83 MedQA); smaller K ≠ cheaper (Judger len dominates) | K-budget latency framing **dead** |
| 2 | SEAL residual @ Judger | 14B / GSM8K | −17 to −39% tokens, acc held/improved, ~2× faster | **POSITIVE** |
| 3 | SEAL transfer | 14B / ARC, MedQA | ARC −25% @ iso-acc; MedQA task-sensitive (−5 pts) | Positive (task-sensitive) |
| 4 | Per-agent SEAL | 14B / GSM8K | only Judger cuts tokens; all-agents ≈ Judger-only | leverage at Judger |
| 5 | Native correctness-contrastive (upstream) | 14B / GSM8K,ARC,MedQA | within noise; hurts MedQA; no token savings | **NEGATIVE** |
| 5b | Correctness probe (mechanism) | 14B | upstream AUC 0.46–0.64; **Judger 0.69–0.80** | signal decided late |
| 6 | KV-cache steering | 14B / GSM8K | handoff falsified; `judger_token` −25 to −33% tokens | latent handoff **falsified** |
| 7 | Learned CES "boost" | 4B / MedQA | steered ≤ noise (all positive α); all-token α=−8 → 0 | **NEGATIVE (destructive-only)** |
| 8 | Role ablation | 14B / GSM8K, MedQA | judger-only ≈ full pipeline (± noise) | upstream low causal value |
| 9 | **Readout diagnostics** (cache-usage + answer probes) | **4B+14B / MedQA; 14B / GSM8K** | real≈shuffled (content ignored) on both tasks/models; real−none is a token-budget artifact / content-independent boost; gold barely decodable upstream | **latent = conciseness scaffold, not info channel (cross-model, cross-task)** |
| 10 | **Synthetic latent scaffold** (mechanism sweep + fixed synth + shrink + learned) | 4B+14B / MedQA; 14B / GSM8K | 14B: a FIXED question-independent synthetic KV cache (0 upstream forwards) ≈ real acc + conciseness; learned/short versions recover most but not all | **POSITIVE: replace upstream agents with a synthetic scaffold (saves compute)** |

---

## 1. Unsteered K-curve (Gate 1) — `GATE1_STATUS.md`

**MedQA, 14B, n=100, max_new_tokens=4096, seed 42:**

| K | Acc (95% CI) | Latency (s) | Out tokens | Latent fwds |
|---|---|---|---|---|
| 0 | 0.78 [0.69, 0.86] | 40.6 | 1731 | 3 |
| 5 | 0.77 [0.68, 0.85] | 26.2 | 1097 | 18 |
| **10** | **0.83 [0.76, 0.90]** | **22.9** | **944** | 33 |
| 20 | 0.79 [0.71, 0.87] | 24.9 | 992 | 63 |
| 40 | 0.79 [0.71, 0.86] | 25.8 | 967 | 123 |

**AIME 2024, 14B, n=30, max_new_tokens=8192:** K0 0.53 [0.37,0.70] / 178s / 7237 tok;
**K10 0.67 [0.50,0.83] / 146s / 5927 tok**; K40 0.50 [0.33,0.67] / 159s / 6401 tok.
Paired K10−K0 = +0.133 [0.033, 0.267] (credible).

**Takeaway:** best accuracy at K=10, not K=40. Cutting K does **not** save wall-clock —
Judger CoT length dominates and *falls* as K rises. The "small-K + steering to recover"
compute story is dead. Also: Judger is largely insensitive to K beyond 10.

---

## 2. SEAL residual steering @ Judger — dose-response — `RESULTS.md` §3

GSM8K, 14B, n=120. Vector `v = mean(exec) − mean(reflection+transition)`, layer 28.
Control = 621 tok / 93.3%.

| coef | acc | tokens | Δ tokens | s/sample |
|---|---|---|---|---|
| 0 | 93.3% | 621 | — | 6.08 |
| 20 | 94.2% | 562 | −9.6% | 6.16 |
| 40 | **95.0%** | 515 | −17.2% | 4.17 |
| 60 | 94.2% | 437 | −29.7% | 4.55 |
| 80 | 94.2% | 379 | **−38.9%** | 3.16 |

**Takeaway:** monotonic token reduction, accuracy held/improved, ~2× faster. Clean win.

---

## 3. SEAL cross-task transfer — `RESULTS.md` §4

| Task | control | steered | Δ tokens | Δ acc |
|---|---|---|---|---|
| GSM8K | 621 / 93.3% | coef80: 379 / 94.2% | −38.9% | +0.9 |
| ARC-C | 498 / 95.8% | coef40: 372 / 95.8% | −25.2% | 0.0 |
| ARC-C | " | coef80: 310 / 93.3% | −37.7% | −2.5 |
| MedQA | 1003 / 78.0% | coef60: 880 / 73.0% | −12.2% | −5.0 |

**Takeaway:** transfers to ARC at iso-accuracy; MedQA is task-sensitive (needs gentler coef).

---

## 4. Per-agent SEAL steering — `RESULTS.md` §5

GSM8K, 14B, n=120, generic GSM8K vector, control 621 / 93.3%:

| Steered agent | coef 40 | coef 80 |
|---|---|---|
| **Judger** | 515 / 95.0% | **379 / 94.2%** |
| Planner | 580 / 95.0% | 593 / 94.2% |
| Critic | 635 / 94.2% | 615 / 95.8% |
| Refiner | 623 / 91.7% | 608 / 95.0% |
| All agents (coef 60) | 455 / 93.3% | — |

Thought-type mix ~identical across roles (~88% execution) → the Judger advantage is
**structural** (sole text-emitter), not distributional. Isolation-proxy critic-native
vector gave a small acc bump (95.8% @ coef40) that did **not** survive in-pipeline (§5).

**Takeaway:** only the text-emitting Judger yields token savings; all-agents ≈ Judger-only.

---

## 5. Native correctness-contrastive upstream steering — `RESULTS.md` §9

`v_agent = mean(layer-28 latent | correct) − mean(… | incorrect)`, captured in-pipeline,
applied per-role. Test splits. Compared to generic Judger SEAL.

**GSM8K (test, n=300; control 92.7% / 602 tok):**

| Steering | coef | acc | tokens |
|---|---|---|---|
| Upstream-3 (native) | 80 | 93.3% | 608 (+0.9%) |
| All (native) | 80 | 92.0% | 678 (+12.6%) |
| Judger (native) | 80 | 91.0% | 658 (+9.1%) |
| **Judger (generic SEAL)** | 80 | 93.0% | **390 (−35.3%)** |

**MedQA (test, n=100 held-out; control 75.0% / 1038 tok):** upstream-3 coef80 72.0% /
1003; all coef80 66.0% / 1193; **generic Judger coef80 77.0% / 768 (+2 pts, −26%)**.

**ARC-C (test, n=200; control 94.5% / 511):** upstream-3 coef80 95.0% / 501 (within noise);
**generic Judger coef80 92.5% / 324 (−36.7%)**.

### 5b. Correctness probe (the mechanism) — `RESULTS.md` §9.1

5-fold AUC of each agent's in-pipeline layer-28 latent state vs final correctness:

| Agent | GSM8K | ARC-C | MedQA |
|---|---|---|---|
| Planner | 0.54 | 0.57 | 0.57 |
| Critic | 0.62 | 0.51 | 0.63 |
| Refiner | 0.59 | 0.46 | 0.64 |
| **Judger** | **0.69** | **0.73** | **0.80** |

Also: wrong answers over-generate by +68–82% tokens; ~92–100% of errors are wrong-value
(reasoning), not format.

**Takeaway:** upstream steering does not beat Judger-only SEAL on any axis. Correctness is
**barely linearly encoded upstream (near chance)** and decided **late at the Judger** —
so there is no linear "toward-correct" direction upstream to steer along.

---

## 6. KV-cache steering — `KV_STEERING_FINDINGS.md`

GSM8K, 14B, n=150. One-shot `V += c_v·S` (value-only). Control 94.7% / 587.7 tok.

| Surface | c_v | acc | tokens |
|---|---|---|---|
| handoff_last (1 of ~130 cols) | +8 / +12 / −8 | 91.3 / 92.0 / 93.3% | 590 / 610 / 604 |
| handoff_all | 0.1 | 94.0% | 584 |
| handoff_all | **1.0** | **72.0%** | 894 (collapse) |
| **judger_token** | +8 | 94.0% | **440 (−25%)** |
| **judger_token** | +12 | 93.3% | **392 (−33%)** |

**Mechanism:** two KV position types — dense *content* (steering corrupts; all-columns
saturates attention since weights sum to 1 → collapse at cv=1) vs *aggregation/routing*
(`judger_token`; steering rides). Single content column ≈ 1/130 attention → too weak.

**Takeaway:** steering the latent handoff is **falsified**; the aggregation position
(judger_token) works for efficiency (−25 to −33% tokens at unchanged acc).

---

## 7. Learned CES "boost" (this study) — `CES_LATENT_STEERING_FINDINGS.md`

Qwen3-4B, MedQA, K=10, layer 20, upstream {P,C,R} latent-only. CES rank loss
`softplus(E(y+)−E(y−))` trained end-to-end through the differentiable KV; `y+`=gold
letter, `y-`=mined wrong pred. Controls: α=0 sanity, α<0 negate, random-v; paired
bootstrap CIs; dev split (test never consumed). Gate-2 grad smoke passed
(`artifacts/ces/gate2_smoke.json`: grads reach `v` only, loss ↓, zero-v allclose baseline).

**7a. Dev go/no-go, last-token, n=80, unsteered = 0.650:**

| Train config | unsteered | steered α>0 | negate α<0 | decision |
|---|---|---|---|---|
| LR 1e-2, no clip | 0.650 | 0.662 | 0.650 | NULL, paired CI [−0.075, +0.10] |
| LR 5e-3, grad-clip 1.0 | 0.650 | 0.662 | 0.650 | NULL (same) |

Grad clipping stabilized `v_norm` (~4.95; raw grads had spiked 1e4–1e5) but changed nothing.

**7b. Coef sweep, last-token vector, dev n=40, ‖v‖≈4.95, unsteered 0.650:**

| α | acc | Δ | 95% CI |
|---|---|---|---|
| +4 | 0.600 | −0.05 | [−0.15, +0.05] |
| +8 | 0.650 | 0.00 | [−0.125, +0.125] |
| +16 | 0.600 | −0.05 | [−0.15, +0.05] |
| −8 | 0.600 | −0.05 | [−0.15, +0.05] |

Inert even at α=16 (Δ‖ ‖≈80); no dose-response, no polarity.

**7c. Coef sweep, all-tokens vector, dev n=40, ‖v‖≈4.99, unsteered 0.650:**

| α | acc | Δ | 95% CI | mean tokens |
|---|---|---|---|---|
| +1 | 0.600 | −0.05 | [−0.15, +0.05] | 962 |
| +2 | 0.675 | +0.025 | [−0.075, +0.15] | 1018 |
| +4 | 0.625 | −0.025 | [−0.125, +0.075] | 969 |
| +8 | 0.600 | −0.05 | [−0.15, +0.05] | 1005 |
| **−8** | **0.000** | **−0.65** | **[−0.775, −0.50]** | **1946** |

All-tokens training was well-conditioned (grad norms ~15). Positive α within noise;
α=−8 collapses to 0 with output ≈ 2048 cap (Judger degenerates).

**Takeaway:** an optimized, end-to-end-trained latent steer can only find **destructive**
directions, never an improving one. Last-token inert; all-tokens destructive-only.
Matches the §5b probe.

---

## 8. Role ablation (causal value of upstream agents) — `artifacts/sweeps/ablation_*`

| Task | config | acc | tokens |
|---|---|---|---|
| GSM8K n=300 | full | 0.93 | 587 |
| GSM8K n=300 | judger-only | 0.91 | 750 |
| MedQA n=300 (s42) | full | 0.717 | — |
| MedQA n=300 (s42) | judger-only | 0.737 | — |
| MedQA (s123) | planner+critic+judger | 0.69 | — |

**Takeaway:** dropping the latent upstream agents costs little accuracy (± noise) — a
third independent signal that upstream latent value is low.

---

## 9. Latent-readout diagnostics ("steer the reader" test) — `DIAG_READOUT_FINDINGS.md`

Qwen3-4B, MedQA, K=10, greedy (temp=0). Tests (a) does the Judger USE the cache, and
(b) does the cache CONTAIN the answer. Scripts: `scripts/diag_cache_usage.py`,
`scripts/diag_answer_probes.py`. Artifacts: `artifacts/diag/4b/`.

**9a. Cache-usage — decode the Judger with the item's own cache vs interventions, across
Judger token budgets. idx-aligned paired bootstrap CIs.**

n=60, budget 1024 (with the `zero` arm):

| condition | acc | mean tokens |
|---|---|---|
| real | 0.417 | 846 |
| shuffled (wrong-question) | 0.467 | 821 |
| zero (content zeroed) | 0.217 | 961 |
| none (judger-only) | 0.167 | 985 |

n=40, budget sweep {1024, 2048, 4096}:

| budget | real | shuffled | none | real−shuffled | real−none | tokens real/shuf/none |
|---|---|---|---|---|---|---|
| 1024 | 0.450 | 0.450 | 0.150 | 0.00 [−.15,.15] | +0.30 [.15,.45] | 822 / 797 / 983 |
| 2048 | 0.650 | 0.575 | 0.550 | +0.075 [.00,.15] | +0.10 [.00,.22] | 1000 / 934 / 1476 |
| 4096 | 0.675 | 0.600 | 0.625 | +0.075 [−.025,.175] | +0.05 [−.05,.15] | 1062 / 997 / 1750 |

**14B (test, n=40) — replicates 4B on the model where LatentMAS was validated:**

| budget | real | shuffled | none | real−shuffled | real−none | tokens real/shuf/none |
|---|---|---|---|---|---|---|
| 1024 | 0.675 | 0.625 | 0.350 | +0.05 | +0.325 | 722 / 712 / 916 |
| 2048 | 0.775 | 0.775 | 0.700 | 0.00 | +0.075 | 882 / 867 / 1329 |
| 4096 | 0.800 | 0.775 | 0.750 | +0.025 | +0.05 | 892 / 878 / 1603 |

- **Content ignored (both models):** real ≈ shuffled (a wrong-question cache is ~as good);
  at most a weak +0.05–0.075 at budget ≥ 2048 (CIs touch/cross 0).
- **Token-budget artifact (both models):** real−none collapses (4B +0.30→+0.05; 14B
  +0.325→+0.05); at 4096 judger-only ≈ full — the Judger re-solves when unconstrained.
- **Conciseness scaffold (both models):** any non-zero cache (real/shuffled) → ~40–45%
  fewer Judger tokens than judger-only at comparable accuracy, content-independent (4B
  ~1000 vs ~1600; 14B ~890 vs ~1600). Zeroed cache ≈ none (non-zero content, not mere positions).

**GSM8K 14B (test, n=60) — second task (open-ended numeric):**

| budget | real | shuffled | none | real−shuffled | real−none | tokens real/shuf/none |
|---|---|---|---|---|---|---|
| 256 | 0.233 | 0.200 | 0.250 | +0.03 | −0.02 | 254 / 255 / 256 |
| 512 | 0.567 | 0.550 | 0.467 | +0.02 | +0.10 | 444 / 451 / 477 |
| 1024 | 0.933 | 0.933 | 0.800 | 0.00 | +0.13 | 542 / 560 / 668 |

Same signature on a non-MC task: **real ≈ shuffled** at every budget. At 1024 real/shuffled
finish (~550 tok, not starved) yet beat judger-only (0.933 vs 0.80, none also uncapped at
668) — so the scaffold gives a **content-independent accuracy boost** here, not just tokens.

**9b. Answer probes — gold 4-way decodable from per-agent per-layer latent residual?
(train, n=200, chance 0.25, linear + MLP, 5-fold).**

| agent | best layer | linear acc | MLP acc | AUC |
|---|---|---|---|---|
| planner | 32 | 0.325 | 0.333 | ~0.50 |
| critic | 14 | 0.258 | 0.317 | ~0.50 |
| refiner | 22 | 0.267 | 0.283 | ~0.49 |

MLP ≈ linear, both near chance → the gold answer is barely encoded upstream.
(14B replicates: planner/critic/refiner best-layer MLP 0.32–0.33, AUC ~0.50 — near chance.)

**Takeaway:** the latent inter-agent channel is **not an information channel** — it's a
content-independent conciseness scaffold, and its accuracy "benefit" over judger-only is
mostly a token-budget artifact. This **falsifies** the readout-bottleneck premise
(info-present-but-unread). Surviving threads: a weak content signal (+0.075 @ budget ≥
2048, underpowered) and an efficiency story (~40% Judger-token reduction).

---

## 10. Synthetic latent scaffold — the POSITIVE — `SCAFFOLD_RESULTS.md`, `DIAG_READOUT_FINDINGS.md` §4b

Can the 3 upstream agents (33 latent forwards) be replaced by a synthetic KV cache?
Scripts: `scripts/diag_scaffold_sweep.py`, `scripts/train_synth_scaffold.py`.
Artifacts: `artifacts/diag/{4b,14b}/*`, `artifacts/ces/synth_scaffold_m64/`.

**10a. Mechanism sweep (which variant reproduces the effect?).**
- **4B MedQA, b1024, n=40** (real 0.450, none 0.150): only real/shuffled reproduce (0.450);
  matched-noise, single-vector, trunc≤256, pca≤64 all fail → on 4B nothing synthetic works
  (small model is brittle to off-manifold caches).
- **14B MedQA, b1024, n=40 — FLIP** (real 0.675, none 0.350): matched (0.600), pca32 (0.650)
  **now reproduce** real; repeat_last (0.225) and trunc64 (0.400) still fail. On the capable
  model the scaffold = coarse statistics at full length (not fine content, not a single vector).

**10b. Fixed universal synthetic scaffold — POSITIVE.** `synthglobal` = one fixed,
question-independent synthetic KV (Gaussian matched to averaged per-channel stats of 8 train
caches, full length), **zero upstream forwards**:

| set | real | synthglobal | none |
|---|---|---|---|
| 14B MedQA b1024 n=40 | 0.675 | **0.600** (tok 701) | 0.350 |
| 14B MedQA b1024 n=80 | 0.600 | **0.537** (~76% recovery) | 0.338 |
| 14B GSM8K b512 n=60 | 0.567 | **0.567** (full) | 0.467 |
| 14B GSM8K b1024 n=60 | 0.933 | **0.883** | 0.800 |

Robustness (3 independent synth seeds, MedQA b1024): 0.55 / 0.60 / 0.65 — not a lucky cache.

**10c. Shrinking (14B MedQA b1024 n=40; real 0.675, none 0.350).**
- Length: synth l32/l64 ≈ 0.575 (work!), l256 = 0.30 (single-seed noise). Rank: r64 0.55 →
  r32 0.525 → r8/16 0.45 (graceful).
- **Learned m=64 slots** (distilled to full-LatentMAS behavior): **0.500** — trains (loss 0.08,
  grad flows) but **underperforms the fixed full-length synth** (0.54–0.60). Compressing to a
  few learned slots loses accuracy; m=32/16 not run.

**Takeaway:** the KEEPER positive is a **fixed, full-length, question-independent synthetic KV
scaffold that ≈ real accuracy + conciseness with zero upstream forwards** (cross-model on the
flip, cross-task MedQA+GSM8K). It saves upstream **compute** but not KV **memory** (still
~1023 positions); shrinking it (short/low-rank/learned) recovers most but not all — the
"tiny scaffold matches real" outcome did not materialize.

---

## Cross-cutting conclusions

1. **Leverage is at the text-emitting Judger.** Three steering methods (residual SEAL,
   KV-cache, learned CES) + role ablation + the readout diagnostics all agree.
2. **The upstream latent channel is not a usable control surface for accuracy:** weak
   interventions are inert (1/130 attention), strong ones saturate/destroy, no direction
   improves — and the diagnostics show *why*: the Judger ignores cache **content**
   (real≈shuffled) and there's little answer info to read (probes near chance).
3. **What the latent channel actually does:** it is a **content-independent conciseness
   scaffold** (shown on MedQA 4B+14B and GSM8K 14B). A wrong-question cache helps ~as much
   as the correct one. On MedQA its accuracy edge over judger-only is largely a token-budget
   artifact (vanishes to ~+0.05 at 4096); on GSM8K it yields a genuine but still
   content-independent +0.13 boost at iso-budget — always with ~40% fewer Judger tokens.
4. **Mechanistic root:** correctness is decided late (Judger probe 0.69–0.80) and barely
   encoded upstream (0.46–0.64); gold answer barely decodable upstream (≈chance).
5. **The clean positives:** (a) SEAL/KV steering of the Judger for **efficiency** (−17 to
   −39% tokens at retained accuracy, transfers); (b) latent agents give a ~40% Judger-token
   reduction vs judger-only at iso-accuracy (content-independent); (c) **the headline
   method result (§10): a fixed, question-independent synthetic KV scaffold reproduces
   full-LatentMAS accuracy + conciseness with ZERO upstream forwards** on 14B (MedQA+GSM8K).
   Shrinking it (short/low-rank/learned) recovers most but not all; it saves compute, not KV memory.

## State of play (what is settled)

The central finding is **cross-model (4B+14B) and cross-task (MedQA+GSM8K), with clean
causal controls**: the LatentMAS latent inter-agent channel is a **content-independent
conciseness scaffold, not an information channel**. The single most quotable result is
**real ≈ shuffled** (a wrong-question cache helps the Judger as much as the correct one).
This is a defensible workshop/Findings-tier mechanistic reframing today (see
[`PAPER_OUTLINE.md`](PAPER_OUTLINE.md)). What we do **not** have: any mechanism that makes
latent steering improve accuracy.

## How to make progress from here (two forks)

**Fork A — write up the honest reframing (low risk).** Package what exists; needs only
light hardening:
- multi-seed + larger-n on the cache-swap (tighten real vs shuffled; currently n=40–60, 1 seed);
- a hard task (AIME) where the Judger cannot re-solve → the strongest token-budget crossover;
- an accuracy–token Pareto ({judger-only, full, shuffled} × budget) to package the efficiency positive.

**Fork B — take one real swing at a positive (higher ceiling, needs new code).** Since the
channel carries ~no decodable answer info, don't steer content that isn't there — *make*
the channel carry it:
- **learned bridge/aggregation token(s)** between Refiner and Judger, trained (through the
  existing differentiable-KV path) so the answer becomes decodable from the handoff and the
  Judger causally uses it; success = beats the shuffled-cache control (not just none).
- alternative: state-conditioned / low-rank readout gates on the Judger's attention over the
  latent cache. (Details in `LATENT_STEERING_RESEARCH_HANDOFF.md` §6 and `DIAG_READOUT_FINDINGS.md` §4–5.)
- **Updated headwind (§10):** the scaffold sweep shows the effect is NOT compressible —
  rank-64 PCA and 256 real positions already fail to reproduce it. So a learned *few*-slot
  scaffold reproducing full-LatentMAS behavior is now a strong long-shot; a bridge token that
  must *beat shuffled* (i.e., transmit real content) faces the additional wall that answer
  content is barely decodable upstream at all. Fork A is the higher-EV path.

## Fully-built infra to reuse (all validated, on the pod)

- Diagnostics: `scripts/diag_cache_usage.py` (`--task {medqa,gsm8k}`, cache real/shuffled/zero/none
  × budget), `scripts/diag_answer_probes.py` (per-agent/layer probes), `scripts/run_diagnostics.sh`.
- Differentiable KV path + steerers/losses: `models.py` (`generate_latent_batch_grad`,
  `teacher_force_nll`), `seal/ces_steerer.py`, `seal/ces.py` — can train *any* inserted module
  (e.g., a bridge token), not just a static vector.
- Boost/eval harness: `scripts/train_ces_claim_b.py`, `scripts/eval_ces_heldout.py`
  (`--mode boost --control`), `scripts/coef_sweep.py`.

## Artifact index (on pod `/workspace/latentmas-baseline/artifacts/`)

`diag/4b/cache_usage/` (n=60 @1024 w/ zero arm), `diag/4b/cache_usage_hibudget/`
(n=40 @{1024,2048,4096}), `diag/4b/answer_probes/` (report.json + features.npz),
`ces/boost_4b/` (last-token: pairs, train, dev report), `ces/boost_4b_tuned/`,
`ces/boost_4b_all/coef_sweep/report.json`, `ces/gate2_smoke.json`,
`gate1/*/summary.json`, `sweeps/kv_*`, `sweeps/ablation_*`, `kv_steer_vectors/`, `plots/`.
(Native vectors / capture CSVs and `seal_vectors/` live on the pod volume, not in the
local checkout.)
