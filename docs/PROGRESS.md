# Project Worklog & Goals

This is the running log of what we're building, why, and what's been done so far.
For the architecture/mechanics explainer see [`EXPERIMENT_OVERVIEW.md`](EXPERIMENT_OVERVIEW.md).

---

## The overarching goal

Combine two training-free methods to make multi-agent LLM reasoning more
efficient:

- **LatentMAS** — agents collaborate by passing **latent state (KV cache)**
  instead of text.
- **SEAL** — a **steering vector** that suppresses redundant "reflection /
  transition" reasoning.

**Near-term objective (current focus): token efficiency.** Use SEAL to cut the
output-token wastage (over-reasoning) of LatentMAS *without losing accuracy*.

The work is staged as a sequence of PRs, each building on a verified foundation.

---

## Phase 1 — Establish a verified baseline (PR #1)

**What:** Created a **verbatim fork** of the upstream LatentMAS implementation
(`Gen-Verse/LatentMAS@9a9e4d3`) as our baseline.

**Why:** Before changing anything, we need a known-good reference that provably
matches the paper, so any later gains are attributable to *our* changes, not to
implementation drift.

**Done:**
- All 24 upstream files reproduced byte-for-byte (git blob SHAs match).
- Apache-2.0 `LICENSE` preserved; `NOTICE` attribution added.
- Branch `baseline/latentmas-fork` pushed; `main` scaffolded.
- No-GPU smoke tests pass locally.

---

## Phase 2 — Verify the fork reproduces the paper (on GPU)

**What:** Ran the fork on a RunPod box (2x NVIDIA H200) and confirmed accuracy
matches the LatentMAS paper on Qwen3-14B.

**Why:** "Proof of proper forking" — the code is not just structurally identical
but **functionally faithful**.

**Done (Qwen3-14B, sequential MAS, n=150, latent_steps=40, HF backend):**

| Task | Our fork | Paper (LatentMAS) |
|---|---|---|
| GSM8K | 92.7% | 95.2% |
| ARC-Challenge | 94.7% | 95.6% |
| MedQA | 79.3% | 80.7% |

All within ~1-2.5 pts (subset / single seed / stochastic decoding vs. the paper's
3-seed full-set means). The fork is faithful.

Environment that works: torch 2.8.0+cu128, transformers 4.57.1, vllm 0.11.0,
CUDA 12.8, Python 3.12; venv on root disk reusing system torch; HF cache +
results on the persistent `/workspace` volume.

---

## Phase 3 — SEAL for token efficiency (PR #2, in progress)

**What:** Clean-room SEAL implementation that steers the **Judger's text
decoding** to suppress reflection/transition thoughts and shorten output.

**Why the Judger specifically:** in LatentMAS only the Judger emits text; the
Planner/Critic/Refiner reason in latent space and emit zero tokens. So the only
place to *reduce output tokens* is the Judger's decode loop. (A prior internal
pilot steered the latent agents instead and saw no effect — by construction it
couldn't touch text-token usage.)

**Built & committed (`feature/seal-token-efficiency`):**
- `seal/` module: `thought_classifier` (exec/reflection/transition), `vector_generation`
  (`v = mean(exec) - mean(refl+trans)`), `extraction` (offline CoT pipeline),
  `hooks` (`SealSteerer` residual-stream forward hook).
- `scripts/extract_seal_vector.py`: CLI to build + save the steering vector.
- `models.py`: apply the hook during `generate_text_batch` (layer 28); count
  generated tokens.
- `run.py`: `--seal/--seal_vector/--seal_layer/--seal_coef` flags; report
  `total_output_tokens` / `mean_output_tokens` in the result JSON.
- `methods/{latent_mas,baseline}.py`: thread `output_tokens` into results.
- `docs/EXPERIMENT_OVERVIEW.md`: full architecture explainer.

**Steering vector extracted:** `artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt`
from 50 GSM8K-train CoT traces (1765 execution / 218 reflection / 76 transition
steps), **raw_norm = 52.7** (this sets the natural scale for the steering coef).

**A/B sweep (PENDING — interrupted by pod going offline):**
Plan: GSM8K, n=120, control vs. `coef ∈ {20, 40, 60, 80}` plus `-40` as a
sign-sanity check, split across both H200s. Metric: **Δ mean_output_tokens at
matched accuracy**. The pod stopped/migrated mid-run; results files (if any
completed) are on the persistent `/workspace` volume.

---

## Current status (Jun 27)

- Implementation: **complete and pushed.**
- Steering vector: **extracted.**
- Empirical SEAL A/B numbers: **pending** — the RunPod pod is offline
  (`container not found`); needs a restart. `/workspace` (code, vector, HF cache,
  any finished results) persists; the root-disk venv must be recreated (~3 min).

## Next steps (when the pod is back)
1. Recreate the venv; re-pull `feature/seal-token-efficiency`.
2. Read any completed A/B summaries from `results/`; relaunch unfinished coefs.
3. Report the accuracy-vs-token curve; pick the coef with the best token
   reduction at retained accuracy.
4. (Optional) test cross-task transfer of the GSM8K vector to ARC-C / MedQA.
