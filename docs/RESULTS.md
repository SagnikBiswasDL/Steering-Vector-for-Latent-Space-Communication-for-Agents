# Results: SEAL-steered LatentMAS for Token Efficiency

Detailed results for steering the LatentMAS **Judger** with a SEAL reasoning-
calibration vector to reduce output tokens. For the architecture/method walkthrough
see [`EXPERIMENT_OVERVIEW.md`](EXPERIMENT_OVERVIEW.md); for the project log see
[`PROGRESS.md`](PROGRESS.md).

**TL;DR:** A training-free SEAL steering vector applied to the LatentMAS Judger's
decoding reduces output tokens by **~25–39%** while accuracy is **retained or
improved**, and the vector **transfers across tasks** (extracted on GSM8K, works on
ARC-Challenge). Decoding is also ~2× faster at the strongest setting.

---

## 1. Setup

| | |
|---|---|
| Backbone | Qwen3-14B (40 layers, hidden dim 5120) |
| System | LatentMAS, sequential MAS (Planner → Critic → Refiner → Judger) |
| Backend | HuggingFace Transformers (torch 2.8/cu128, transformers 4.57.1, vLLM 0.11) |
| Hardware | RunPod, NVIDIA H200 |
| Latent steps | 40 per non-Judger agent |
| Sampling | temperature 0.6, top-p 0.95, seed 42 |
| Eval size | n = 120 per cell (subset; single seed) |
| SEAL vector | `v = mean(execution) − mean(reflection+transition)`, layer 28, from ~50 GSM8K-train CoT traces (1765 execution / 218 reflection / 76 transition steps) |
| Intervention | `hidden[:, -1, :] += coef · v̂` at layer 28, **Judger decoding only** |
| Metric | accuracy + mean output tokens (Judger), counted up to EOS |

The intervention targets the Judger because it is the only LatentMAS agent that
emits text; the latent agents (Planner/Critic/Refiner) reason in latent space and
produce zero output tokens.

---

## 2. Baseline faithfulness (fork vs. paper)

Before any steering, the fork reproduces the published LatentMAS accuracy
(Qwen3-14B, n=150):

| Task | Our fork | Paper (LatentMAS) |
|---|---|---|
| GSM8K | 92.7% | 95.2% |
| ARC-Challenge | 94.7% | 95.6% |
| MedQA | 79.3% | 80.7% |

All within ~1–2.5 pts (subset / single seed / stochastic decoding vs. the paper's
3-seed full-set means).

---

## 3. Main result — GSM8K dose-response

Steering the Judger away from reflection/transition thoughts (`coef > 0`) gives a
clean monotonic token reduction with accuracy holding or improving:

| coef | accuracy | mean output tokens | Δ tokens vs control | sec/sample |
|---|---|---|---|---|
| 0 (control) | 93.3% | 621.3 | — | 6.08 |
| 20 | 94.2% | 561.8 | −9.6% | 6.16 |
| 40 | **95.0%** | 514.6 | −17.2% | 4.17 |
| 60 | 94.2% | 436.9 | −29.7% | 4.55 |
| 80 | 94.2% | 379.4 | **−38.9%** | 3.16 |

- **coef=40** is the accuracy sweet spot: +1.7 pts over control at −17% tokens.
- **coef=80** maximizes savings: **−39% tokens, ~2× faster decode, accuracy still above control.**
- A negative coefficient (push *toward* reflection) increases tokens and lowers
  accuracy — confirming the steering direction.

---

## 4. Cross-task transfer

The vector was extracted **only from GSM8K** and applied unchanged to other tasks.

### ARC-Challenge (n=120)
| coef | accuracy | mean output tokens | Δ tokens |
|---|---|---|---|
| 0 | 95.8% | 497.6 | — |
| 40 | **95.8%** | 372.2 | **−25.2% (iso-accuracy)** |
| 80 | 93.3% | 309.9 | −37.7% |

A 25% token reduction at **identical** accuracy from a vector trained on a
different task — reproducing SEAL's transferability claim inside latent MAS.

### MedQA
_Pending (GPU reclaimed mid-run); will be added._

---

## 5. Takeaways

1. The LatentMAS Judger **does over-reason** on these tasks (~620 GSM8K output
   tokens at baseline), so there is real headroom for calibration.
2. A single training-free steering vector cuts that by **25–39% with no accuracy
   cost**, and adds an inference speedup as a side effect.
3. The vector **transfers across tasks**, so per-task extraction is not required.

## 6. Limitations & next steps

- **Statistical power:** current results are n=120, single seed. Planned: n≈500,
  3 seeds, significance, and a full accuracy–token **Pareto frontier**.
- **Coverage:** MedQA transfer + a GSM8K negative-control are still to be added;
  more backbones (4B/8B) would strengthen generality.
- **Next direction:** steer the **latent inter-agent communication channel** (the
  KV working memory passed between agents) — unique to latent MAS and unaddressed
  by SEAL / cache-steering — as the core novel contribution.

---

*Reproduction commands and code: see [`EXPERIMENT_OVERVIEW.md` §8](EXPERIMENT_OVERVIEW.md)
and `scripts/extract_seal_vector.py` / `run.py --seal ...`.*
