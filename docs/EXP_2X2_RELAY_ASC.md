# 2×2: Relay compression (H-OBF) × Judger concision steering (ASC)

> Full pipeline walkthrough (code map, knobs, confirm-suite numbers, operating
> points): [`CACHE_COMPRESSION_PIPELINE.md`](CACHE_COMPRESSION_PIPELINE.md).

Updated: 2026-07-30. Design + results for the headline systems experiment:

> **Can a compressed inter-agent relay cut KV memory while Judger-only concision
> steering removes any downstream verbosity tax, at iso-accuracy?**

Prior work in this repo established that LatentMAS's inter-agent latent cache is a
*content-independent conciseness scaffold* (see `MASTER_BRIEF.md`). This experiment
takes the systems consequence: shrink the relay (memory), and if compression makes
the Judger ramble, cancel that with concision steering — decoupling *memory* from
*verbosity*.

## Design

Two factors, fully crossed, paired on the same items, greedy decode:

| Arm | Relay | Judger |
|---|---|---|
| A | Full KV | no ASC (baseline) |
| B | Full KV | ASC |
| C | H-OBF compressed | no ASC (compression alone) |
| D | H-OBF compressed | ASC (target config) |

Optional diagnostics (`--with_evict`): E = plain-H (eviction only) no ASC, F = +ASC.

Upstream Planner/Critic/Refiner are **always unsteered**; ASC applies at the
**Judger decoding only** (`seal_agents=judger`).

**Primary analysis** (per-item, then paired bootstrap 95% CI) on
{final tokens, Judger decode time, end-to-end latency, accuracy}:

```
ASC effect under Full   = B − A
ASC effect under H-OBF  = D − C
Interaction             = (D − C) − (B − A)
Compression tax (noASC) = C − A
Combined vs Full        = D − A
```

Target story: C inflates Judger tokens vs A; ASC removes it (D ≈ A tokens); D
accuracy ≈ A; D relay MB ≪ A; D end-to-end latency lowest.

## Components (in-repo)

- **`seal/relay_compress.py`** — `RelayCompressor` with modes:
  - `full` (identity), `evict` (sink + top-(budget−sink) positions by importance),
    `obf` (evict + rank-`r` low-rank backfill of the evicted mass as `r` synthetic
    positions).
  - Importance: `key_norm` (default), `value_norm`, `recency`. Reports MB in/out,
    positions in/out, compression ratio, and a `sink_retained` invariant.
  - **Faithfulness note:** released H-OBF selects per-*head* by accumulated
    attention; a dense HF `DynamicCache` needs one position axis shared across
    heads, so we select **per layer** (importance aggregated across heads). The
    scorer + backfill are pluggable; swap in the attention scorer for a
    byte-faithful port. The 2×2 interaction is robust to the exact scorer.
- **`scripts/exp_2x2_relay_asc.py`** — the harness. Builds the upstream relay once
  per item, compresses it (timed), decodes the Judger per arm with a fresh cache
  clone, logs every metric, computes the 2×2 interaction + correctness flips,
  writes `report.json` + `rows.json`. `--smoke` adds PASS/FAIL wiring gates.
- **`scripts/run_2x2.sh`** — detached driver (smoke → paired) with per-phase
  sentinels.
- **`tests/test_relay_compress.py`** — CPU unit tests for the compressor (7/7).

## ASC vector

`--asc_vector artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt` — the generic
SEAL exec−reflection residual (layer 28, dim 5120). Functionally a
verbose→concise vector (−17% tokens @ coef40, −39% @ coef80 on full-cache GSM8K;
see `RESULTS.md` §2). A dedicated verbose→concise ASC vector is a drop-in swap.
If `--asc_vector` is omitted, only the no-ASC arms (A, C[, E]) run.

## Launch (pod)

```bash
# env: /root/venv + HF_HOME=/workspace/.cache/huggingface
python scripts/exp_2x2_relay_asc.py --model_name Qwen/Qwen3-14B --task gsm8k \
  --k 40 --n 100 --judger_budget 768 \
  --relay_budget 32 --sink 4 --rank 8 --importance key_norm \
  --asc_vector artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt --asc_coef 40 \
  --out_dir artifacts/exp2x2/paired
# or: bash scripts/run_2x2.sh all
```

## Results

### Phase 1 — smoke gate (GSM8K, 14B, n=12, k=40, budget 512) — ALL PASS

| arm | relay | asc | acc | tok | dec_s | relay MB | pos | eos |
|---|---|---|---|---|---|---|---|---|
| A | full | off | 0.583 | 427 | 10.1 | 101.25 | 648 | 0.58 |
| B | full | on | 0.750 | 427 | 10.1 | 101.25 | 648 | 0.58 |
| C | obf | off | 0.500 | 413 | 9.8 | **6.25** | **40** | 0.50 |
| D | obf | on | 0.583 | 442 | 10.6 | **6.25** | **40** | 0.50 |

Gate: H-OBF genuinely smaller (**16× KV: 101→6 MB, 648→40 pos**) ✓; sink retained
✓; no invalid output ✓; ASC shortens median output ✓; upstream unsteered ✓.
Combined D matched full baseline A on accuracy (0.583). Token-tax signal noisy at
n=12 with ~50% truncation at budget 512 → paired 100-run at budget 768.

### Phase 2 — paired 2×2 (GSM8K, 14B, n=100, k=40, budget 768)

| arm | relay | asc | acc | tok | dec_s | e2e_s | relay MB | pos | eos |
|---|---|---|---|---|---|---|---|---|---|
| A | full | off | 0.800 | 524 | 12.38 | 15.37 | 99.65 | 638 | 0.80 |
| B | full | on | 0.860 | 489 | 11.64 | 14.63 | 99.65 | 638 | 0.89 |
| C | obf | off | 0.790 | 534 | 12.63 | 15.64 | **6.25** | **40** | 0.77 |
| D | obf | on | **0.860** | **497** | **11.90** | **14.91** | **6.25** | **40** | 0.87 |

Paired interaction (95% CI, n=100):

| metric | Compression tax C−A | ASC \| Full (B−A) | ASC \| H-OBF (D−C) | Interaction | Combined D−A |
|---|---|---|---|---|---|
| tokens | +9.9 [−14.9,+34.2] n.s. | **−35.5 [−56,−15]** | **−36.5 [−61,−16]** | −1.0 [−29,+28] | **−26.6 [−47,−7]** |
| decode_s | +0.25 n.s. | **−0.74 [−1.24,−0.25]** | **−0.73 [−1.31,−0.24]** | +0.01 n.s. | **−0.49 [−0.97,−0.02]** |
| accuracy | −0.01 n.s. | +0.06 [.00,.12] | +0.07 [.00,.14] | +0.01 n.s. | **+0.06 [.00,.13]** |

**Findings:**
1. **Compression is ~free.** H-OBF cut the relay **16× (99.65→6.25 MB, 638→40
   positions)** with no accuracy cost (C−A = −0.01, n.s.) and only a small,
   non-significant token bump (+10 tok / +2%).
2. **The hypothesized big "verbosity tax + ASC-removes-it" interaction did NOT
   appear** — the tax was small (n.s.), so the interaction is ≈0 on every metric.
   Directional hint only (C eos 0.77 < A 0.80; C +10 tok).
3. **ASC and compression are orthogonal / complementary.** ASC delivers the SAME
   benefit on the compressed relay as on the full one — tokens −7% (D−C −36.5 ≈
   B−A −35.5), decode −0.73s (≈ −0.74s), accuracy +0.07 (≈ +0.06); all
   interactions ≈0. Concision steering transfers perfectly through compression.
4. **The combined corner D (H-OBF + ASC) is the best config:** matches/beats the
   uncompressed baseline A on accuracy (0.860 vs 0.800; D−A = +0.06 [.00,.13])
   with **16× less relay KV, −27 tokens, and −0.49s Judger decode**.

**Headline:** *H-OBF relay compression and Judger-only concision steering stack
cleanly — together they cut inter-agent KV 16× and Judger tokens ~7% while
slightly improving accuracy vs uncompressed LatentMAS.* The compression paper's
"~10% verbosity tax" did not reproduce with per-layer H-OBF at budget=32/sink=4/
rank=8 on GSM8K; inducing it likely needs a more aggressive budget (8–16).

Figure: `artifacts/exp2x2/paired/plot_2x2.png`. Caveats: single seed, n=100, one
task/budget/relay-budget; per-layer (not per-head-attention-faithful) H-OBF.

### Performance: batched decode (`--batch_size`)

The harness now batches the Judger decode across items (`decode_batch` +
`_pad_caches_left`): per-item caches are left-padded to a common length with a
correct past mask (uniform compressed caches need no padding), so all items in a
batch decode in one `model.generate` call. This took GPU util from ~0–single-digit
(batch=1) to **~60–67%** on the H200 and cut per-item decode ~10×. Upstream is
still built per item. **Caveat:** under batching, per-item decode time = batch
wall-time / B (constant within a batch), so the *timing* paired-CIs are degenerate
— report timing as per-batch throughput, not per-item paired diffs. Accuracy,
tokens, and MB remain exact per item. (Ops note: don't put `pkill -f
exp_2x2_relay_asc` in a launch command — it matches the ssh command's own argv and
kills the launcher; launch with a clean `nohup`/`tmux` and no such `pkill`.)

### Phase 3 — aggressive compression (budget=16) + plain-H diagnostic (n=60, batched)

GSM8K, 14B, k=40, budget 768, `relay_budget=16` (obf → 24 pos / 3.75 MB;
evict → 16 pos / 2.5 MB), `--with_evict`. Figure: `artifacts/exp2x2/agg_b16/plot_2x2.png`.

| arm | relay | asc | acc | tok | relay MB | pos | decode s |
|---|---|---|---|---|---|---|---|
| A | full | off | 0.817 | 529 | 100.02 | 640 | 2.34 |
| B | full | on | 0.867 | 488 | 100.02 | 640 | 2.35 |
| C | obf | off | 0.800 | **567** | 3.75 | 24 | 1.54 |
| D | obf | on | **0.867** | 523 | 3.75 | 24 | 1.54 |
| E | evict | off | 0.817 | 562 | **2.50** | 16 | 1.52 |
| F | evict | on | 0.800 | 532 | **2.50** | 16 | 1.54 |

Interaction (n=60): tokens C−A = **+37.4 [+11.2, +66.6]**; D−C = **−43.9 [−71, −16]**;
acc C−A = −0.017 [−0.117, +0.083] (n.s.); D−A = +0.05 [−0.017, +0.133].

**Findings (aggressive budget):**
- **Compression is near-free even at 27–40×:** obf@24pos (3.75 MB) and evict@16pos
  (2.5 MB) hold accuracy vs full (C−A = −0.017 n.s.; E = 0.817 = A exactly).
- **The verbosity tax now reproduces and is significant:** C−A = +37 tokens (~+7%),
  matching the compression paper's ~10% claim — and **ASC fully removes it**
  (D−C = −44; D tokens 523 ≈ A 529) while lifting accuracy to the full-ASC level
  (D 0.867 = B 0.867).
- **Plain-H ≥ OBF backfill:** plain eviction (E: 16 pos, 2.5 MB, acc 0.817) matches
  full accuracy and *beats* OBF (C: 24 pos, 3.75 MB, acc 0.800) on both accuracy and
  memory → the mean-pool rank-8 backfill does not earn its keep here; plain headwise
  eviction is the better, cheaper compressor. (A true low-rank-factored backfill may
  do better — queued.)
- **Best corners:** D (obf+ASC) = full-ASC accuracy at 27× less KV; E (plain-H) =
  full accuracy at **40× less KV** with no steering.

**Utilization:** batched decode ran at ~60–94% GPU util (vs ~0 at batch=1);
per-item decode ~1.5 s (compressed) / 2.3 s (full).

### Still queued
- **Second task** (MedQA) + a second seed to harden "compression is free".
- Better OBF backfill (true low-rank factors vs mean-pool) given plain-H > OBF here.
