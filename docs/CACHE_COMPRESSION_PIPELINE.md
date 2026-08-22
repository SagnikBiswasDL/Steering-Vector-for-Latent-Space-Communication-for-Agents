# Cache Compression Pipeline (LatentMAS Relay)

Updated: 2026-08-05. End-to-end documentation of how we compress the LatentMAS
inter-agent KV cache, how to run it, and what the confirm suite showed.

Related: [`EXP_2X2_RELAY_ASC.md`](EXP_2X2_RELAY_ASC.md) (original 2×2 design),
[`MASTER_BRIEF.md`](MASTER_BRIEF.md) (mechanism context), `paper/main.tex`
(compression-first narrative).

---

## 1. What gets compressed

LatentMAS pipeline:

```
Question → Planner → Critic → Refiner → [relay KV cache] → Judger (text) → Answer
              └── latent space (no tokens) ──┘
```

- Upstream agents run `K` latent steps each and grow a shared `past_key_values`.
- Only the **Judger** decodes tokens. It reads the whole relay as a prefix KV.
- On Qwen3-14B that relay is typically **~640–1100 positions / ~100–180 MB**.

**Compression sits between Refiner and Judger:** build the real upstream cache
once, shrink it, then decode the Judger from the compressed cache.

```
upstream (unsteered)  →  RelayCompressor  →  Judger decode (± ASC)
     real KV                  ↓
                        keep sink + top-k
                        (± low-rank backfill)
```

Mechanism motivation (prior work): the Judger is largely insensitive to
*instance-level* cache content (wrong-question ≈ right-question), so the relay is
highly redundant within a task and should compress.

---

## 2. Code map

| Piece | Path | Role |
|---|---|---|
| Compressor | `seal/relay_compress.py` | `RelayCompressor` modes `full` / `evict` / `obf` |
| Unit tests | `tests/test_relay_compress.py` | CPU tests (selection, sink, MB) |
| Harness | `scripts/exp_2x2_relay_asc.py` | Build cache → compress → batched Judger arms |
| Drivers | `scripts/run_2x2.sh`, `run_compress_confirm.sh`, `run_night_aggressive.sh` | Unattended chains |
| ASC vector | `artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt` | Judger-only concision steer |
| Artifacts | `artifacts/exp2x2/{paired,agg_b16,compress_confirm,night}/` | `report.json` + `rows.json` |

---

## 3. Compressor (`RelayCompressor`)

### Modes

| Mode | Behavior | Output length (per layer) |
|---|---|---|
| `full` | Identity | original `S` |
| `evict` (plain-H) | Keep first `sink` positions + top-`(budget−sink)` by importance; drop rest | `budget` |
| `obf` (H-OBF-style) | `evict`, then append `rank` synthetic positions that low-rank–summarize the *evicted* mass | `budget + rank` |

### Importance scoring (per layer, shared across heads)

Dense HF `DynamicCache` needs one position axis for all heads, so scores are
aggregated across heads within each layer:

- `key_norm` (default) — \(\|K\|\) summed over heads  
- `value_norm` — \(\|V\|\) summed over heads  
- `recency` — later positions preferred  

Kept indices are sorted ascending (causal order preserved). Sink positions are
always retained (invariant checked in the harness).

### Knobs

```
--relay_budget   # kept prompt positions per layer (e.g. 16, 32, 64, 128)
--sink           # always-kept prefix (default 4)
--rank           # OBF backfill synthetic positions (default 8)
--importance     # key_norm | value_norm | recency
```

Memory scales roughly with kept positions. Example on GSM8K/14B: full ~100 MB /
640 pos → `evict@16` → **2.5 MB / 16 pos (~40×)**.

### Faithfulness note

Upstream H-OBF selects per-*head* via attention. We select per-*layer* so the
result fits a standard dense cache. Scorer/backfill are pluggable; the 2×2
interaction claims do not require byte-faithful attention scoring.

---

## 4. Experiment harness (arms)

For each item the harness:

1. Builds the **real** upstream cache (Planner/Critic/Refiner, unsteered).
2. Compresses clones with `obf` / optional `evict`.
3. Optionally builds **shuffled** (wrong-question), **crosstask** (other-task
   donor), **none** (Judger-only).
4. Batched Judger decode per arm (`--batch_size`), greedy unless noted.
5. Writes per-item `rows.json` + aggregated `report.json` with paired bootstrap CIs.

### Arm table

| Arm | Relay | ASC (Judger SEAL) |
|---|---|---|
| A | full | off |
| B | full | on |
| C | obf | off |
| D | obf | on |
| E | evict | off (optional `--with_evict`) |
| F | evict | on |
| S | shuffled | off (`--with_shuffled`) |
| X | crosstask | off (`--with_crosstask`) |
| N | none | off (`--with_none`) |

### Contrasts (paired, same items)

```
compression tax (no ASC) = C − A
ASC | full               = B − A
ASC | compressed         = D − C
interaction              = (D − C) − (B − A)
combined                 = D − A
shuffled vs full         = S − A
none vs full             = N − A
```

---

## 5. How to run

Env (pod): `source env_native.sh` (HF_HOME + `/root/venv`; rebuild with
`/workspace/recover_env.sh` after `/root` wipe).

```bash
# Smoke
python scripts/exp_2x2_relay_asc.py --smoke --task gsm8k --n 12 \
  --k 40 --judger_budget 512 --relay_budget 32 \
  --asc_vector artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt \
  --out_dir artifacts/exp2x2/smoke

# Aggressive GSM8K 2×2 (headline)
python scripts/exp_2x2_relay_asc.py --task gsm8k --n 60 --k 40 \
  --judger_budget 768 --relay_budget 16 --sink 4 --rank 8 \
  --with_evict --with_shuffled --batch_size 20 \
  --asc_vector artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt --asc_coef 40 \
  --out_dir artifacts/exp2x2/gsm8k_b16

# Gentler MATH / MedQA (use larger --relay_budget)
python scripts/exp_2x2_relay_asc.py --task math --n 60 --k 40 \
  --judger_budget 2048 --relay_budget 128 --with_evict --with_shuffled --with_none \
  --asc_vector artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt \
  --out_dir artifacts/exp2x2/math_b128
```

Unattended chains:

```bash
tmux new -d -s comp  'bash scripts/run_compress_confirm.sh > /workspace/compress_confirm.log 2>&1'
tmux new -d -s night 'bash scripts/run_night_aggressive.sh > /workspace/night_aggressive.log 2>&1'
```

---

## 6. Confirm-suite results (2026-08-04/05, Qwen3-14B)

Artifacts: `artifacts/exp2x2/compress_confirm/`, `artifacts/exp2x2/night/`.

### Compression vs full accuracy (obf, no ASC)

| Task | Keep (`relay_budget`) | Approx ratio | Full → Compress | Verdict |
|---|---|---|---|---|
| GSM8K | 16 | ~40× | 0.817 → 0.800 (evict **0.817**) | **Free** (seeds 123 & 7) |
| MATH | 128 | ~5× | 0.783 → **0.800** | **Free** |
| MATH | 64 | ~9× | 0.783 → 0.733 (−5) | Soft |
| MATH | 32 | ~16× | 0.783 → 0.717 (−7) | Hurts |
| MedQA | 128 | ~9× | 0.633 → 0.567 (−7, n.s.) | Borderline |
| MedQA | 64 | ~16× | 0.633 → 0.517 (−12) | Hurts |
| MedQA | 16–32 | ~40× | −27 / −28 pts | **Breaks** |

**Rule of thumb:** aggressive 16–40× is a GSM8K-style result; on MATH/MedQA use
**gentler** budgets (~64–128 kept positions). Plain `evict` often matches or beats
`obf` on memory at similar accuracy (mean-pool rank-8 backfill is not always worth it).

### Mechanism checks (same suite)

- **Shuffled ≈ full** on GSM8K/MATH (MedQA −5 pts, n.s.).
- **None** collapses MedQA (0.267 vs 0.633) — presence matters under tight budgets.
- **MedQA ← GSM8K donor:** 0.533 vs 0.633 (−10) — cross-task not free.
- **ASC** shortens tokens on compressed and full arms; interaction ≈ 0 when the
  verbosity tax is real (GSM8K aggressive).

### AIME caveat

The cache-swap harness at long budgets under-scores AIME (many 8192 truncations).
**Do not** use that for claims. Full `LatentMASMethod` Gate1 sanity still shows
K=10 **0.667** vs K=0 **0.567** — the fork is fine; use Gate1-style eval for AIME.

---

## 7. Recommended operating points

| Goal | Task | Setting |
|---|---|---|
| Max memory save, iso-acc | GSM8K | `evict` or `obf`, `relay_budget=16`, `sink=4` |
| Harder math, iso-acc | MATH | `relay_budget=128` (or 64 if accepting ~5 pt risk) |
| MedQA | MedQA | start at `relay_budget=128`; do not use 16–32 |
| Verbosity after compress | any | Judger ASC `@ coef=40` on the GSM8K SEAL vector |

---

## 8. What this pipeline does *not* do

- It does **not** remove upstream latent forwards (cache must still be built once
  per question, unless you switch to a synthetic scaffold — separate line).
- It does **not** claim per-head attention-faithful H-OBF.
- AIME long-decode cache-swap numbers from `exp_2x2` are **not** paper-grade.

---

## 9. Quick mental model

1. Upstream writes a big, redundant relay KV.  
2. `RelayCompressor` keeps sinks + high-importance positions (± tiny backfill).  
3. Judger decodes from the small cache; ASC optional for length.  
4. Easy tasks tolerate extreme shrink; hard tasks need a thicker residual.
