# 4–6 hour Claim-B sprint

Budget: ~$50–70 at ~$8/h → target **4–6h wall-clock** with 2×H200.

## Claim target (B)

`Accuracy(K=5 + latent CES) ≈ Accuracy(K=10)` on frozen MedQA test `[0,100)`.

Latent forwards: 18 vs 33 (−45%). Wall-clock not required for B.

## Frozen splits

| Split | Indices | Use |
|-------|---------|-----|
| test | 0–99 | Held-out only (Gate1) |
| train | 100–219 | Pair mining / CES train |
| dev | 220–299 | Role ablation if time |

## Dual-GPU schedule (both busy)

| Phase | GPU0 | GPU1 |
|-------|------|------|
| Mine (~25–35 min) | **K=10** on train n=50 | **K=5** on same n=50 |
| Merge | — | `merge_budget_pairs.py` → `pairs.json` |
| Train | Idle / later held-out | 4B overfit → 4B CES → **14B CES** |
| Eval | Held-out low vs steered | Controls if time |

Do **not** leave a GPU waiting on the other for the whole mine.

## Decision

- recovery ≥ 0.8 → Claim B live; optional Judger ASC combo later
- 0.4–0.8 → one more objective or layer, then stop
- < 0.4 → stop Claim B; pivot Judger ASC with remaining budget

## Skip for time

- Full 8-config role matrix on 14B
- Every-layer sweep
- Adaptive K controller
- AIME as primary (transfer only if B succeeds)
