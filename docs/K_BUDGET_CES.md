# K-Budget Recovery via CES (working notes)

Primary RQ: can a learned residual steering vector applied during **recurrent latent steps** recover full-budget LatentMAS accuracy at smaller `K`?

## Implementation vs assumptions

| Assumption in early prompts | Actual LatentMAS fork |
|---|---|
| One shared K-sequence | Per upstream agent; total forwards ≈ `3*(K+1)` |
| Gradients through latent chain | Inference is `@torch.no_grad()`; train path is separate |
| Steer all agent forwards | Primary: **latent steps only**; prefill+latent is ablation |
| Primary compute = K or forward count | **Wall-clock latency** primary; forwards secondary (unequal cost) |
| Answer-only NLL as main loss | Smoke only; main = CES ranking / Judger distillation |
| Bit-identical baseline when disabled | **Allclose** at explicit tolerance |

## Steering phase

- `latent_only` (default): disable steerer during agent prompt prefill; enable for each recurrent latent `inputs_embeds` step.
- `prefill_and_latent` (ablation): steerer on for the whole `generate_latent_batch` call.

## Gates

1. Unsteered K-curve with paired bootstrap CIs + latency.
2. Gradients reach `v` through K_small latent → KV → Judger teacher-force (answer NLL smoke).
3. Tiny overfit, then CES ranking / distillation.
4. Held-out frontier eval.

Do not mine pairs or scale training until Gates 1–2 pass.
