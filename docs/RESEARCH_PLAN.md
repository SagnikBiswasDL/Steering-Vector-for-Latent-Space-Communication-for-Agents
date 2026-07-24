# Research Plan: Per-Agent Steering of LatentMAS

Response to advisor (Jiayi) feedback: combining SEAL + LatentMAS is not a
publication on its own, so we turn it into an analysis-driven method contribution
by (1) evaluating steering **per sub-agent**, (2) explaining *why* it works or not
via thought-type distributions, and (3) designing **new steering-vector strategies**
for the agents where vanilla SEAL underperforms.

Current status: Judger-only SEAL gives ~25–39% token reduction at retained
accuracy (see [`RESULTS.md`](RESULTS.md)). This plan generalizes beyond the Judger.

---

## Key technical fact that shapes the design

Only the **Judger** decodes text. Planner/Critic/Refiner reason in **latent space**
(a fixed 40-step loop, no tokens emitted). Consequences:

- **Steering a sub-agent** = applying the SEAL hook during that agent's *latent*
  forward passes (implemented via `--seal_agents`, which toggles the hook per role).
- **"Token wastage" is a Judger-side metric.** Steering a latent agent cannot
  reduce its own tokens (it has none); it can only change the KV working memory it
  hands downstream, which then affects the Judger's accuracy + token count. So the
  outcome metric for *every* per-agent experiment is still downstream
  accuracy + Judger `mean_output_tokens`.
- **Thought-distribution analysis** requires text. We characterize each role's
  reasoning style by decoding its role prompt to text (isolation proxy;
  `scripts/analyze_agent_thoughts.py`). A context-aware / latent-debug-decode
  variant is a follow-up.

---

## Experiment matrix (Phase A: where does SEAL help per role?)

Model Qwen3-14B, GSM8K first (then ARC-C, MedQA), n≥120, sweep `coef ∈ {0,20,40,60,80}`.

| Steered role(s) | `--seal_agents` | Question answered |
|---|---|---|
| Judger only (current) | `judger` | baseline effect (done) |
| Planner only | `planner` | does cleaning the *plan* help downstream? |
| Critic only | `critic` | does steering an inherently-reflective role help or hurt? |
| Refiner only | `refiner` | effect on the refinement step |
| All latent (no judger) | `planner,critic,refiner` | latent-channel-only steering |
| All | `all` | combined |

Metric for each: downstream accuracy + Judger `mean_output_tokens` (+ speed).
Deliverable: a per-role table showing where steering helps, is neutral, or hurts.

## Analysis (Phase B: why?)

`scripts/analyze_agent_thoughts.py` produces, per role, the fraction of
execution / reflection / transition steps (SEAL Figure-1 style bar chart).
Hypotheses to test against the plot:

- **Planner / Judger**: execution-heavy → SEAL's "suppress reflection" direction
  is well-matched → steering helps (consistent with the Judger result).
- **Critic**: reflection/transition-heavy *by design* (its job is to critique) →
  suppressing reflection may **hurt** it → vanilla SEAL is the wrong vector here.
- **Refiner**: mixed.

The plot + per-role steering results together explain the "works/doesn't" pattern.

## New steering strategies (Phase C: for the roles SEAL fails on)

Vanilla SEAL uses one division: `execution` vs `reflection ∪ transition`, and one
vector direction `mean(exec) − mean(refl+trans)`. For roles where that is wrong
(likely Critic/Refiner), propose **role-appropriate divisions** and extract
role-specific vectors (`extract_seal_vector.py --role <role>` already generates
traces with each role's own prompt). Candidate divisions:

- **Critic**: `productive critique` (identifies a concrete flaw / correction) vs
  `redundant hedging` (vague doubt, repetition). Steer toward productive critique
  rather than away from all reflection.
- **Refiner**: `applied change` (incorporates feedback) vs `restating` (echoes the
  plan without improving it). Steer toward change.
- **General**: contrastive pairs (correct vs incorrect traces) instead of thought
  types — steer toward the "correct-trace" direction (accuracy-oriented, not just
  length-oriented).

Extraction hooks are in place: `seal/extraction.py` accepts a `message_builder`
(role prompts) and `seal/thought_classifier.py` is a pluggable function where
alternative label schemes can be added.

## Related steering work to draw from

- **SEAL** (Chen et al., 2504.07986) — our base: thought-type steering for CoT calibration.
- **Contrastive Activation Addition / CAA** (Rimsky et al.) — difference-of-means steering from contrastive pairs; motivates the correct-vs-incorrect division.
- **Representation Engineering / RepE** (Zou et al., 2310.01405) — reading/controlling concept directions; useful for probing where thought types separate per layer/role.
- **Cache steering** (Belitsky et al., 2507.08799) — one-shot KV edit; relevant to steering the inter-agent latent channel directly (a distinct, more novel direction).

## Implementation status (pre-GPU, done)

- `--seal_agents` (per-role hook toggling) — `models.py`, `methods/latent_mas.py`, `run.py`.
- `scripts/analyze_agent_thoughts.py` — per-role thought distribution + plot.
- `scripts/extract_seal_vector.py --role` — role-specific vector extraction.

## GPU runs to launch (when a pod is up)

1. Phase B analysis plot (fast; ~1 generation pass per role over n≈60).
2. Per-role vector extraction for planner/critic/refiner.
3. Phase A per-role steering sweeps (GSM8K), then cross-task.
4. Phase C: extract role-appropriate vectors, re-run the roles SEAL failed on.
