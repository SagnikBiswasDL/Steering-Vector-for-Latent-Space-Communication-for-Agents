# Handoff: In-Pipeline, Natively-Acquired Steering Vectors for All 4 LatentMAS Agents

You are picking up an active research project with fresh context. This document
gives you everything: the goal, what exists, what we found, the infrastructure,
the gotchas, and the specific next problem to solve. Read it fully before acting.

---

## 0. The one-sentence mission

We already showed SEAL steering of the **Judger** cuts output tokens ~25–39% at
retained accuracy. Now: **build in-pipeline, natively-acquired steering vectors for
each of the four agents (Planner, Critic, Refiner, Judger), and test whether
steering the first three — even though their runtime is fixed — sets up a smoother /
better Judger optimization (on accuracy and/or length).** This requires gathering a
lot of pipeline data and digging into wrong answers.

---

## 1. Project background

- **LatentMAS** (Zou et al., arXiv:2511.20639): training-free multi-agent framework.
  Four agents run sequentially: **Planner → Critic → Refiner → Judger**. The first
  three reason in **latent space** — each runs a fixed `latent_steps` (=40) forward
  passes, feeding the last-layer hidden state back as the next input embedding (via a
  realignment matrix `W_a`), emitting **no text**. They communicate by a shared,
  growing **KV cache** (`past_key_values`). Only the **Judger** decodes text (the
  final answer).
- **SEAL** (Chen et al., arXiv:2504.07986): training-free activation steering. Build
  `v = mean(execution) − mean(reflection ∪ transition)` hidden states at a deep layer,
  then add `coef·v̂` to the residual stream during decoding to suppress redundant
  reasoning. In this repo `coef > 0` shortens output.
- **Our project goal**: combine them, but (per advisor) go beyond "SEAL+LatentMAS"
  to a real contribution via per-agent analysis and new vector-generation strategies.

## 2. Advisor (Jiayi)'s asks (verbatim intent)

1. Evaluate how SEAL's vector works on Planner/Critic/Refiner/Judger **separately**. ✓ done.
2. For the sub-agents SEAL doesn't help, develop **new steering-vector generation
   strategies**: (a) plot execution vs non-execution thought ratios per sub-agent and
   explain why SEAL works/not; (b) propose different thought-type divisions / vectors,
   drawing on the steering literature.

## 3. What we've already established (results)

All on Qwen3-14B, GSM8K/ARC n=120, MedQA n=100, seed 42, temp 0.6, layer 28,
latent_steps 40, HF backend. Control = plain LatentMAS. Full detail in
[`RESULTS.md`](RESULTS.md); plot in `docs/assets/agent_thoughts_gsm8k.png`.

- **Fork is faithful** (byte-for-byte + reproduces paper accuracy). Don't re-verify.
- **Judger dose-response (GSM8K):** control 621 tok / 93.3%; coef40 515/95.0; coef60
  437/94.2; coef80 379/94.2. → **−17 to −39% tokens, accuracy held/improved, ~2× faster.**
- **Cross-task transfer (GSM8K vector, Judger):** ARC-C −25% tokens at iso-accuracy
  (95.8%); MedQA −12% tokens but −5 pts accuracy (task-sensitive; needs gentler coef).
- **Per-agent steering (generic GSM8K vector), control 621/93.3%:**
  - Judger: coef40 515/95.0, coef80 379/94.2  ← only agent that cuts tokens
  - Planner: 580/95.0, 593/94.2
  - Critic: 635/94.2, 615/95.8
  - Refiner: 623/91.7, 608/95.0
  - All agents (coef60): 455/93.3 ≈ Judger-only (437/94.2)
- **Thought distribution (isolation proxy):** all four roles ~88% execution / ~12%
  non-execution → the Judger's advantage is **structural** (it is the only
  text-emitter), NOT because it is more reflective.
- **Phase C (critic-native vector):** steering the Critic never changes tokens, but a
  critic-native vector improved Critic accuracy (coef40 → 95.8% vs 93.3% control).
  → latent-agent steering is a **quality lever, not a length lever.**

## 4. THE KEY INSIGHT that frames your work

- The first three agents run a **fixed** number of latent steps. You **cannot** save
  their runtime — so do not frame their steering as an efficiency win.
- Their value is **setting up the Judger**: they shape the shared KV working memory
  the Judger decodes from. So the hypothesis is: **better latent thoughts from
  Planner/Critic/Refiner → the Judger reaches a correct answer with fewer/cleaner
  tokens.** Steering the sub-agents is a means to a **better or smoother Judger
  optimization** (accuracy and/or length), not an end in itself.
- Much of our prior per-agent vector work was done in **isolation** (each role
  prompted alone, in text). We want to move **in-pipeline**: derive each agent's
  vector from its **real latent activations while running inside the full pipeline**,
  conditioned on upstream agents' KV.

## 5. Your concrete task

Build and evaluate **in-pipeline, natively-acquired steering vectors per agent**, and
measure whether steering the upstream agents improves the end-to-end Judger outcome
vs. Judger-only SEAL. Specifically:

1. **Activation capture (in-pipeline):** add a mode that, during a real LatentMAS run,
   records each agent's layer-L latent hidden states (the 40 latent-step vectors) and
   the run's final correctness (Judger right/wrong). No text classification needed.
2. **Contrastive, correctness-based extraction:** for each agent,
   `v_agent = mean(latent acts | correct runs) − mean(latent acts | incorrect runs)`.
   This is fully in-pipeline, text-free, and matches the "quality lever" finding.
   (Also consider: thought-type divisions decoded from real latent thoughts via
   LatentMAS "debug mode" — harder; the contrastive route avoids needing text.)
3. **Error analysis (dig into wrong answers):** gather a large set of pipeline runs;
   analyze *where* wrong answers go wrong (bad plan? bad critique? Judger mis-decode?).
   Use this to motivate which agent's vector matters and what direction helps.
4. **Evaluate combinations:** steer {each agent alone, upstream-3, all} with the
   native vectors, and measure downstream Judger accuracy AND tokens. Key comparison:
   **does upstream steering beat Judger-only SEAL** on the accuracy–token frontier?
5. **Scale + rigor:** larger n (≥300–500), multiple seeds where feasible; report an
   accuracy–token Pareto frontier.

## 6. Codebase orientation (branch `feature/seal-token-efficiency`)

- `run.py` — CLI. Flags: `--method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k
  --prompt sequential --latent_steps 40 --max_new_tokens 2048 --max_samples N
  --generate_bs 25 --temperature 0.6 --top_p 0.95` plus SEAL flags:
  `--seal --seal_vector <path.pt> --seal_layer 28 --seal_coef <c>
  --seal_agents {planner,critic,refiner,judger|comma-list|all}`. Reports JSON incl.
  `accuracy`, `mean_output_tokens`.
- `models.py` — `ModelWrapper`. `generate_latent_batch` (sub-agents; feeds hidden
  states back with `W_a` realignment; **this is where you add activation capture**),
  `generate_text_batch` (Judger decode; SEAL hook + token counting). SEAL hook
  registered once, toggled per role via `_seal_activate_for(role)` and
  `self.seal_active_roles`.
- `methods/latent_mas.py` — the pipeline loop; passes `role=agent.role` to both
  generate calls; threads `output_tokens` into results; has `correct` per item.
- `seal/` — `hooks.py` (`SealSteerer`: forward hook, `apply_to="last"` = current
  position at layer L), `vector_generation.py` (`build_steering_vector`),
  `extraction.py` (offline text-CoT extraction; accepts a `message_builder`),
  `thought_classifier.py` (keyword exec/reflection/transition; pluggable).
- `scripts/extract_seal_vector.py` — CLI extractor; `--role` uses a role's prompt.
- `scripts/analyze_agent_thoughts.py` — per-role thought distribution + plot.
- Vectors saved as `torch.save({'unit_vector','vector','layer_index','raw_norm',...})`
  under `artifacts/seal_vectors/qwen3-14b/`. `SealSteerer.from_artifact` loads them.

## 7. Infrastructure & environment (READ — many gotchas)

- **Compute:** RunPod pods with NVIDIA H200 (140 GB). The pod **auto-stops after
  ~18h idle** and often **migrates to a new pod ID / SSH endpoint**. The `/workspace`
  network volume (code, vectors, HF cache, results) **persists** across stop/migrate;
  the root disk (`/root`, incl. the venv) is **ephemeral** (wiped on stop).
- **Working dir on pod:** `/workspace/latentmas-baseline` (a clone of this repo;
  `git pull` to sync). `/workspace` has a **~50 GB quota** — DO NOT put the venv there.
- **Env rebuild (~5 min)** after any pod restart, on the ROOT disk:
  `python3 -m venv --system-site-packages /root/venv` (reuse system torch 2.8+cu128),
  then `pip install transformers datasets accelerate numpy tqdm matplotlib hf_transfer`,
  then `pip install -c <(echo "torch==2.8.0") vllm`, then
  `pip install -c <(echo "torch==2.8.0") transformers==4.57.1`. A ready script is on
  the volume: `/workspace/recover_env.sh`. `env.sh` sets `HF_HOME=/workspace/.cache/
  huggingface`, `TMPDIR=/root/tmp`, activates the venv, cd's in.
- **Pinned versions that WORK:** torch 2.8.0+cu128, **transformers==4.57.1** (5.x
  breaks LatentMAS KV handling), vllm 0.11.0. `methods/latent_mas.py` has an
  unguarded `from vllm import SamplingParams`, so vllm must be importable even for the
  HF backend. The pod sets `HF_HUB_ENABLE_HF_TRANSFER=1`, so `hf_transfer` must be
  installed or dataset downloads fail.
- **Model/data cache:** Qwen3-14B and GSM8K are cached on `/workspace/.cache`. ARC-C
  and MedQA download on first use (MedQA also has a local `data/medqa.json`).
- **Driving the pod:** the RunPod SSH proxy (`<id>@ssh.runpod.io`) is
  interactive-only (no exec, no SCP). We drive it via a helper that pipes commands
  over a forced PTY and strips ANSI — see `/tmp/rp.sh` pattern in prior work (feed a
  heredoc of commands to `ssh -tt ... <<'EOF'`). Run long jobs with `nohup ... &` and
  poll log files. Get results back by reading files over the proxy (or base64 for
  binaries like plots).
- **Known bugs already fixed** (don't reintroduce): output-token counting must slice
  generated tokens at `input_ids.shape[1]` (padded width), not per-row prompt length;
  role prompt builders need `args.task` set (SimpleNamespace(model_name, task)).

## 8. Deliverables expected from you

- New code: in-pipeline activation capture + contrastive (correct-vs-incorrect)
  per-agent vector extraction, on the feature branch (or a new one).
- A data/error-analysis pass over wrong answers (where does the pipeline fail?).
- An experiment table + Pareto plot: native per-agent steering vs. Judger-only SEAL,
  on accuracy AND tokens.
- Update `RESULTS.md` with findings; keep everything committed/pushed.
- An honest verdict: does steering the fixed-runtime upstream agents actually help the
  Judger (accuracy or length), or is Judger-only steering sufficient?

## 9. Guardrails

- The baseline fork is verified — don't modify upstream files' semantics; layer new
  work as additions.
- Only commit when asked by the user; push feature branches, don't touch `main`
  destructively.
- Report numbers honestly with n / seed caveats; a negative result (upstream steering
  doesn't help) is a valid, reportable finding.
