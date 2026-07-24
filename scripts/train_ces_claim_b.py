#!/usr/bin/env python3
"""Claim-B CES trainer (lean 4–6h path).

Objectives:
  answer_nll  — gold-answer teacher-force NLL (overfit / smoke)
  ces_rank    — softplus(E(y+) - E(y-)) with y+=gold boxed, y-=failed pred boxed
  distill     — KL(teacher K_full Judger logits || student) on gold answer tokens
                (teacher past detached; expensive — use sparingly)

Only the CES vector is trainable. Steers latent_only on upstream agents.

Example (overfit 8 pairs on 4B):
  python scripts/train_ces_claim_b.py \\
    --model_name Qwen/Qwen3-4B --pairs artifacts/ces/pairs_k10_vs_k5/pairs.json \\
    --objective answer_nll --max_pairs 8 --steps 30 --ces_layer 20 \\
    --out_dir artifacts/ces/train_4b_overfit

Then promote best vector to 14B held-out eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from methods import default_agents  # noqa: E402
from models import ModelWrapper  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.ces import ces_rank_loss, combined_ces_objective, kl_tokenwise  # noqa: E402
from seal.ces_steerer import TrainableSteerer  # noqa: E402
from utils import set_seed, auto_device  # noqa: E402


def make_ns(args) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name,
        task="medqa",
        prompt="sequential",
        think=False,
        latent_only=False,
        sequential_info_only=False,
        agents=None,
        use_vllm=False,
        device=args.device,
        device2="cuda:1",
        max_new_tokens=args.max_new_tokens,
        text_mas_context_length=-1,
        temperature=0.0,
        top_p=1.0,
        seed=args.seed,
        seal=False,
        kvsteer=False,
        ces=True,
        capture_acts=None,
        latent_space_realign=False,
    )


def boxed_ids(tokenizer, text: str, device) -> torch.Tensor:
    s = str(text).strip()
    if not s.startswith("\\boxed"):
        s = f"\\boxed{{{s}}}"
    return tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def rollout_past_nograd(wrapper, question: str, k: int, ns, agents) -> torch.Tensor:
    past = None
    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=question, context="", method="latent_mas", args=ns
        )
        _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        past = wrapper.generate_latent_batch(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=int(k),
            past_key_values=past,
            role=agent.role,
        )
    return past


def rollout_past_grad(wrapper, question: str, k: int, ns, agents):
    past = None
    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=question, context="", method="latent_mas", args=ns
        )
        _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        past = wrapper.generate_latent_batch_grad(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=int(k),
            past_key_values=past,
            role=agent.role,
        )
    return past


def judger_prompt_ids(wrapper, question: str, ns):
    j_messages = build_agent_message_sequential_latent_mas(
        role="judger", question=question, context="", method="latent_mas", args=ns
    )
    _, j_ids, _, _ = wrapper.prepare_chat_batch([j_messages], add_generation_prompt=True)
    return j_ids


def energy_on_targets(wrapper, past, j_ids, target_ids) -> torch.Tensor:
    return wrapper.teacher_force_nll(past, j_ids, target_ids, role="judger", steer_judger=False)


def distill_kl(wrapper, past_teacher, past_student, j_ids, target_ids) -> torch.Tensor:
    """KL(teacher || student) on answer positions (teacher detached)."""
    # Build logits manually via a short teacher-force forward for both.
    full_ids = torch.cat([j_ids, target_ids], dim=1)
    attn = torch.ones_like(full_ids, device=full_ids.device)

    def _logits(past):
        past_len = 0
        if past is not None:
            from models import _past_length
            past_len = _past_length(past)
        if past_len > 0:
            past_mask = torch.ones((attn.shape[0], past_len), dtype=attn.dtype, device=attn.device)
            am = torch.cat([past_mask, attn], dim=-1)
        else:
            am = attn
        out = wrapper.model(
            input_ids=full_ids,
            attention_mask=am,
            past_key_values=past,
            use_cache=False,
            return_dict=True,
        )
        # logits for target tokens: positions [j_len-1 .. j_len+t-2] predict targets
        j_len = j_ids.shape[1]
        return out.logits[:, j_len - 1 : j_len - 1 + target_ids.shape[1], :]

    with torch.no_grad():
        # Detach teacher past graph
        if past_teacher is not None:
            # past may be DynamicCache; leave as-is under no_grad
            pass
        logits_t = _logits(past_teacher)
    logits_s = _logits(past_student)
    return kl_tokenwise(logits_t, logits_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B")
    ap.add_argument("--pairs", required=True, help="pairs.json from mine_budget_pairs.py")
    ap.add_argument("--objective", default="answer_nll",
                    choices=["answer_nll", "ces_rank", "distill", "ces_rank+distill"])
    ap.add_argument("--k_low", type=int, default=5)
    ap.add_argument("--k_full", type=int, default=10)
    ap.add_argument("--max_pairs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="Max-norm gradient clip on v (0 disables). Stabilizes the "
                         "backprop-through-KV path, whose raw grads can spike to 1e4-1e5.")
    ap.add_argument("--max_v_norm", type=float, default=0.0,
                    help="If >0, project v back to this L2 norm after each step (trust region on magnitude).")
    ap.add_argument("--ces_layer", type=int, default=20)
    ap.add_argument("--ces_coef", type=float, default=1.0)
    ap.add_argument("--init_std", type=float, default=0.01)
    ap.add_argument("--apply_to", default="last", choices=["last", "all"],
                    help="Steer only the last latent token ('last') or every latent token position ('all').")
    ap.add_argument("--alpha_distill", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/ces/train_claim_b")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.pairs) as f:
        pairs = json.load(f)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        print("No pairs — abort", file=sys.stderr)
        sys.exit(1)
    print(f"[train] n_pairs={len(pairs)} objective={args.objective} model={args.model_name}", flush=True)

    ns = make_ns(args)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    for p in wrapper.model.parameters():
        p.requires_grad_(False)
    # Construct the steerer directly (rather than wrapper.attach_ces) so we can set
    # apply_to; the pod's models.py attach_ces does not expose that argument.
    hidden = int(wrapper.model.config.hidden_size)
    ces = TrainableSteerer(
        hidden_size=hidden,
        layer_index=args.ces_layer,
        coef=args.ces_coef,
        init_std=args.init_std,
        agents={"planner", "critic", "refiner"},
        steer_phase="latent_only",
        apply_to=args.apply_to,
    )
    ces.register(wrapper.model)
    ces.to(wrapper.device)
    ces.disable()
    wrapper.ces = ces
    ces.freeze_host_model(wrapper.model)
    opt = torch.optim.Adam([ces.v], lr=args.lr)
    agents = [a for a in default_agents() if a.role != "judger"]
    device = wrapper.device

    history = []
    t_start = time.time()
    for step in range(args.steps):
        pair = pairs[step % len(pairs)]
        question = pair["question"]
        gold = pair["gold"]
        y_pos = boxed_ids(wrapper.tokenizer, gold, device)
        neg_pred = pair.get("low_prediction") or "x"
        if str(neg_pred).strip().lower() == str(gold).strip().lower():
            # fallback wrong letter
            neg_pred = {"a": "b", "b": "c", "c": "d", "d": "a"}.get(str(gold).lower()[:1], "b")
        y_neg = boxed_ids(wrapper.tokenizer, neg_pred, device)

        opt.zero_grad(set_to_none=True)
        ces.enable()
        past_s = rollout_past_grad(wrapper, question, args.k_low, ns, agents)
        j_ids = judger_prompt_ids(wrapper, question, ns)
        e_pos = energy_on_targets(wrapper, past_s, j_ids, y_pos)

        loss = None
        extras = {}
        if args.objective == "answer_nll":
            loss = combined_ces_objective(e_pos, use_answer_nll=True)
        elif args.objective == "ces_rank":
            e_neg = energy_on_targets(wrapper, past_s, j_ids, y_neg)
            loss = ces_rank_loss(e_pos, e_neg)
            extras = {"e_pos": float(e_pos.detach()), "e_neg": float(e_neg.detach())}
        elif args.objective == "distill":
            ces.disable()
            past_t = rollout_past_nograd(wrapper, question, args.k_full, ns, agents)
            ces.enable()
            # recompute student past (graph)
            past_s = rollout_past_grad(wrapper, question, args.k_low, ns, agents)
            loss = distill_kl(wrapper, past_t, past_s, j_ids, y_pos)
        else:  # ces_rank+distill
            e_neg = energy_on_targets(wrapper, past_s, j_ids, y_neg)
            rank = ces_rank_loss(e_pos, e_neg)
            ces.disable()
            past_t = rollout_past_nograd(wrapper, question, args.k_full, ns, agents)
            ces.enable()
            past_s2 = rollout_past_grad(wrapper, question, args.k_low, ns, agents)
            kd = distill_kl(wrapper, past_t, past_s2, j_ids, y_pos)
            loss = rank + float(args.alpha_distill) * kd
            extras = {"rank": float(rank.detach()), "kd": float(kd.detach())}

        loss.backward()
        grad_norm = float(ces.v.grad.norm().item()) if ces.v.grad is not None else 0.0
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([ces.v], float(args.grad_clip))
        opt.step()
        if args.max_v_norm and args.max_v_norm > 0:
            with torch.no_grad():
                vn = ces.v.norm()
                if float(vn) > args.max_v_norm:
                    ces.v.mul_(float(args.max_v_norm) / (float(vn) + 1e-8))
        v_norm = float(ces.v.detach().norm().item())
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "v_norm": v_norm,
            "idx": pair.get("idx"),
            **extras,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if (step + 1) % 10 == 0 or step == args.steps - 1:
            ckpt = {
                "v": ces.v.detach().cpu(),
                "layer_index": args.ces_layer,
                "coef": args.ces_coef,
                "steer_phase": "latent_only",
                "apply_to": args.apply_to,
                "agents": ["planner", "critic", "refiner"],
                "objective": args.objective,
                "model_name": args.model_name,
                "k_low": args.k_low,
                "step": step,
            }
            path = os.path.join(args.out_dir, f"ces_step{step:04d}.pt")
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(args.out_dir, "ces_latest.pt"))

    summary = {
        "config": vars(args),
        "n_pairs": len(pairs),
        "history": history,
        "elapsed_sec": time.time() - t_start,
        "final_loss": history[-1]["loss"] if history else None,
        "loss_improved": (
            history[-1]["loss"] < history[0]["loss"] if len(history) >= 2 else None
        ),
    }
    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("n_pairs", "elapsed_sec", "final_loss", "loss_improved")}, indent=2))


if __name__ == "__main__":
    main()
