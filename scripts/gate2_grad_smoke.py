#!/usr/bin/env python3
"""Gate 2: smallest gradient-flow smoke test for CES on K_small LatentMAS.

Answer-NLL only (smoke). Does NOT mine pairs or run CES ranking at scale.

Checks:
  1. Only the CES vector is trainable
  2. Nonzero grad reaches v through latent rollout → KV → Judger teacher-force
  3. Loss decreases over a few steps on 1 example
  4. Disabled / zero vector matches unsteered energy within allclose tolerance
  5. Optionally probe whether gradient checkpointing + KV is usable (document fail)

Usage (needs GPU for real models):
  python scripts/gate2_grad_smoke.py \\
    --model_name Qwen/Qwen3-4B --k_small 5 --ces_layer 20 --steps 5

CPU toy mode (no HF weights; validates hook + autograd plumbing only):
  python scripts/gate2_grad_smoke.py --toy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.ces import combined_ces_objective  # noqa: E402
from seal.ces_steerer import TrainableSteerer  # noqa: E402
from seal.hooks import _find_decoder_layers  # noqa: E402


ALLCLOSE_RTOL = 1e-4
ALLCLOSE_ATOL = 1e-5


def _toy_smoke() -> Dict:
    """Minimal decoder-like module: residual add must receive grads into v."""

    class TinyBlock(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.lin = nn.Linear(d, d, bias=False)

        def forward(self, x):
            # Mimic HF layer output tuple (hidden, ...)
            return (self.lin(x),)

    class TinyModel(nn.Module):
        def __init__(self, d: int = 32, n_layers: int = 4):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TinyBlock(d) for _ in range(n_layers)])
            self.lm_head = nn.Linear(d, 50, bias=False)

        def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
            if inputs_embeds is None:
                # unused in toy
                inputs_embeds = torch.randn(1, 2, self.lm_head.in_features)
            h = inputs_embeds
            for layer in self.model.layers:
                h = layer(h)[0]
            return type("O", (), {"logits": self.lm_head(h), "past_key_values": None})()

    d = 32
    model = TinyModel(d=d)
    for p in model.parameters():
        p.requires_grad_(False)
    steerer = TrainableSteerer(
        hidden_size=d, layer_index=2, coef=1.0, init_std=0.01,
        agents={"planner"}, steer_phase="latent_only",
    )
    steerer.freeze_host_model(model)
    steerer.register(model)
    steerer.set_active_role("planner")
    steerer.set_phase("latent")
    steerer.enable()

    x = torch.randn(1, 3, d, requires_grad=False)
    out = model.model.layers[2](x)[0]
    # Hook already applied inside forward of layer 2 when we call through registered module
    # Re-run via full path:
    steerer.enable()
    h = x
    for i, layer in enumerate(model.model.layers):
        if i == 2:
            steerer.set_phase("latent")
        h = layer(h)[0]
    logits = model.lm_head(h)
    target = torch.randint(0, 50, (1, 3))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 50), target.reshape(-1))
    loss.backward()

    assert steerer.v.grad is not None and steerer.v.grad.abs().sum() > 0, "v.grad missing"
    grad_norm = float(steerer.v.grad.norm().item())
    assert steerer.trainable_parameter_names() == ["v"]

    # Zero-vector allclose: disable steerer vs coef=0
    steerer.disable()
    with torch.no_grad():
        h0 = x
        for layer in model.model.layers:
            h0 = layer(h0)[0]
    steerer.coef = 0.0
    steerer.enable()
    steerer.set_phase("latent")
    with torch.no_grad():
        h1 = x
        for layer in model.model.layers:
            h1 = layer(h1)[0]
    ok = torch.allclose(h0, h1, rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL)
    steerer.remove()
    return {
        "mode": "toy",
        "passed": bool(ok),
        "trainable": ["v"],
        "grad_norm": grad_norm,
        "allclose_disabled": bool(ok),
        "allclose_rtol": ALLCLOSE_RTOL,
        "allclose_atol": ALLCLOSE_ATOL,
    }


def _probe_checkpointing_with_kv(model) -> Dict:
    """Explicitly test whether gradient checkpointing + use_cache is usable.

    Many HF models disallow use_cache=True under gradient checkpointing.
    We document the result; we do not silently assume compatibility.
    """
    report = {"attempted": False, "compatible": None, "error": None}
    if not hasattr(model, "gradient_checkpointing_enable"):
        report["error"] = "no gradient_checkpointing_enable"
        return report
    report["attempted"] = True
    try:
        model.gradient_checkpointing_enable()
        # Tiny forward with cache
        input_ids = torch.randint(0, min(100, model.config.vocab_size), (1, 4), device=next(model.parameters()).device)
        out = model(input_ids=input_ids, use_cache=True)
        # One more step with past
        past = out.past_key_values
        out2 = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        loss = out2.logits.float().sum()
        loss.backward()
        report["compatible"] = True
    except Exception as e:  # noqa: BLE001
        report["compatible"] = False
        report["error"] = str(e)
    finally:
        try:
            model.gradient_checkpointing_disable()
        except Exception:  # noqa: BLE001
            pass
        model.zero_grad(set_to_none=True)
    return report


def _real_model_smoke(args) -> Dict:
    from data import load_gsm8k
    from methods import default_agents
    from models import ModelWrapper
    from prompts import build_agent_message_sequential_latent_mas
    from utils import set_seed, auto_device

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(
            "CUDA GPU required for --model_name smoke. "
            "Re-run with GPU info, or use --toy for CPU plumbing checks."
        )

    set_seed(args.seed)
    device = auto_device(args.device)
    ns = argparse.Namespace(
        method="latent_mas",
        model_name=args.model_name,
        task="gsm8k",
        prompt="sequential",
        think=False,
        latent_space_realign=False,
        use_vllm=False,
        seal=False,
        kvsteer=False,
        capture_acts=None,
        ces=False,
        agents=None,
        device=str(device),
        device2="cuda:1",
        enable_prefix_caching=False,
        use_second_HF_model=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        text_mas_context_length=-1,
        latent_only=False,
        sequential_info_only=False,
        max_new_tokens=512,
    )
    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=ns)
    ckpt_report = _probe_checkpointing_with_kv(wrapper.model)

    n_layers = int(wrapper.model.config.num_hidden_layers)
    layer = int(args.ces_layer)
    if layer < 0:
        layer = max(0, n_layers - 2)

    steerer = wrapper.attach_ces(
        layer_index=layer,
        coef=1.0,
        init_std=float(args.init_std),
        agents={"planner", "critic", "refiner"},
        steer_phase="latent_only",
    )
    steerer.freeze_host_model(wrapper.model)

    trainable = [n for n, p in wrapper.model.named_parameters() if p.requires_grad]
    ces_trainable = steerer.trainable_parameter_names()
    assert len(trainable) == 0, f"Host model still trainable: {trainable[:5]}"
    assert ces_trainable == ["v"], f"Unexpected CES params: {ces_trainable}"

    item = next(iter(load_gsm8k(split="train")))
    agents = [a for a in default_agents() if a.role != "judger"]
    # Build short gold answer string for teacher-force smoke
    gold = str(item.get("gold") or "42")
    answer_text = f"\\boxed{{{gold}}}"
    answer_ids = wrapper.tokenizer(
        answer_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)

    def rollout_energy(enable_ces: bool) -> torch.Tensor:
        past = None
        if enable_ces:
            steerer.enable()
        else:
            steerer.disable()
        for agent in agents:
            messages = build_agent_message_sequential_latent_mas(
                role=agent.role, question=item["question"], context="", method="latent_mas", args=ns
            )
            _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch(
                [messages], add_generation_prompt=True
            )
            past = wrapper.generate_latent_batch_grad(
                input_ids,
                attention_mask=attention_mask,
                latent_steps=int(args.k_small),
                past_key_values=past,
                role=agent.role,
            )
        # Judger prompt
        j_messages = build_agent_message_sequential_latent_mas(
            role="judger", question=item["question"], context="", method="latent_mas", args=ns
        )
        _, j_ids, _, _ = wrapper.prepare_chat_batch([j_messages], add_generation_prompt=True)
        # Detach past for disabled baseline energy comparison path when needed
        return wrapper.teacher_force_nll(past, j_ids, answer_ids, role="judger", steer_judger=False)

    # Allclose: zero vector vs disabled
    with torch.no_grad():
        steerer.v.zero_()
    steerer.disable()
    # Need a fresh no-grad baseline energy via inference path for fairness
    with torch.no_grad():
        past0 = None
        for agent in agents:
            messages = build_agent_message_sequential_latent_mas(
                role=agent.role, question=item["question"], context="", method="latent_mas", args=ns
            )
            _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch(
                [messages], add_generation_prompt=True
            )
            past0 = wrapper.generate_latent_batch(
                input_ids,
                attention_mask=attention_mask,
                latent_steps=int(args.k_small),
                past_key_values=past0,
                role=agent.role,
            )
        j_messages = build_agent_message_sequential_latent_mas(
            role="judger", question=item["question"], context="", method="latent_mas", args=ns
        )
        _, j_ids, _, _ = wrapper.prepare_chat_batch([j_messages], add_generation_prompt=True)
        e_base = wrapper.teacher_force_nll(past0, j_ids, answer_ids)

    # Zero-vector steered (latent_only) should allclose to baseline
    with torch.no_grad():
        steerer.v.zero_()
    steerer.enable()
    with torch.no_grad():
        # Use grad path under no_grad for zero-v comparison
        past_z = None
        for agent in agents:
            messages = build_agent_message_sequential_latent_mas(
                role=agent.role, question=item["question"], context="", method="latent_mas", args=ns
            )
            _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch(
                [messages], add_generation_prompt=True
            )
            past_z = wrapper._generate_latent_batch_impl(
                input_ids,
                attention_mask,
                latent_steps=int(args.k_small),
                past_key_values=past_z,
                role=agent.role,
                detach_debug=True,
            )
        e_zero = wrapper.teacher_force_nll(past_z, j_ids, answer_ids)
    allclose_ok = torch.allclose(e_base, e_zero, rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL)

    # Train a few steps with nonzero init
    with torch.no_grad():
        steerer.v.copy_(torch.randn_like(steerer.v) * float(args.init_std))
    opt = torch.optim.Adam([steerer.v], lr=float(args.lr))
    losses: List[float] = []
    grad_norms: List[float] = []
    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)
        energy = rollout_energy(enable_ces=True)
        loss = combined_ces_objective(energy, use_answer_nll=True)
        loss.backward()
        g = steerer.v.grad
        assert g is not None, "v.grad is None — gradients did not reach the steering vector"
        grad_norms.append(float(g.norm().item()))
        opt.step()
        losses.append(float(loss.detach().item()))
        print(f"[gate2] step={step} loss={losses[-1]:.6f} grad_norm={grad_norms[-1]:.6e}")

    return {
        "mode": "real",
        "model_name": args.model_name,
        "k_small": int(args.k_small),
        "ces_layer": layer,
        "steer_phase": "latent_only",
        "trainable_host": trainable,
        "trainable_ces": ces_trainable,
        "losses": losses,
        "grad_norms": grad_norms,
        "grad_nonzero": all(g > 0 for g in grad_norms),
        "loss_decreased": bool(len(losses) >= 2 and losses[-1] < losses[0]),
        "allclose_zero_vs_baseline": bool(allclose_ok),
        "allclose_rtol": ALLCLOSE_RTOL,
        "allclose_atol": ALLCLOSE_ATOL,
        "e_base": float(e_base.item()),
        "e_zero": float(e_zero.item()),
        "checkpointing_kv": ckpt_report,
        "passed": bool(
            all(g > 0 for g in grad_norms)
            and allclose_ok
            and len(trainable) == 0
            and ces_trainable == ["v"]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true", help="CPU toy autograd plumbing test")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--k_small", type=int, default=5)
    ap.add_argument("--ces_layer", type=int, default=20)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--init_std", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="artifacts/ces/gate2_smoke.json")
    args = ap.parse_args()

    if args.toy:
        report = _toy_smoke()
    else:
        report = _real_model_smoke(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report.get("passed", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
