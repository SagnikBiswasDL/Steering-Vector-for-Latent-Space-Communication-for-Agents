"""Unit tests for CES steerer / losses (CPU; no large model download)."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.ces import (  # noqa: E402
    ces_rank_loss,
    combined_ces_objective,
    hinge_kl_penalty,
    kl_tokenwise,
    length_normalized_nll_from_logits,
)
from seal.ces_steerer import TrainableSteerer  # noqa: E402
from seal.hooks import SealSteerer  # noqa: E402

RTOL = 1e-5
ATOL = 1e-6


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return (self.lin(x),)


class _Host(nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(d) for _ in range(n)])


def test_hook_adds_at_layer_and_is_out_of_place():
    d = 16
    host = _Host(d=d, n=4)
    for p in host.parameters():
        p.requires_grad_(False)
    steerer = TrainableSteerer(d, layer_index=1, coef=2.0, init_std=0.0, agents={"planner"})
    with torch.no_grad():
        steerer.v.fill_(0.5)
    steerer.register(host)
    steerer.set_active_role("planner")
    steerer.set_phase("latent")
    steerer.enable()

    x = torch.randn(2, 5, d)
    # Capture pre-hook identity: run layer without steerer
    steerer.disable()
    y0 = host.model.layers[1](x)[0]
    steerer.enable()
    y1 = host.model.layers[1](x)[0]
    # Last position should differ by coef * v
    delta = y1[:, -1, :] - y0[:, -1, :]
    expected = 2.0 * steerer.v.detach()
    assert torch.allclose(delta, expected.expand_as(delta), rtol=RTOL, atol=ATOL)
    # Earlier positions unchanged
    assert torch.allclose(y1[:, :-1, :], y0[:, :-1, :], rtol=RTOL, atol=ATOL)
    steerer.remove()


def test_latent_only_skips_prefill_phase():
    d = 8
    host = _Host(d=d, n=2)
    steerer = TrainableSteerer(d, layer_index=0, coef=1.0, agents={"planner"}, steer_phase="latent_only")
    with torch.no_grad():
        steerer.v.fill_(1.0)
    steerer.register(host)
    steerer.set_active_role("planner")
    steerer.enable()

    x = torch.randn(1, 4, d)
    steerer.set_phase("prefill")
    y_pre = host.model.layers[0](x)[0]
    steerer.disable()
    y_base = host.model.layers[0](x)[0]
    assert torch.allclose(y_pre, y_base, rtol=RTOL, atol=ATOL)

    steerer.enable()
    steerer.set_phase("latent")
    y_lat = host.model.layers[0](x)[0]
    assert not torch.allclose(y_lat[:, -1, :], y_base[:, -1, :], rtol=RTOL, atol=ATOL)
    steerer.remove()


def test_gradients_reach_v_only():
    d = 8
    host = _Host(d=d, n=2)
    for p in host.parameters():
        p.requires_grad_(False)
    steerer = TrainableSteerer(d, layer_index=0, coef=1.0, init_std=0.1, agents={"planner"})
    steerer.freeze_host_model(host)
    steerer.register(host)
    steerer.set_active_role("planner")
    steerer.set_phase("latent")
    steerer.enable()

    x = torch.randn(1, 3, d)
    y = host.model.layers[0](x)[0]
    loss = y.pow(2).mean()
    loss.backward()
    assert steerer.v.grad is not None
    assert steerer.v.grad.abs().sum() > 0
    for n, p in host.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0
    steerer.remove()


def test_zero_vector_allclose_to_disabled():
    d = 8
    host = _Host(d=d, n=2)
    steerer = TrainableSteerer(d, layer_index=0, coef=1.0, agents={"planner"})
    with torch.no_grad():
        steerer.v.zero_()
    steerer.register(host)
    steerer.set_active_role("planner")
    steerer.set_phase("latent")
    x = torch.randn(1, 3, d)
    steerer.disable()
    y0 = host.model.layers[0](x)[0]
    steerer.enable()
    y1 = host.model.layers[0](x)[0]
    assert torch.allclose(y0, y1, rtol=RTOL, atol=ATOL)
    steerer.remove()


def test_seal_steerer_also_out_of_place():
    d = 8
    host = _Host(d=d, n=2)
    vec = torch.ones(d)
    steerer = SealSteerer(vec / vec.norm(), layer_index=0, coef=1.0)
    steerer.register(host)
    steerer.enable()
    x = torch.randn(1, 2, d)
    y = host.model.layers[0](x)[0]
    assert y.shape == x.shape
    steerer.remove()


def test_ces_losses():
    logits = torch.randn(2, 5, 20)
    targets = torch.randint(0, 20, (2, 5))
    e = length_normalized_nll_from_logits(logits, targets)
    assert e.ndim == 0 and torch.isfinite(e)
    e_pos = torch.tensor(1.2)
    e_neg = torch.tensor(2.0)
    rank = ces_rank_loss(e_pos, e_neg)
    assert rank < ces_rank_loss(e_neg, e_pos)
    kl = kl_tokenwise(logits, logits + torch.randn_like(logits))
    assert float(kl.item()) > 0.0
    pen = hinge_kl_penalty(torch.tensor(0.05), epsilon=0.02, lam=20.0)
    assert float(pen.item()) > 0.0
    loss = combined_ces_objective(e_pos, e_neg, kl=kl, beta_rank=1.0, lam_kl=20.0, epsilon_kl=0.02)
    assert torch.isfinite(loss)
    smoke = combined_ces_objective(e_pos, use_answer_nll=True)
    assert abs(float(smoke.item()) - float(e_pos.item())) < 1e-8


if __name__ == "__main__":
    test_hook_adds_at_layer_and_is_out_of_place()
    test_latent_only_skips_prefill_phase()
    test_gradients_reach_v_only()
    test_zero_vector_allclose_to_disabled()
    test_seal_steerer_also_out_of_place()
    test_ces_losses()
    print("all tests passed")
