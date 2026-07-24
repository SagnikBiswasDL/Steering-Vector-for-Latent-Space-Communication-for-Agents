"""Contrastive Energy-Based Steering (CES) losses for K-budget experiments.

Adapted from Azizi et al., ACL 2026 Findings (ASC). We do not claim a faithful
ASC reproduction for CoT compression; the ranking / KL forms are reused for
latent-compute recovery.

Main training targets (after Gate 2/3 smoke):
  - L_rank: softplus(E(y+) - E(y-)) on full-budget success vs small-budget failure
  - optional distillation of full-budget Judger token distribution
  - KL on actual low-budget Planner/Critic/Refiner contexts (+ optional general text)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def length_normalized_nll_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean per-token NLL (energy) over non-ignored positions.

    logits: [B, T, V] predicting targets at the same positions (already aligned),
    or [B, T, V] for causal LM where caller has shifted appropriately.
    target_ids: [B, T]
    """
    if logits.dim() != 3 or target_ids.dim() != 2:
        raise ValueError("logits must be [B,T,V] and target_ids [B,T]")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # Gather token log-probs
    tok_lp = log_probs.gather(-1, target_ids.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = target_ids.ne(ignore_index)
    # Also ignore pad-like zeros if caller used pad as ignore; clamp gather used 0
    # only when ignore — restore those positions via mask.
    tok_lp = tok_lp * mask
    denom = mask.sum().clamp_min(1).to(tok_lp.dtype)
    energy = -tok_lp.sum() / denom
    return energy


def ces_rank_loss(energy_pos: torch.Tensor, energy_neg: torch.Tensor) -> torch.Tensor:
    """softplus(E(y+) - E(y-)); prefers lower energy on the positive trajectory."""
    return F.softplus(energy_pos - energy_neg)


def kl_tokenwise(
    logits_ref: torch.Tensor,
    logits_steered: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean tokenwise KL(ref || steered) over masked positions.

    logits_*: [B, T, V]
    mask: [B, T] bool, optional
    """
    p = F.log_softmax(logits_ref.float(), dim=-1)
    q = F.log_softmax(logits_steered.float(), dim=-1)
    # KL(p||q) = sum_v p * (log p - log q)
    probs = p.exp()
    kl = (probs * (p - q)).sum(dim=-1)  # [B, T]
    if mask is None:
        return kl.mean()
    mask_f = mask.to(kl.dtype)
    return (kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def hinge_kl_penalty(kl: torch.Tensor, *, epsilon: float, lam: float) -> torch.Tensor:
    """λ * max(KL - ε, 0)."""
    return float(lam) * torch.relu(kl - float(epsilon))


def combined_ces_objective(
    energy_pos: torch.Tensor,
    energy_neg: Optional[torch.Tensor] = None,
    *,
    kl: Optional[torch.Tensor] = None,
    beta_rank: float = 1.0,
    lam_kl: float = 0.0,
    epsilon_kl: float = 2e-2,
    use_answer_nll: bool = False,
) -> torch.Tensor:
    """Compose smoke / main objectives.

    - use_answer_nll=True: return energy_pos only (Gate 2/3 smoke).
    - else: beta * softplus(E+ - E-) + hinge KL.
    """
    if use_answer_nll or energy_neg is None:
        loss = energy_pos
    else:
        loss = float(beta_rank) * ces_rank_loss(energy_pos, energy_neg)
    if kl is not None and lam_kl > 0:
        loss = loss + hinge_kl_penalty(kl, epsilon=epsilon_kl, lam=lam_kl)
    return loss
