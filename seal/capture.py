"""In-pipeline activation capture + correctness-contrastive native vectors.

This module supports the "natively-acquired" steering-vector program: instead of
extracting a steering direction from *isolated* text CoT (the earlier proxy), we
read each LatentMAS agent's **real layer-L latent activations while it runs inside
the full pipeline** (conditioned on the upstream KV cache), together with the
run's final correctness (Judger right/wrong). We then build, per agent,

    v_agent = mean(acts | final answer correct) - mean(acts | incorrect)

which is fully in-pipeline and text-free, and matches the "quality lever"
finding for the latent sub-agents.

Components:
  - ActivationRecorder: read-only forward hook on a deep decoder layer that
    records the current-token residual-stream state (the same point SEAL steers).
  - build_contrastive_vector: correct-vs-incorrect difference-of-means direction.
  - correctness_probe_auc: k-fold AUC of each agent's activation -> correctness,
    projected onto the (train-fold) contrastive direction (a 1-D linear probe).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _find_decoder_layers(model) -> List[torch.nn.Module]:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is not None:
        return list(layers)
    raise RuntimeError("Could not locate decoder layers (expected model.model.layers).")


class ActivationRecorder:
    """Read-only residual-stream recorder at a single deep decoder layer.

    While enabled, every forward call through the target layer appends the
    current-token hidden state ``hs[:, -1, :]`` (shape ``[B, D]``) to a buffer.
    This is the exact position SEAL later steers, so the captured direction is
    applied at a matched location. For a sub-agent's ``generate_latent_batch``
    this yields one vector per latent step (plus the prompt's last token from the
    initial prefill forward); for the Judger's decode it yields one vector per
    generated token.
    """

    def __init__(self, layer_index: int) -> None:
        self.layer_index = int(layer_index)
        self._handle = None
        self._enabled = False
        self.buffer: List[torch.Tensor] = []

    def _hook(self, module, inputs, output):
        if not self._enabled:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        # Record current-token state; keep on CPU in fp32 to bound GPU memory.
        self.buffer.append(hs[:, -1, :].detach().float().cpu())
        return output

    def register(self, model) -> None:
        if self._handle is not None:
            return
        layers = _find_decoder_layers(model)
        if not (0 <= self.layer_index < len(layers)):
            raise IndexError(
                f"capture layer_index {self.layer_index} out of range (model has {len(layers)} layers)"
            )
        self._handle = layers[self.layer_index].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def enable(self) -> None:
        self.buffer = []
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def pop_mean(self) -> Optional[torch.Tensor]:
        """Mean over recorded steps -> ``[B, D]`` (fp32 CPU), then clears buffer.

        Returns None if nothing was recorded.
        """
        if not self.buffer:
            return None
        stacked = torch.stack(self.buffer, dim=0)  # [steps, B, D]
        mean = stacked.mean(dim=0)                 # [B, D]
        self.buffer = []
        return mean


def build_contrastive_vector(
    acts: torch.Tensor,
    correct: torch.Tensor,
    *,
    normalize: bool = True,
) -> Dict[str, object]:
    """v = mean(acts | correct) - mean(acts | incorrect).

    Args:
        acts:    [N, D] fp32 activations (one per run for a single agent).
        correct: [N] bool/0-1 tensor of final-answer correctness.

    Returns dict with vector, unit_vector, raw_norm, n_correct, n_incorrect.
    """
    acts = acts.float()
    correct = correct.bool()
    n_correct = int(correct.sum().item())
    n_incorrect = int((~correct).sum().item())
    if n_correct == 0 or n_incorrect == 0:
        raise ValueError(
            f"Need both correct and incorrect runs (got correct={n_correct}, incorrect={n_incorrect})."
        )
    mean_correct = acts[correct].mean(dim=0)
    mean_incorrect = acts[~correct].mean(dim=0)
    vector = mean_correct - mean_incorrect
    raw_norm = vector.norm()
    unit_vector = vector / raw_norm.clamp_min(1e-8) if normalize else vector
    return {
        "vector": vector,
        "unit_vector": unit_vector,
        "raw_norm": raw_norm,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
    }


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUC via the rank (Mann-Whitney U) statistic. labels: 1=positive."""
    labels = labels.bool()
    pos = scores[labels]
    neg = scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    # Average ranks for ties.
    _, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    sums = torch.zeros(counts.numel(), dtype=torch.float64)
    sums.index_add_(0, inv, ranks)
    avg = sums / counts.double()
    ranks = avg[inv]
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    sum_pos = ranks[labels].sum().item()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def correctness_probe_auc(
    acts: torch.Tensor,
    correct: torch.Tensor,
    *,
    k: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """K-fold AUC of a 1-D linear probe (contrastive direction) -> correctness.

    In each fold we build the correct-vs-incorrect direction on the training
    split, project the held-out activations onto it, and score AUC against
    held-out correctness. Reports mean/std AUC. This directly measures how much
    each agent's in-pipeline latent state linearly determines the final outcome,
    using the same direction we steer with (so it doubles as a sanity check on
    the vector's discriminativeness).
    """
    acts = acts.float()
    correct = correct.bool()
    n = acts.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    folds = [perm[i::k] for i in range(k)]
    aucs: List[float] = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = torch.cat([folds[j] for j in range(k) if j != i])
        c_tr = correct[train_idx]
        if int(c_tr.sum()) == 0 or int((~c_tr).sum()) == 0:
            continue
        v = acts[train_idx][c_tr].mean(0) - acts[train_idx][~c_tr].mean(0)
        v = v / v.norm().clamp_min(1e-8)
        scores = acts[val_idx] @ v
        auc = _auc(scores, correct[val_idx])
        if auc == auc:  # not nan
            aucs.append(auc)
    if not aucs:
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "n_folds": 0}
    t = torch.tensor(aucs)
    return {
        "auc_mean": float(t.mean()),
        "auc_std": float(t.std(unbiased=False)),
        "n_folds": len(aucs),
    }
