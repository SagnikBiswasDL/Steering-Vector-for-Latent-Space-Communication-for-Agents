"""Build the SEAL steering vector from labeled hidden states.

The steering direction we want for *token efficiency* is the one that moves the
residual stream AWAY from reflection/transition thoughts and TOWARD execution
thoughts:

    v = mean(execution) - mean(reflection + transition)

Applying `hidden += coef * v` with coef > 0 therefore suppresses redundant
reflection/transition generation, shortening the reasoning trace. (This is the
sign convention used throughout the repo; a negative coef would do the opposite.)
"""

from __future__ import annotations

from typing import Dict, List

import torch


def build_steering_vector(
    hidden_by_type: Dict[str, List[torch.Tensor]],
    *,
    normalize: bool = True,
) -> Dict[str, object]:
    """Compute the steering vector from per-type hidden-state samples.

    Args:
        hidden_by_type: dict mapping "execution"/"reflection"/"transition" to a
            list of 1-D hidden-state tensors (each shape [D]).
        normalize: if True, also return a unit-norm version of the vector.

    Returns:
        dict with keys: vector (raw diff [D]), unit_vector ([D]), raw_norm,
        counts (per-type sample counts).
    """

    def _mean(tensors: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([t.float() for t in tensors], dim=0)
        return stacked.mean(dim=0)

    counts = {k: len(v) for k, v in hidden_by_type.items()}
    if not hidden_by_type.get("execution"):
        raise ValueError("Need at least one execution-thought sample.")

    exec_mean = _mean(hidden_by_type["execution"])

    reflect_trans: List[torch.Tensor] = []
    reflect_trans += hidden_by_type.get("reflection", [])
    reflect_trans += hidden_by_type.get("transition", [])
    if not reflect_trans:
        raise ValueError("Need at least one reflection/transition sample.")
    rt_mean = _mean(reflect_trans)

    vector = exec_mean - rt_mean
    raw_norm = vector.norm()
    unit_vector = vector / raw_norm.clamp_min(1e-8) if normalize else vector

    return {
        "vector": vector,
        "unit_vector": unit_vector,
        "raw_norm": raw_norm,
        "counts": counts,
    }
