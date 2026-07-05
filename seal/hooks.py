"""On-the-fly SEAL intervention via a forward hook on a decoder layer.

During the Judger's text decoding we add `coef * unit_vector` to the residual
stream at the current token position, at a single deep layer. Because we steer
toward execution (away from reflection/transition), this shortens the reasoning
trace and reduces token usage.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _find_decoder_layers(model) -> List[torch.nn.Module]:
    """Locate the list of decoder layers across common HF architectures."""
    # Qwen3 / Llama style: model.model.layers
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is not None:
        return list(layers)
    raise RuntimeError("Could not locate decoder layers (expected model.model.layers).")


class SealSteerer:
    """Holds a steering vector and (de)registers a residual-stream hook."""

    def __init__(
        self,
        unit_vector: Optional[torch.Tensor],
        layer_index: int,
        coef: float,
        *,
        apply_to: str = "last",  # "last" position only, or "all" positions
        role_vectors: Optional[Dict[str, torch.Tensor]] = None,
        role_coefs: Optional[Dict[str, float]] = None,
    ) -> None:
        self.unit_vector = unit_vector.detach().float() if unit_vector is not None else None
        self.layer_index = int(layer_index)
        self.coef = float(coef)
        self.apply_to = apply_to
        self._handle = None
        self._enabled = False
        # Optional per-role native vectors. When set and an active role has an
        # entry, the hook uses that role's own (vector, coef) instead of the
        # shared unit_vector/coef. This lets us steer, e.g., planner+critic+refiner
        # each with its OWN correctness-contrastive direction in a single run.
        self.role_vectors: Dict[str, torch.Tensor] = {
            r: v.detach().float() for r, v in (role_vectors or {}).items()
        }
        self.role_coefs: Dict[str, float] = dict(role_coefs or {})
        self._active_role: Optional[str] = None

    def set_active_role(self, role: Optional[str]) -> None:
        """Record which agent role is currently running (selects its vector)."""
        self._active_role = role

    def has_effect_for(self, role: Optional[str]) -> bool:
        """True if enabling for ``role`` would apply a nonzero delta."""
        vec, coef = self._resolve(role)
        return vec is not None and coef != 0.0

    def _resolve(self, role: Optional[str]):
        if role is not None and role in self.role_vectors:
            coef = self.role_coefs.get(role, self.coef)
            return self.role_vectors[role], float(coef)
        return self.unit_vector, self.coef

    @classmethod
    def from_artifact(cls, path: str, *, coef: float, layer_index: Optional[int] = None,
                      apply_to: str = "last", map_location: str = "cpu") -> "SealSteerer":
        blob = torch.load(path, map_location=map_location)
        vec = blob.get("unit_vector")
        if vec is None:
            vec = blob["vector"]
            vec = vec / vec.norm().clamp_min(1e-8)
        li = layer_index if layer_index is not None else int(blob["layer_index"])
        return cls(vec, li, coef, apply_to=apply_to)

    def _hook(self, module, inputs, output):
        if not self._enabled:
            return output
        vec, coef = self._resolve(self._active_role)
        if vec is None or coef == 0.0:
            return output
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        delta = (coef * vec).to(dtype=hs.dtype, device=hs.device)
        if self.apply_to == "all":
            hs = hs + delta
        else:  # "last": current token position
            hs[:, -1, :] = hs[:, -1, :] + delta
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def register(self, model) -> None:
        if self._handle is not None:
            return
        layers = _find_decoder_layers(model)
        if not (0 <= self.layer_index < len(layers)):
            raise IndexError(
                f"seal layer_index {self.layer_index} out of range (model has {len(layers)} layers)"
            )
        self._handle = layers[self.layer_index].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disable()
        return False
