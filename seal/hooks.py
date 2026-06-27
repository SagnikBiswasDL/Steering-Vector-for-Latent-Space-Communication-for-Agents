"""On-the-fly SEAL intervention via a forward hook on a decoder layer.

During the Judger's text decoding we add `coef * unit_vector` to the residual
stream at the current token position, at a single deep layer. Because we steer
toward execution (away from reflection/transition), this shortens the reasoning
trace and reduces token usage.
"""

from __future__ import annotations

from typing import List, Optional

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
        unit_vector: torch.Tensor,
        layer_index: int,
        coef: float,
        *,
        apply_to: str = "last",  # "last" position only, or "all" positions
    ) -> None:
        self.unit_vector = unit_vector.detach().float()
        self.layer_index = int(layer_index)
        self.coef = float(coef)
        self.apply_to = apply_to
        self._handle = None
        self._enabled = False

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
        if not self._enabled or self.coef == 0.0:
            return output
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        delta = (self.coef * self.unit_vector).to(dtype=hs.dtype, device=hs.device)
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
