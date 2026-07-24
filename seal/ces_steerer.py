"""Trainable residual-stream steerer for CES / K-budget experiments.

Unlike SealSteerer (detached MoD unit vector), this holds an nn.Parameter so
gradients can flow into v through hooked forwards. The hook always clones the
hidden-state tensor before editing to avoid unsafe in-place mutation of views.
"""

from __future__ import annotations

from typing import List, Optional, Set

import torch
import torch.nn as nn

from .hooks import _find_decoder_layers


STEER_PHASES = ("none", "latent_only", "prefill_and_latent")


class TrainableSteerer(nn.Module):
    """Learnable activation-addition direction at one decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        layer_index: int,
        *,
        coef: float = 1.0,
        apply_to: str = "last",
        init_std: float = 0.0,
        agents: Optional[Set[str]] = None,
        steer_phase: str = "latent_only",
    ) -> None:
        super().__init__()
        if steer_phase not in STEER_PHASES:
            raise ValueError(f"steer_phase must be one of {STEER_PHASES}, got {steer_phase!r}")
        self.layer_index = int(layer_index)
        self.coef = float(coef)
        self.apply_to = apply_to
        self.steer_phase = steer_phase
        self.agents: Set[str] = set(agents or {"planner", "critic", "refiner"})
        if init_std > 0:
            v = torch.randn(hidden_size) * float(init_std)
        else:
            v = torch.zeros(hidden_size)
        self.v = nn.Parameter(v)
        self._handle = None
        self._enabled = False
        self._active_role: Optional[str] = None
        self._phase: str = "latent"  # "prefill" | "latent"

    def set_active_role(self, role: Optional[str]) -> None:
        self._active_role = role

    def set_phase(self, phase: str) -> None:
        """Call with 'prefill' or 'latent' around the corresponding forwards."""
        if phase not in ("prefill", "latent"):
            raise ValueError(f"phase must be 'prefill' or 'latent', got {phase!r}")
        self._phase = phase

    def should_apply(self) -> bool:
        if not self._enabled or self.coef == 0.0:
            return False
        if self.steer_phase == "none":
            return False
        role = (self._active_role or "").lower()
        if role and role not in self.agents:
            return False
        if self.steer_phase == "prefill_and_latent":
            return True
        # latent_only
        return self._phase == "latent"

    def _hook(self, module, inputs, output):
        if not self.should_apply():
            return output
        if isinstance(output, tuple):
            hs = output[0]
            rest = tuple(output[1:])
        else:
            hs = output
            rest = None
        # Clone before edit: never mutate a view in-place.
        hs = hs.clone()
        delta = (self.coef * self.v).to(dtype=hs.dtype, device=hs.device)
        if self.apply_to == "all":
            hs = hs + delta
        else:
            hs = torch.cat([hs[:, :-1, :], hs[:, -1:, :] + delta], dim=1)
        if rest is None:
            return hs
        return (hs,) + rest

    def register(self, model) -> None:
        if self._handle is not None:
            return
        layers = _find_decoder_layers(model)
        if not (0 <= self.layer_index < len(layers)):
            raise IndexError(
                f"ces layer_index {self.layer_index} out of range "
                f"(model has {len(layers)} layers)"
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

    def trainable_parameter_names(self) -> List[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def freeze_host_model(self, model: nn.Module) -> None:
        """Freeze all host parameters; keep only this module's v trainable."""
        for p in model.parameters():
            p.requires_grad_(False)
        self.v.requires_grad_(True)

    @classmethod
    def from_artifact(cls, path: str, *, coef: float = 1.0, map_location: str = "cpu",
                      agents: Optional[Set[str]] = None, steer_phase: str = "latent_only"):
        blob = torch.load(path, map_location=map_location)
        layer_index = int(blob["layer_index"])
        vec = blob["vector"] if "vector" in blob else blob["v"]
        steerer = cls(
            hidden_size=int(vec.numel()),
            layer_index=layer_index,
            coef=coef,
            agents=agents,
            steer_phase=steer_phase,
        )
        with torch.no_grad():
            steerer.v.copy_(vec.float().view(-1))
        return steerer

    def save_artifact(self, path: str, **meta) -> None:
        blob = {
            "vector": self.v.detach().cpu().float(),
            "layer_index": self.layer_index,
            "coef": self.coef,
            "steer_phase": self.steer_phase,
            "agents": sorted(self.agents),
            **meta,
        }
        torch.save(blob, path)
