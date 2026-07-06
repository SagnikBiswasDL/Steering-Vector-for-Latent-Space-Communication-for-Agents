"""One-shot KV-cache steering for the LatentMAS handoff.

Reference: "KV Cache Steering for Controlling Frozen LLMs", Belitsky et al.,
arXiv:2507.08799.

Unlike SEAL (which adds `coef * v` to the residual stream at a single layer on
*every* decode step), cache steering makes a single, one-shot edit to the
key/value cache *after* it is populated, across *all* layers:

    K*_l = K_l + c_k * S_k_l        V*_l = V_l + c_v * S_v_l

applied at a chosen subset of cache positions. Because the cached K/V of past
tokens are terminal (nothing re-propagates them through the network), editing
all layers at once does not compound, so it is stable and adds ~zero latency.

In LatentMAS the Planner/Critic/Refiner never emit text; their entire output is
the K/V they write into the shared cache that the Judger consumes. This module
edits that handoff cache right before the Judger decodes, so it steers the only
channel by which the latent agents influence the answer.

Position modes (relative to the handoff cache the Judger reads):
  - "handoff_last":  the final cache column (Refiner's last latent thought).
  - "handoff_lastk": the final ``last_k`` columns.
  - "handoff_all":   every handoff column.
  - "judger_token":  control arm; steer the Judger's own last prompt token
                     instead of the handoff (handled during decoding).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

HANDOFF_MODES = ("handoff_last", "handoff_lastk", "handoff_all")
POSITION_MODES = HANDOFF_MODES + ("judger_token",)


def iter_layer_kv(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return per-layer ``(key, value)`` tensors from a KV cache, in place.

    Handles the transformers>=4.54 ``Cache`` layout (``cache.layers[l].keys``),
    the older ``key_cache``/``value_cache`` lists, and legacy tuple-of-tuples
    caches. The returned tensors are the live cache tensors (shape
    ``[B, H_kv, T, D_h]``), so index-assignment edits them in place.
    """
    if cache is None:
        return []
    # transformers >= 4.54: DynamicCache.layers -> DynamicLayer(.keys, .values)
    layers = getattr(cache, "layers", None)
    if layers is not None and len(layers) > 0 and hasattr(layers[0], "keys"):
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in layers:
            if layer is None or layer.keys is None or layer.values is None:
                continue
            out.append((layer.keys, layer.values))
        return out
    # older Cache objects exposing parallel lists
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return [(k, v) for k, v in zip(key_cache, value_cache) if k is not None and v is not None]
    # legacy tuple/list of (key, value) per layer
    if isinstance(cache, (tuple, list)):
        out = []
        for layer in cache:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                out.append((layer[0], layer[1]))
        return out
    raise TypeError(f"Unsupported KV cache type for steering: {type(cache)!r}")


def cache_seq_length(cache) -> int:
    kv = iter_layer_kv(cache)
    if not kv:
        return 0
    return int(kv[0][0].shape[-2])


class KVCacheSteerer:
    """Holds per-layer K/V steering directions and applies a one-shot cache edit."""

    def __init__(
        self,
        keys: Dict[int, torch.Tensor],
        values: Dict[int, torch.Tensor],
        *,
        c_v: float,
        c_k: float = 0.0,
        positions: str = "handoff_last",
        last_k: int = 40,
    ) -> None:
        if positions not in POSITION_MODES:
            raise ValueError(
                f"positions must be one of {POSITION_MODES}, got {positions!r}"
            )
        self.keys: Dict[int, torch.Tensor] = {
            int(l): t.detach().float() for l, t in (keys or {}).items()
        }
        self.values: Dict[int, torch.Tensor] = {
            int(l): t.detach().float() for l, t in (values or {}).items()
        }
        self.c_v = float(c_v)
        self.c_k = float(c_k)
        self.positions = positions
        self.last_k = int(last_k)

    @classmethod
    def from_artifact(
        cls,
        path: str,
        *,
        c_v: float,
        c_k: float = 0.0,
        positions: str = "handoff_last",
        last_k: int = 40,
        map_location: str = "cpu",
    ) -> "KVCacheSteerer":
        blob = torch.load(path, map_location=map_location)
        keys = blob.get("keys", {})
        values = blob.get("values", {})
        return cls(keys, values, c_v=c_v, c_k=c_k, positions=positions, last_k=last_k)

    @property
    def has_effect(self) -> bool:
        if self.c_v != 0.0 and len(self.values) > 0:
            return True
        if self.c_k != 0.0 and len(self.keys) > 0:
            return True
        return False

    def target_positions(self, seq_len: int) -> List[int]:
        """Absolute cache columns to steer, for a handoff cache of length ``seq_len``."""
        if seq_len <= 0:
            return []
        if self.positions == "handoff_last":
            return [seq_len - 1]
        if self.positions == "handoff_lastk":
            start = max(0, seq_len - max(1, self.last_k))
            return list(range(start, seq_len))
        if self.positions == "handoff_all":
            return list(range(0, seq_len))
        # judger_token is applied elsewhere (during decoding), not on the handoff
        return []

    @torch.no_grad()
    def apply(self, cache, positions: Sequence[int]) -> int:
        """Add the steering directions into ``cache`` at ``positions`` (all layers).

        Returns the number of layers edited. In-place; ``cache`` is mutated.
        """
        if not positions or not self.has_effect:
            return 0
        pos = list(positions)
        layer_kv = iter_layer_kv(cache)
        n_edited = 0
        for layer_idx, (K, V) in enumerate(layer_kv):
            edited = False
            if self.c_v != 0.0 and layer_idx in self.values:
                sv = self.values[layer_idx].to(dtype=V.dtype, device=V.device)
                # [H_kv, D_h] -> [1, H_kv, 1, D_h] broadcasts over batch + positions
                V[:, :, pos, :] += (self.c_v * sv).unsqueeze(0).unsqueeze(2)
                edited = True
            if self.c_k != 0.0 and layer_idx in self.keys:
                sk = self.keys[layer_idx].to(dtype=K.dtype, device=K.device)
                K[:, :, pos, :] += (self.c_k * sk).unsqueeze(0).unsqueeze(2)
                edited = True
            if edited:
                n_edited += 1
        return n_edited

    @torch.no_grad()
    def apply_to_handoff(self, cache, handoff_len: Optional[int] = None) -> int:
        """Steer the handoff cache at the configured handoff positions."""
        if self.positions not in HANDOFF_MODES:
            return 0
        seq_len = handoff_len if handoff_len is not None else cache_seq_length(cache)
        return self.apply(cache, self.target_positions(seq_len))

    def summary(self) -> str:
        return (
            f"positions={self.positions} c_v={self.c_v} c_k={self.c_k} "
            f"last_k={self.last_k} layers={len(self.values)}"
        )
