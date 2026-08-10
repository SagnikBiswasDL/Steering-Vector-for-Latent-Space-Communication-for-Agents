"""CPU unit tests for seal.relay_compress (no CUDA required)."""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.relay_compress import (  # noqa: E402
    RelayCompressor,
    num_positions,
    kv_mb,
    to_legacy,
    from_legacy,
)


def _fake_cache(layers=4, heads=8, seq=200, dim=128, dtype=torch.float32):
    legacy = []
    for _ in range(layers):
        k = torch.randn(1, heads, seq, dim, dtype=dtype)
        v = torch.randn(1, heads, seq, dim, dtype=dtype)
        legacy.append((k, v))
    return from_legacy(legacy)


def test_full_is_identity_shape():
    past = _fake_cache(seq=200)
    c = RelayCompressor(mode="full")
    out, st = c.compress(past)
    assert num_positions(out) == 200
    assert st.positions_out == 200
    assert abs(st.ratio - 1.0) < 1e-6


def test_evict_shrinks_and_keeps_sink():
    past = _fake_cache(seq=200)
    c = RelayCompressor(mode="evict", budget=32, sink=4, importance="key_norm")
    out, st = c.compress(past)
    assert num_positions(out) == 32, st.positions_out
    assert st.sink_retained is True
    assert st.mb_out < st.mb_in
    assert st.positions_out < st.positions_in


def test_obf_adds_rank_positions():
    past = _fake_cache(seq=200)
    c = RelayCompressor(mode="obf", budget=32, sink=4, rank=8)
    out, st = c.compress(past)
    # kept budget (32) + rank backfill (8) = 40 positions
    assert num_positions(out) == 40, st.positions_out
    assert st.sink_retained is True
    assert st.mb_out < st.mb_in


def test_recency_selects_last_positions():
    past = _fake_cache(seq=50)
    c = RelayCompressor(mode="evict", budget=10, sink=2, importance="recency")
    out, st = c.compress(past)
    assert num_positions(out) == 10


def test_budget_ge_seq_is_noop_positions():
    past = _fake_cache(seq=16)
    c = RelayCompressor(mode="evict", budget=64, sink=4)
    out, st = c.compress(past)
    assert num_positions(out) == 16  # cannot keep more than exist


def test_no_nans_and_dtype_preserved():
    past = _fake_cache(seq=120, dtype=torch.float16)
    for mode in ("full", "evict", "obf"):
        out, st = RelayCompressor(mode=mode, budget=32, sink=4, rank=8).compress(past)
        for k, v in to_legacy(out):
            assert torch.isfinite(k.float()).all()
            assert torch.isfinite(v.float()).all()
            assert k.dtype == torch.float16 and v.dtype == torch.float16


def test_memory_ratio_reported():
    past = _fake_cache(seq=1000)
    _, st = RelayCompressor(mode="obf", budget=32, sink=4, rank=8).compress(past)
    # 40/1000 positions -> ~4% of KV bytes
    assert st.ratio < 0.1
    print(f"[obf] {st.positions_in}->{st.positions_out} pos, "
          f"{st.mb_in:.2f}->{st.mb_out:.2f} MB, ratio={st.ratio:.3f}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
