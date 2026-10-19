"""Chunk-stream packing: packer output, reader round trip, qweight transcode."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from dpa_moe import HEADER_SIZE, HEADER_STRUCT, HP, MAGIC, STREAM_VERSION
from dpa_moe.format import FLAG_ALPHA_FP16, FLAG_ANY_PREC, FLAG_BETA_FP16
from dpa_moe.packers import Codebook, build_ss_anybcq_stream
from dpa_moe.reader import _packed_chunks_ss_anybcq


def _pack(cb: Codebook, hp: HP):
    return build_ss_anybcq_stream(cb.B, cb.alpha, cb.beta, n=cb.n, m=cb.m,
                                  base_precision=hp.base_precision,
                                  group_size=hp.group_size)


def _random_codebook(n: int, m: int, gs: int, P: int, seed: int = 0) -> Codebook:
    g = torch.Generator().manual_seed(seed)
    d1 = n * (m // gs)
    B = torch.randint(0, 2, (d1, gs, P), generator=g).float() * 2 - 1
    alpha = torch.rand((d1, P), generator=g) + 0.1
    beta = torch.randn((d1,), generator=g)
    return Codebook(B=B, alpha=alpha, beta=beta, n=n, m=m)


@pytest.mark.parametrize("gs", [64, 128])
def test_ss_anybcq_stream_layout(gs):
    hp = HP(group_size=gs, base_precision=2, target_precision=8)
    n, m = 16, 256
    cb = _random_codebook(n, m, gs, hp.target_precision)
    pc = _pack(cb, hp)

    magic, version, group_size, base_p, target_p, flags, n_, m_, ngpr, d1 = (
        HEADER_STRUCT.unpack(pc.data[:HEADER_SIZE]))
    assert (magic, version) == (MAGIC, STREAM_VERSION)
    assert (group_size, base_p, target_p) == (gs, 2, 8)
    assert flags & FLAG_ALPHA_FP16 and flags & FLAG_BETA_FP16
    assert not flags & FLAG_ANY_PREC                      # shared-scale layout
    assert (n_, m_, ngpr, d1) == (n, m, m // gs, n * (m // gs))

    # Prefix-decodable: fixed part + one equal-size chunk per precision level.
    assert len(pc.chunk_sizes) == hp.target_precision
    assert len(set(pc.chunk_sizes)) == 1
    assert len(pc.data) == pc.fixed_overhead_bytes + sum(pc.chunk_sizes)


def test_reader_round_trip(tmp_path):
    hp = HP(group_size=128, base_precision=2, target_precision=8)
    hidden, inter = 256, 128
    streams = {
        "e000.gate": _pack(_random_codebook(inter, hidden, 128, 8, 1), hp),
        "e000.up":   _pack(_random_codebook(inter, hidden, 128, 8, 2), hp),
        "e000.down": _pack(_random_codebook(hidden, inter, 128, 8, 3), hp),
    }
    save_file({f"{k}.bytes": torch.frombuffer(bytearray(pc.data), dtype=torch.uint8)
               for k, pc in streams.items()},
              str(tmp_path / "layer_00.safetensors"))

    packed = _packed_chunks_ss_anybcq(tmp_path, hp, hidden, inter, limit_layers=None)

    assert set(packed) == {(0, 0, "gate"), (0, 0, "up"), (0, 0, "down")}
    for (_, _, kind), (pc, n_out, m_in) in packed.items():
        src = streams[f"e000.{kind}"]
        assert pc.data == src.data
        assert pc.chunk_sizes == src.chunk_sizes
        assert pc.fixed_overhead_bytes == src.fixed_overhead_bytes
        assert (n_out, m_in) == ((hidden, inter) if kind == "down" else (inter, hidden))


@pytest.mark.parametrize("gs", [32, 64, 128])
def test_qweight_transcode_parity(gs):
    """The encoder's qweight unpack + packer matches an independent transcode."""
    from dpa_moe._qweight_unpack import _verify_transcode_parity
    _verify_transcode_parity(HP(group_size=gs, base_precision=2, target_precision=8))


def test_pack_bits_msb_first():
    from dpa_moe import pack_bits_msb_first
    bits = torch.tensor([1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)
    assert pack_bits_msb_first(bits).tolist() == [0x81, 0x40]
    assert np.array_equal(pack_bits_msb_first(torch.zeros(8, dtype=torch.uint8)).numpy(), [0])
