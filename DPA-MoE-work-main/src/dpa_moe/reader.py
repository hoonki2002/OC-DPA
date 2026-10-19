"""Readers for per-layer codebook safetensors → PackedChunk dict.

After calibrate writes one safetensors per decoder block (with keys
`e{xid:03d}.{kind}.bytes`), assemble reads them back via these
functions and feeds the result into the GGUF writer.

The bytes-on-disk are already in the runtime-readable ss_anybcq layout
(header + β + (signs‖α)×P); we just need to recompute the chunk-size
metadata from `hp` + tensor dims (deterministic).
"""
from __future__ import annotations

import pathlib


from .format import (
    HEADER_SIZE,
)
from .hp import HP
from .packers import PackedChunk


def _packed_chunks_ss_anybcq(codebooks_dir: pathlib.Path, hp: HP,
                              hidden: int, intermediate: int,
                              limit_layers: int | None):
    """Read per-layer safetensors holding raw ss_anybcq disk-format bytes.

    Returns `{(bid, xid, kind): (PackedChunk, n_out, m_in)}`.
    """
    from safetensors import safe_open
    out: dict[tuple[int, int, str], tuple[PackedChunk, int, int]] = {}
    layer_files = sorted(codebooks_dir.glob("layer_*.safetensors"))
    if not layer_files:
        raise RuntimeError(f"no layer_*.safetensors found in {codebooks_dir}")
    P = hp.target_precision
    gs = hp.group_size
    for f in layer_files:
        bid = int(f.stem.split("_")[1])
        if limit_layers is not None and bid >= limit_layers:
            continue
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in list(st.keys()):
                head, kind, buf = key.split(".", 2)
                if buf != "bytes":
                    continue
                xid = int(head[1:])
                data = bytes(st.get_tensor(key).numpy())
                if kind == "down":
                    n_out, m_in = hidden, intermediate
                else:
                    n_out, m_in = intermediate, hidden
                n_input_groups = -(-m_in // gs)
                d1 = n_out * n_input_groups
                plane_sign_bytes = (d1 * gs + 7) // 8
                alpha_bytes = d1 * 2          # FLAG_ALPHA_FP16
                beta_bytes  = d1 * 2          # FLAG_BETA_FP16
                fixed_bytes = HEADER_SIZE + beta_bytes
                chunk_size  = plane_sign_bytes + alpha_bytes
                expected    = fixed_bytes + P * chunk_size
                if len(data) != expected:
                    raise RuntimeError(
                        f"layer {bid} e{xid:03d}.{kind}: got {len(data)} "
                        f"bytes, expected {expected} "
                        f"(d1={d1}, gs={gs}, P={P})")
                mid = (bid, xid, kind)
                pc = PackedChunk(
                    data=data,
                    fixed_overhead_bytes=fixed_bytes,
                    chunk_sizes=[chunk_size] * P,
                    plane_sign_bytes=plane_sign_bytes,
                    alpha_bytes=alpha_bytes,
                    beta_bytes=beta_bytes,
                )
                out[mid] = (pc, n_out, m_in)
    return out


def _packed_chunks_anybcq(codebooks_dir: pathlib.Path, hp: HP,
                           hidden: int, intermediate: int,
                           limit_layers: int | None):
    """Read per-layer safetensors holding raw anybcq disk-format bytes.

    Returns `{(bid, xid, kind): (PackedChunk, n_out, m_in)}`.

    Chunk-size layout (FLAG_ANY_PREC=1, fixed_overhead = HEADER_SIZE only):
        chunk 0 (precision = bp):
            bp × plane_sign_bytes  + d1 × bp × 2  + d1 × 2
        chunk i > 0 (precision = bp + i):
            plane_sign_bytes       + d1 × (bp+i) × 2  + d1 × 2
    where plane_sign_bytes = (d1 × gs + 7) // 8 (one plane's bits).
    """
    from safetensors import safe_open
    out: dict[tuple[int, int, str], tuple[PackedChunk, int, int]] = {}
    layer_files = sorted(codebooks_dir.glob("layer_*.safetensors"))
    if not layer_files:
        raise RuntimeError(f"no layer_*.safetensors found in {codebooks_dir}")
    bp = hp.base_precision
    P  = hp.target_precision
    gs = hp.group_size
    a_size = 2  # fp16
    b_size = 2  # fp16
    for f in layer_files:
        bid = int(f.stem.split("_")[1])
        if limit_layers is not None and bid >= limit_layers:
            continue
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in list(st.keys()):
                head, kind, buf = key.split(".", 2)
                if buf != "bytes":
                    continue
                xid = int(head[1:])
                data = bytes(st.get_tensor(key).numpy())
                if kind == "down":
                    n_out, m_in = hidden, intermediate
                else:
                    n_out, m_in = intermediate, hidden
                n_input_groups = -(-m_in // gs)
                d1 = n_out * n_input_groups
                plane_sign_bytes = (d1 * gs + 7) // 8
                chunk_sizes = (
                    [bp * plane_sign_bytes + d1 * bp * a_size + d1 * b_size]
                    + [plane_sign_bytes + d1 * (bp + i) * a_size + d1 * b_size
                       for i in range(1, P - bp + 1)]
                )
                fixed_bytes = HEADER_SIZE
                expected = fixed_bytes + sum(chunk_sizes)
                if len(data) != expected:
                    raise RuntimeError(
                        f"layer {bid} e{xid:03d}.{kind}: got {len(data)} "
                        f"bytes, expected {expected} "
                        f"(d1={d1}, gs={gs}, bp={bp}, P={P})")
                pc = PackedChunk(
                    data=data,
                    fixed_overhead_bytes=fixed_bytes,
                    chunk_sizes=chunk_sizes,
                    plane_sign_bytes=plane_sign_bytes,
                    alpha_bytes=d1 * P * a_size,
                    beta_bytes=d1 * b_size,
                )
                out[(bid, xid, kind)] = (pc, n_out, m_in)
    return out
