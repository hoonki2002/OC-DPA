"""Disk-byte stream packers for AnyBCQ chunks.

Two variants, both returning :class:`PackedChunk`:

  * `build_ss_anybcq_stream`  — FLAG_ANY_PREC=0; one shared α set
    trained at target precision. Header + β + per-plane (signs ‖ α)×P.

  * `build_anybcq_stream`     — FLAG_ANY_PREC=1; per-precision α/β
    snapshots. Header + per-precision (signs ‖ α ‖ β) chunks.

PORTED from `dpa_moe/encoders/impl/anybcq.py:_build_ss_anybcq_stream`
and `_build_any_prec_stream`, adapted to return :class:`PackedChunk`
(not the dpa_moe-specific `CompressedTensor`).
"""
from __future__ import annotations

import dataclasses
from typing import Iterable

import torch

from .format import (
    FLAG_ALPHA_FP16, FLAG_ANY_PREC, FLAG_BETA_FP16,
    HEADER_SIZE, HEADER_STRUCT, MAGIC, STREAM_VERSION,
    alpha_dtype_for, beta_dtype_for, pack_bits_msb_first,
)


@dataclasses.dataclass(frozen=True)
class Codebook:
    """Per-tensor AnyBCQ codebook (matrix-wise output)."""
    B: torch.Tensor     # (d1, gs, P) float32 in {-1, +1}
    alpha: torch.Tensor # (d1, P) float32
    beta: torch.Tensor  # (d1,) float32
    n: int              # out_features
    m: int              # in_features (≤ padded_m)


@dataclasses.dataclass(frozen=True)
class PackedChunk:
    """On-disk byte stream for one managed tensor."""
    data: bytes
    fixed_overhead_bytes: int       # header + β (the per-tensor fixed-meta region)
    chunk_sizes: list[int]          # per-chunk byte sizes (sum + fixed = len(data))
    plane_sign_bytes: int
    alpha_bytes: int                # per chunk
    beta_bytes: int                 # in the fixed-meta region


def build_ss_anybcq_stream(
    B: torch.Tensor, Alpha: torch.Tensor, beta: torch.Tensor,
    *, n: int, m: int, base_precision: int, group_size: int,
) -> PackedChunk:
    """Build FLAG_ANY_PREC=0 disk bytes for one tensor.

    Byte layout:

        offset 0      : HEADER (32 B)
        offset 32     : β block — d1 × β_dtype_size bytes
        offset 32+βb  : chunk 0 = [plane-0 signs ‖ plane-0 α]
        ...           : chunk p = [plane-p signs ‖ plane-p α]
        ...           : up to chunk (P-1)
    """
    flags = FLAG_ALPHA_FP16 | FLAG_BETA_FP16
    a_dtype = alpha_dtype_for(flags)
    b_dtype = beta_dtype_for(flags)
    gs = group_size
    d1, gs_, P = B.shape
    assert gs_ == gs, (gs_, gs)
    n_groups_per_row = -(-m // gs)

    data = bytearray()
    data += HEADER_STRUCT.pack(
        MAGIC, STREAM_VERSION, gs, base_precision, P, flags,
        n, m, n_groups_per_row, d1,
    )
    beta_np = beta.detach().cpu().numpy().astype(b_dtype)
    data += beta_np.tobytes()
    beta_bytes = beta_np.nbytes
    fixed_overhead_bytes = HEADER_SIZE + beta_bytes

    plane_sign_bytes = (d1 * gs + 7) // 8
    alpha_bytes_per_chunk = d1 * a_dtype.itemsize

    # NAVER convention: bit=1 → sign=+1. Pack signs on-device before
    # the D2H copy: one (d1*gs)-element bit-pack per plane gives 8x
    # less PCIe traffic than copying a uint8 mask.
    B_signs_bool = (B >= 0)
    alpha_np = Alpha.detach().cpu().numpy().astype(a_dtype)

    chunk_sizes: list[int] = []
    for p in range(P):
        before = len(data)
        plane_bits = B_signs_bool[:, :, p].reshape(-1).to(torch.uint8)
        if plane_bits.numel() % 8 != 0:
            pad = 8 - (plane_bits.numel() % 8)
            plane_bits = torch.cat([
                plane_bits, torch.zeros(pad, dtype=torch.uint8, device=plane_bits.device)
            ])
        data += bytes(pack_bits_msb_first(plane_bits).numpy())
        data += alpha_np[:, p].tobytes()
        chunk_sizes.append(len(data) - before)

    return PackedChunk(
        data=bytes(data),
        fixed_overhead_bytes=fixed_overhead_bytes,
        chunk_sizes=chunk_sizes,
        plane_sign_bytes=plane_sign_bytes,
        alpha_bytes=alpha_bytes_per_chunk,
        beta_bytes=beta_bytes,
    )


def pack_ss_anybcq_chunk(cb: Codebook, *, base_precision: int, target_precision: int,
                         group_size: int) -> PackedChunk:
    """Thin convenience wrapper over `build_ss_anybcq_stream` that takes
    a :class:`Codebook` instead of raw (B, Alpha, beta).
    """
    assert cb.B.shape[2] == target_precision, (cb.B.shape, target_precision)
    return build_ss_anybcq_stream(
        cb.B, cb.alpha, cb.beta,
        n=cb.n, m=cb.m,
        base_precision=base_precision, group_size=group_size,
    )


def build_anybcq_stream(
    snaps: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *, n: int, m: int, base_precision: int, target_precision: int,
    group_size: int,
) -> PackedChunk:
    """Build FLAG_ANY_PREC=1 disk bytes (per-precision α / β tables).

    Each chunk-i (i = 0..target − base) carries everything needed at
    precision p = base_p + i:
        [signs new at this level]   (base_p planes for chunk 0)
        [α^(p): d1 × p × α-dtype]
        [β^(p): d1 × β-dtype]
    α/β are NOT additive — consumer uses chunk-(k-base_p)'s α and β at
    chosen precision k.
    """
    flags = FLAG_ALPHA_FP16 | FLAG_BETA_FP16 | FLAG_ANY_PREC
    a_dtype = alpha_dtype_for(flags)
    b_dtype = beta_dtype_for(flags)
    gs = group_size
    snaps = list(snaps)
    bp = base_precision
    tp = target_precision
    if len(snaps) != tp - bp + 1:
        raise RuntimeError(
            f"any-prec snapshot count {len(snaps)} != precisions {tp - bp + 1}")

    B0 = snaps[-1][0]
    d1 = B0.shape[0]
    n_groups_per_row = -(-m // gs)

    data = bytearray()
    data += HEADER_STRUCT.pack(
        MAGIC, STREAM_VERSION, gs, bp, tp, flags,
        n, m, n_groups_per_row, d1,
    )
    fixed_overhead_bytes = HEADER_SIZE       # header-only; no fixed_meta

    plane_sign_bytes = (d1 * gs + 7) // 8
    chunk_sizes: list[int] = []
    alpha_bytes_per_chunk = 0
    beta_bytes_per_chunk = d1 * b_dtype.itemsize
    for i, (B_p, A_p, b_p) in enumerate(snaps):
        before = len(data)
        P_here = bp + i
        B_bits = (B_p >= 0)
        alpha_np = A_p.detach().cpu().numpy().astype(a_dtype)
        beta_np = b_p.detach().cpu().numpy().astype(b_dtype)
        if i == 0:
            for p in range(P_here):
                plane_bits = B_bits[:, :, p].reshape(-1).to(torch.uint8)
                if plane_bits.numel() % 8 != 0:
                    pad = 8 - (plane_bits.numel() % 8)
                    plane_bits = torch.cat([
                        plane_bits, torch.zeros(pad, dtype=torch.uint8, device=plane_bits.device)
                    ])
                data += bytes(pack_bits_msb_first(plane_bits).numpy())
        else:
            plane_bits = B_bits[:, :, P_here - 1].reshape(-1).to(torch.uint8)
            if plane_bits.numel() % 8 != 0:
                pad = 8 - (plane_bits.numel() % 8)
                plane_bits = torch.cat([
                    plane_bits, torch.zeros(pad, dtype=torch.uint8, device=plane_bits.device)
                ])
            data += bytes(pack_bits_msb_first(plane_bits).numpy())
        data += alpha_np.tobytes()
        data += beta_np.tobytes()
        chunk_sizes.append(len(data) - before)
        alpha_bytes_per_chunk = max(alpha_bytes_per_chunk, alpha_np.nbytes)

    return PackedChunk(
        data=bytes(data),
        fixed_overhead_bytes=fixed_overhead_bytes,
        chunk_sizes=chunk_sizes,
        plane_sign_bytes=plane_sign_bytes,
        alpha_bytes=alpha_bytes_per_chunk,
        beta_bytes=beta_bytes_per_chunk,
    )
