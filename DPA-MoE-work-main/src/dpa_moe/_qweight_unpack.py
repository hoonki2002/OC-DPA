"""Transcode AnyBCQ `qweight` int32 layout to ss_anybcq disk bytes.

`AnyBCQ.minimize()` leaves each managed Linear as an `AnyBCQLinear`
with `qweight` shape (in/32, max_prec, out) int32 (LSB-first per-byte
sign packing). The runtime decoder consumes MSB-first per-byte signs
in (out, in/8) row-major.

  (a) `_unpack_qweight` — expands qweight into a (d1, gs, P) sign tensor
      in {-1, +1}; the encoder runs it on the GPU and packs the result
      with `packers.build_ss_anybcq_stream`.

  (b) `transcode_expert_chunk` — independent byte-level implementation
      (bit-reverse LUT + transpose, no int8 expansion). Not used by the
      encoder; it is the oracle in `_verify_transcode_parity`.

  (c) `_verify_transcode_parity` — synthetic check that (a) + packing and
      (b) produce identical bytes (run by tests/test_packing.py).

  (d) `anybcq_pack_expert_fast` — packer for the per-tier (anybcq) variant.
"""
from __future__ import annotations

import numpy as np
import torch

from .format import (
    FLAG_ALPHA_FP16, FLAG_ANY_PREC, FLAG_BETA_FP16,
    HEADER_SIZE, HEADER_STRUCT, MAGIC, STREAM_VERSION,
)
from .hp import HP
from .packers import Codebook, PackedChunk, pack_ss_anybcq_chunk


# Byte-wise bit reversal LUT (LSB-first → MSB-first within each byte).
# Used by `transcode_expert_chunk` to convert AnyBCQ's `qweight` packing
# (int32 with LSB = sign at lowest input column) into the disk byte
# stream the runtime decoder expects (each disk byte stores 8 signs
# MSB-first along the input axis).
_BITREV8_LUT: np.ndarray = np.array(
    [int(f"{i:08b}"[::-1], 2) for i in range(256)], dtype=np.uint8,
)


def _unpack_qweight(qweight: torch.Tensor, target_precision: int,
                    group_size: int) -> torch.Tensor:
    """qweight has shape (in/32, max_prec, out) int32. Unpack into a
    sign tensor of shape (d1, gs, P) with values in {-1, +1} stored as
    int8 (so memory is 1 byte/element instead of 4).

        d1   = out_features * (in_features / group_size)
        gs   = group_size
        P    = target_precision

    The d1 flattening matches the encoder's `w_grouped =
    W_padded.reshape(-1, gs)` convention (W_padded shape = (out, in)):

        d1_idx = out_idx * n_input_groups + input_group_idx
    """
    in_div_32, _max_prec, out_features = qweight.shape
    P = target_precision
    bits = []
    for p in range(P):
        plane = qweight[:, p, :]                              # (in/32, out) int32
        bit_idx = torch.arange(32, device=plane.device).view(1, 32, 1)
        unp = ((plane.unsqueeze(1) >> bit_idx) & 1).to(torch.int8)  # (in/32, 32, out)
        unp = unp.reshape(in_div_32 * 32, out_features)
        bits.append(unp)
    bits_stacked = torch.stack(bits, dim=-1)                  # (in, out, P) int8
    in_features = in_div_32 * 32
    n_input_groups = in_features // group_size
    g = bits_stacked.view(n_input_groups, group_size, out_features, P)
    g = g.permute(2, 0, 1, 3).contiguous()                    # (out, n_input_groups, gs, P)
    g = g.view(out_features * n_input_groups, group_size, P)
    return (g * 2 - 1).contiguous()


def transcode_expert_chunk(
    qweight: torch.Tensor,
    alpha_buf: torch.Tensor,
    beta_buf: torch.Tensor,
    hp: HP,
    n_out: int,
    m_in: int,
) -> PackedChunk:
    """Fast transcode: AnyBCQ `qweight` → SS_ANYBCQ disk byte stream.

    Inputs (CPU tensors, read straight out of a per-layer codebook
    safetensors written by `_save_codebooks`):

        qweight   : (in/32, max_prec, out) int32   — AnyBCQ bit-packing
        alpha_buf : (n_input_groups, max_prec, out) fp16/fp32
        beta_buf  : (n_input_groups, out)           fp16/fp32

    Produces the byte stream the runtime decoder consumes:

        [HEADER 32 B] [β block: (out, ng) fp16 row-major]
        for p in 0..P-1:
            [signs: (out, padded_m/8) bytes, MSB-along-input row-major]
            [α plane p: (out, ng) fp16 row-major]
    """
    if hp.any_precision_alpha:
        raise NotImplementedError("POC: FLAG=1 (any-precision-α) packer deferred")

    P = hp.target_precision
    gs = hp.group_size
    if m_in % gs != 0:
        raise NotImplementedError(
            f"POC requires m_in % group_size == 0 (got {m_in} % {gs})")
    n_input_groups = m_in // gs
    padded_m = m_in
    d1 = n_out * n_input_groups

    flags = FLAG_ALPHA_FP16 | FLAG_BETA_FP16
    a_size = 2

    in_div_32 = qweight.shape[0]
    if in_div_32 * 32 != padded_m or qweight.shape[2] != n_out:
        raise ValueError(
            f"qweight shape {tuple(qweight.shape)} inconsistent with "
            f"(n_out={n_out}, m_in={m_in})")

    # Header.
    data = bytearray()
    data += HEADER_STRUCT.pack(
        MAGIC, STREAM_VERSION, gs,
        hp.base_precision, hp.target_precision, flags,
        n_out, m_in, n_input_groups, d1,
    )

    FP16_MAX = 65504.0

    # β block: (n_input_groups, out) → (out, n_input_groups) fp16, row-major.
    beta_np = beta_buf.detach().to(dtype=torch.float32).cpu().numpy()
    beta_np = np.clip(beta_np, -FP16_MAX, FP16_MAX)
    beta_np = np.nan_to_num(beta_np, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
    beta_disk = np.ascontiguousarray(beta_np.T, dtype=np.float16)  # (out, ng)
    data += beta_disk.tobytes()
    beta_bytes = beta_disk.nbytes
    fixed_overhead_bytes = HEADER_SIZE + beta_bytes

    # qweight as a (in/32, P, out, 4) uint8 view — byte 0 = bits[0..7] of int32.
    qw_np = qweight.detach().cpu().numpy()
    if qw_np.dtype != np.int32 and qw_np.dtype != np.uint32:
        qw_np = qw_np.astype(np.int32, copy=False)
    qw_bytes = np.ascontiguousarray(qw_np).view(np.uint8).reshape(
        in_div_32, qw_np.shape[1], n_out, 4)

    # α: (n_input_groups, max_prec, out) → per plane (out, n_input_groups) fp16.
    alpha_np = alpha_buf.detach().to(dtype=torch.float32).cpu().numpy()
    alpha_np = np.clip(alpha_np, -FP16_MAX, FP16_MAX)
    alpha_np = np.nan_to_num(alpha_np, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
    alpha_planes_fp16 = alpha_np[:, :P, :].astype(np.float16, copy=False)

    plane_sign_bytes = n_out * (padded_m // 8)
    alpha_bytes_per_chunk = d1 * a_size
    assert plane_sign_bytes == (d1 * gs + 7) // 8

    chunk_sizes: list[int] = []
    for p in range(P):
        before = len(data)
        plane = qw_bytes[:, p, :, :]                  # (in/32, out, 4)
        plane = plane.transpose(1, 0, 2)              # (out, in/32, 4)
        plane = np.ascontiguousarray(plane)
        plane = _BITREV8_LUT[plane]                   # (out, in/32, 4) uint8
        data += plane.tobytes()
        a_disk = np.ascontiguousarray(alpha_planes_fp16[:, p, :].T)
        data += a_disk.tobytes()
        chunk_sizes.append(len(data) - before)

    return PackedChunk(
        data=bytes(data),
        fixed_overhead_bytes=fixed_overhead_bytes,
        chunk_sizes=chunk_sizes,
        plane_sign_bytes=plane_sign_bytes,
        alpha_bytes=alpha_bytes_per_chunk,
        beta_bytes=beta_bytes,
    )


def anybcq_pack_expert_fast(
    qweight: torch.Tensor,
    alphas: "list[torch.Tensor]",
    betas:  "list[torch.Tensor]",
    base_precision: int,
    target_precision: int,
    n_out: int,
    m_in: int,
    group_size: int,
) -> PackedChunk:
    """Fast FLAG_ANY_PREC=1 packer for one expert projection.

    Same contract as ``build_anybcq_stream`` but avoids the ~100 MB/expert
    float32 sign expansion.  Converts qweight int32 (LSB-first AnyBCQ
    bit-packing) → disk sign bytes via the byte-level bit-reversal LUT,
    with no intermediate float tensor.

    Args:
        qweight  : (in//32, max_prec, out) int32 on CPU
        alphas   : [A_bp, A_{bp+1}, ..., A_P], each (d1, p) float32
        betas    : [b_bp, ..., b_P], each (d1,) float32
        base_precision, target_precision : bp, P (e.g. 2, 8)
        n_out, m_in, group_size : Linear dimensions

    Returns a PackedChunk ready to store as safetensors bytes.
    """
    bp = base_precision
    P  = target_precision
    gs = group_size
    in_div_32 = qweight.shape[0]
    n_input_groups = m_in // gs
    d1 = n_out * n_input_groups
    n_snaps = P - bp + 1
    FP16_MAX = 65504.0

    flags = FLAG_ALPHA_FP16 | FLAG_BETA_FP16 | FLAG_ANY_PREC

    # ── Fast sign extraction ──────────────────────────────────────────────────
    # qweight: (in//32, P, out) int32, C-order
    # Each int32 stores 32 sign bits LSB-first.  Bit k = sign of input k
    # within the word (0 → −1, 1 → +1).  Disk format needs MSB-first bytes.
    # Strategy: view as uint8 (in//32, P, out, 4), bit-reverse each byte via
    # LUT → (P, out, in//8) MSB-first sign bytes, row-major per output row.
    qw_np = np.ascontiguousarray(qweight.detach().cpu().numpy().view(np.uint8))
    qw_np = qw_np.reshape(in_div_32, P, n_out, 4)          # (in/32, P, out, 4)
    qw_rev = _BITREV8_LUT[qw_np]                            # bit-reverse each byte
    sign_bytes_all = (
        qw_rev.transpose(1, 2, 0, 3)                        # (P, out, in/32, 4)
              .reshape(P, n_out, in_div_32 * 4)             # (P, out, in//8)
    )
    # sign_bytes_all[p, out_idx, :] = (in//8) packed sign bytes for plane p,
    # output out_idx, with MSB = sign for the lowest input column in each byte.
    plane_sign_bytes = n_out * (in_div_32 * 4)              # = n_out × in//8

    # ── Build disk stream ─────────────────────────────────────────────────────
    data = bytearray()
    data += HEADER_STRUCT.pack(
        MAGIC, STREAM_VERSION, gs, bp, P, flags,
        n_out, m_in, n_input_groups, d1,
    )

    chunk_sizes: list[int] = []
    max_alpha_bytes = 0
    for i in range(n_snaps):
        before = len(data)
        P_here = bp + i                 # precision tier: bp, bp+1, ..., P

        if i == 0:
            for p in range(P_here):     # write all bp base planes
                data += sign_bytes_all[p].tobytes()
        else:
            data += sign_bytes_all[P_here - 1].tobytes()   # one new plane

        # α_p: (d1, P_here) fp16 row-major
        a_np = alphas[i].detach().cpu().numpy().astype(np.float32)
        a_np = np.clip(np.nan_to_num(a_np, nan=0.0,
                       posinf=FP16_MAX, neginf=-FP16_MAX), -FP16_MAX, FP16_MAX)
        a_fp16 = np.ascontiguousarray(a_np, dtype=np.float16)
        data += a_fp16.tobytes()
        max_alpha_bytes = max(max_alpha_bytes, a_fp16.nbytes)

        # β_p: (d1,) fp16
        b_np = betas[i].detach().cpu().numpy().astype(np.float32)
        b_np = np.clip(np.nan_to_num(b_np, nan=0.0,
                       posinf=FP16_MAX, neginf=-FP16_MAX), -FP16_MAX, FP16_MAX)
        b_fp16 = np.ascontiguousarray(b_np, dtype=np.float16)
        data += b_fp16.tobytes()

        chunk_sizes.append(len(data) - before)

    return PackedChunk(
        data=bytes(data),
        fixed_overhead_bytes=HEADER_SIZE,
        chunk_sizes=chunk_sizes,
        plane_sign_bytes=plane_sign_bytes,
        alpha_bytes=max_alpha_bytes,
        beta_bytes=d1 * 2,             # fp16
    )


def _verify_transcode_parity(hp: HP) -> None:
    """Sanity-check the fast transcode against the slow reference path.

    Builds a tiny synthetic AnyBCQ codebook (random signs + α + β),
    runs both the slow path (`_unpack_qweight` + `pack_ss_anybcq_chunk`)
    and the new fast path (`transcode_expert_chunk`), and asserts the
    two byte streams are identical.
    """
    rng = np.random.default_rng(0)
    P = hp.target_precision
    gs = hp.group_size
    n_out, m_in = 8, 128
    n_input_groups = m_in // gs            # = 2
    in_div_32 = m_in // 32                 # = 4
    qweight = torch.from_numpy(
        rng.integers(low=-(2**31), high=2**31, size=(in_div_32, P, n_out),
                     dtype=np.int64).astype(np.int32))
    alpha_buf = torch.from_numpy(
        rng.standard_normal((n_input_groups, P, n_out)).astype(np.float32))
    beta_buf = torch.from_numpy(
        rng.standard_normal((n_input_groups, n_out)).astype(np.float32))

    fast = transcode_expert_chunk(qweight, alpha_buf, beta_buf, hp, n_out, m_in)

    # Slow reference: same path the old assemble took.
    B = _unpack_qweight(qweight, P, gs)
    a_ref = (alpha_buf.permute(2, 0, 1).contiguous()
             .view(n_out * n_input_groups, P))
    beta_ref = beta_buf.permute(1, 0).contiguous().view(n_out * n_input_groups)
    cb = Codebook(B=B, alpha=a_ref, beta=beta_ref, n=n_out, m=m_in)
    slow = pack_ss_anybcq_chunk(
        cb, base_precision=hp.base_precision,
        target_precision=hp.target_precision, group_size=hp.group_size,
    )

    if fast.data != slow.data:
        for i, (x, y) in enumerate(zip(fast.data, slow.data)):
            if x != y:
                raise RuntimeError(
                    f"transcode parity FAIL at byte {i}: fast=0x{x:02x} "
                    f"slow=0x{y:02x}  (sizes: fast={len(fast.data)} "
                    f"slow={len(slow.data)})")
        raise RuntimeError(
            f"transcode parity FAIL: size mismatch "
            f"fast={len(fast.data)} slow={len(slow.data)}")
