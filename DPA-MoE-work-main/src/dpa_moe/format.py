"""Chunked stream format constants — single source of truth.

These must match the llama.cpp runtime byte-for-byte: the runtime ABI
lives in `dpa-moe-ext/decoder/ss_anybcq/upstream_layout.cpp` (and
`decoder/anybcq/kernels/anybcq_format.h`). Edit both in lock-step if any
constant moves.
"""
from __future__ import annotations

import struct

import numpy as np
import torch


MAGIC: int = 0x414E5942                    # b"BYNA" little-endian == b"ANYB" big-endian
STREAM_VERSION: int = 2

FLAG_ALPHA_FP16: int = 1 << 0
FLAG_BETA_FP16:  int = 1 << 1
FLAG_ANY_PREC:   int = 1 << 2

#   layout: < magic u32 | version u8 | group_size u16 | base_p u8 | target_p u8 | flags u8 | _pad 2 | n u32 | m u32 | n_groups_per_row u32 | d1 u32 | _pad 4 >
HEADER_STRUCT = struct.Struct("<IBHBBB2xIIII4x")
HEADER_SIZE: int = HEADER_STRUCT.size      # 32
assert HEADER_SIZE == 32

KV_PREFIX = "dpa_moe."


def alpha_dtype_for(flags: int) -> np.dtype:
    return np.dtype("<f2") if flags & FLAG_ALPHA_FP16 else np.dtype("<f4")


def beta_dtype_for(flags: int) -> np.dtype:
    return np.dtype("<f2") if flags & FLAG_BETA_FP16 else np.dtype("<f4")


def pack_bits_msb_first(bits: torch.Tensor) -> torch.Tensor:
    """Pack a 1-D tensor of {0, 1} into bytes, MSB-first per byte.

    Delegates to numpy.packbits (C-implemented, ~50x the Python torch
    loop). Length must be a multiple of 8 (callers pad).
    """
    assert bits.numel() % 8 == 0
    arr = bits.detach().to("cpu", dtype=torch.uint8).numpy()
    return torch.from_numpy(np.packbits(arr, bitorder="big"))


# ─── KV-key helpers (PORTED from dpa_moe.runtime.format) ────────────────────
def k_canonical_shape(name: str) -> str:
    return f"{KV_PREFIX}tensor.{name}.canonical_shape"
def k_shape(name: str) -> str:
    return f"{KV_PREFIX}tensor.{name}.shape"
def k_chunk_bytes(name: str) -> str:
    return f"{KV_PREFIX}tensor.{name}.chunk_bytes"
def k_fixed_bytes(name: str) -> str:
    return f"{KV_PREFIX}tensor.{name}.fixed_bytes"
def k_d1(name: str) -> str:
    return f"{KV_PREFIX}tensor.{name}.d1"
def k_n_groups_per_row(name: str) -> str:
    return f"{KV_PREFIX}tensor.{name}.n_groups_per_row"
def k_padded_m(name: str) -> str:
    return f"{KV_PREFIX}tensor.{name}.padded_m"

def bundle_tensor_name(layer: int, expert: int) -> str:
    return f"{KV_PREFIX}bundle.blk.{layer}.e{expert}"

def k_bundle_tensors() -> str:
    return f"{KV_PREFIX}bundle_tensors"

def k_bundle_wids(name: str) -> str:
    return f"{KV_PREFIX}bundle.{name}.wids"

def k_bundle_cids(name: str) -> str:
    return f"{KV_PREFIX}bundle.{name}.cids"

def k_bundle_offsets(name: str) -> str:
    return f"{KV_PREFIX}bundle.{name}.offsets"

def k_bundle_sizes(name: str) -> str:
    return f"{KV_PREFIX}bundle.{name}.sizes"

def k_bundle_format(name: str) -> str:
    return f"{KV_PREFIX}bundle.{name}.format"
# ─── Whole-model expert-output residuals (gate²·‖ΔE_e‖²) ──────────────────────
# Packed once per arch under ``dpa_moe.expert_residuals.*`` so the
# runtime can recompute a static-layout binary at any chosen budget
# without an external ``expert_residuals.npz``.  The values are written
# during assemble from the npz produced by
# ``experiments/3_dynamic_precision/profiling/profile_residuals.py``.
K_EXPERT_R          = f"{KV_PREFIX}expert_residuals.R"             # float32 array, len = L · E · n_K
K_EXPERT_N          = f"{KV_PREFIX}expert_residuals.N"             # int64 array,   len = L · E
K_EXPERT_R_LAYERS   = f"{KV_PREFIX}expert_residuals.n_layers"      # uint32
K_EXPERT_R_EXPERTS  = f"{KV_PREFIX}expert_residuals.n_experts"     # uint32
K_EXPERT_R_K_MIN    = f"{KV_PREFIX}expert_residuals.K_min"         # uint32
K_EXPERT_R_K_MAX    = f"{KV_PREFIX}expert_residuals.K_max"         # uint32
K_EXPERT_R_METRIC   = f"{KV_PREFIX}expert_residuals.metric"        # string
