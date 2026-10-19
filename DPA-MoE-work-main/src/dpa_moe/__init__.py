"""dpa_moe — encoder for DPA-MoE prefix-decodable expert weight chunks.

Encodes MoE expert FFN weights with SharedScale AnyBCQ into chunk streams
and writes the GGUF artifact consumed by the DPA-MoE llama.cpp runtime
(third_party/llama.cpp). Entry point: ``dpa-moe-encode`` (``dpa_moe.cli``).

Submodules:
    format            — MAGIC, STREAM_VERSION, flags, header struct, GGUF keys
    hp                — `HP` hyperparameter dataclass
    adapters          — per-architecture HF → GGUF maps (Qwen3-MoE, DeepSeek-MoE)
    calibrate         — ss_anybcq / anybcq block-reconstruction calibrators
    packers           — disk-byte stream packers (ss_anybcq + anybcq)
    reader            — per-layer codebook safetensors → PackedChunk
    gguf_writer       — hybrid GGUF emitter
    assemble          — fresh HF model + codebooks → GGUF
    runtime           — `set_required_runtime` byte patch (finalize phase)
    upstream_shims    — compatibility patches for the vendored AnyBCQ
    cli               — `dpa-moe-encode` entry point and variant registry

Importing this package does not import ``cli`` (which applies the AnyBCQ
shims); only the lightweight format/packing API is re-exported.
"""
from __future__ import annotations

from .format import (
    FLAG_ALPHA_FP16, FLAG_ANY_PREC, FLAG_BETA_FP16,
    HEADER_SIZE, HEADER_STRUCT, KV_PREFIX, MAGIC, STREAM_VERSION,
    alpha_dtype_for, beta_dtype_for, pack_bits_msb_first,
)
from .hp import HP
from .packers import Codebook, PackedChunk, build_anybcq_stream, build_ss_anybcq_stream

__all__ = [
    "HP",
    "MAGIC", "STREAM_VERSION", "KV_PREFIX",
    "FLAG_ALPHA_FP16", "FLAG_BETA_FP16", "FLAG_ANY_PREC",
    "HEADER_STRUCT", "HEADER_SIZE",
    "alpha_dtype_for", "beta_dtype_for", "pack_bits_msb_first",
    "Codebook", "PackedChunk",
    "build_ss_anybcq_stream", "build_anybcq_stream",
]
