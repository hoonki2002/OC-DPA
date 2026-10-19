"""Hybrid GGUF assembler.

Reads per-layer codebook safetensors + a fresh HF model (for the
unmanaged tensors), dispatches to the variant-specific reader, and
calls the GGUF writer.
"""
from __future__ import annotations

import pathlib
import os
import time

import torch.nn as nn

from .calibrate import _load_hf_model
from .gguf_writer import write_hybrid_gguf
from .hp import HP


def do_assemble(hf_dir: pathlib.Path, codebooks_dir: pathlib.Path,
                out_gguf: pathlib.Path, hp: HP, variant: str,
                limit_layers: int | None, required_runtime: bool,
                reader, adapter, gguf_arch: str,
                unmanaged_qtype: str = "Q8_0",
                residuals_npz: pathlib.Path | None = None):
    """Build the hybrid GGUF from per-layer safetensors + a fresh HF model
    (for the unmanaged tensors). No calibration here — pure assembly.

    `reader` is a `(codebooks_dir, hp, hidden, intermediate, limit_layers)
    -> {(bid, xid, kind): (PackedChunk, n_out, m_in)}` callable, dispatched
    by `cli._VARIANT_REGISTRY`.

    ``adapter`` selects the per-arch HF→GGUF name map + GGUF metadata
    population.
    """
    t0 = time.monotonic()
    print(f"[assemble] variant={variant} arch={gguf_arch}", flush=True)
    print(f"[assemble] loading fresh HF model {hf_dir}", flush=True)
    load_device = os.environ.get("DPA_MOE_HF_LOAD_DEVICE", "cuda:0")
    _tok, model = _load_hf_model(hf_dir, device=load_device)
    print(f"[assemble] model loaded in {time.monotonic()-t0:.1f}s", flush=True)
    if limit_layers is not None:
        model.model.layers = nn.ModuleList(list(model.model.layers)[:limit_layers])
        model.config.num_hidden_layers = limit_layers
    # Multimodal configs nest language hparams under config.text_config.
    _cfg = model.config
    if not hasattr(_cfg, "hidden_size") and hasattr(_cfg, "text_config"):
        _cfg = _cfg.text_config
    hidden = int(_cfg.hidden_size)
    intermediate = int(_cfg.moe_intermediate_size)
    t_read = time.monotonic()
    packed = reader(codebooks_dir, hp, hidden, intermediate, limit_layers)
    print(f"[assemble] read {len(packed)} disk-format chunks in "
          f"{time.monotonic()-t_read:.1f}s", flush=True)
    write_hybrid_gguf(out_gguf, model, packed, hp,
                      required_runtime=required_runtime,
                      limit_layers=limit_layers,
                      unmanaged_qtype=unmanaged_qtype,
                      adapter=adapter, gguf_arch=gguf_arch,
                      residuals_npz=residuals_npz)
    print(f"[assemble] wrote {out_gguf} "
          f"({out_gguf.stat().st_size/1e9:.2f} GB) "
          f"in {time.monotonic()-t0:.1f}s", flush=True)
