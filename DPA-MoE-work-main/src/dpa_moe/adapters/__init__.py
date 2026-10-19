"""Per-model adapters for the chunked encoder.

Each adapter encapsulates the HF→GGUF tensor-name mapping, the MoE
unfusing logic (per-expert nn.Linear materialisation), and the
managed-name predicates that calibrate.py + assemble.py call into.

A new MoE model is added by:
  1. Implementing a MoEAdapter subclass (one file per model).
  2. Registering it via `register_adapter(name, adapter_cls)` in
     this package's __init__.
  3. Selecting it on the dpa_moe CLI via `--model <name>`.

The encoder pipeline itself remains model-agnostic; the runtime side
needs a matching executor in the llama.cpp fork (dpa-moe-ext/moe/).
"""
from __future__ import annotations

from .base import (
    MoEAdapter,
    ManagedID,
    canonical_gguf_name,
)
from . import qwen3 as _qwen3
from . import deepseek_moe as _deepseek_moe

_REGISTRY: dict[str, type[MoEAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[MoEAdapter]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"adapter {name!r} already registered")
    _REGISTRY[name] = adapter_cls


def get_adapter(name: str) -> MoEAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def known_adapters() -> list[str]:
    return sorted(_REGISTRY)


# Built-in registrations.
register_adapter("qwen3_moe", _qwen3.Qwen3MoEAdapter)
register_adapter("deepseek_moe", _deepseek_moe.DeepSeekMoEAdapter)


__all__ = [
    "MoEAdapter", "ManagedID", "canonical_gguf_name",
    "register_adapter", "get_adapter", "known_adapters",
]
