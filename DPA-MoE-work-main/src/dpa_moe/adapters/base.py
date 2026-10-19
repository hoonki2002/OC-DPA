"""MoEAdapter protocol — the per-model surface dpa_moe calls into.

Three responsibilities per adapter:

  1. **Unfuse expert weights**.  HF MoE classes typically store all
     experts as a single 3-D tensor (`Qwen3MoeExperts.gate_up_proj`,
     `DeepseekV2MoE.experts.W2[i]`, etc.).  AnyBCQ.minimize wants
     per-Linear modules — we materialise them per-adapter.

  2. **Managed-tensor predicates**.  ``is_managed_linear`` decides
     whether a named nn.Linear should be routed through anybcq
     calibration.  ``parse_managed_name`` extracts (bid, xid, kind)
     for the canonical GGUF tensor name.

  3. **HF→GGUF tensor-name map**.  Unmanaged tensors (attention,
     embeddings, norms, router, shared experts) get a per-arch
     mapping into the GGUF naming scheme that upstream llama.cpp's
     model graph builder expects.

The canonical-name function for managed experts is shared across
adapters because every supported model uses the same upstream
GGUF convention: ``blk.{bid}.ffn_{kind}_exps.weight`` plus
``:e{xid}`` suffixes appended by dpa_moe.
"""
from __future__ import annotations

import abc
import dataclasses
from typing import Protocol, runtime_checkable

import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class ManagedID:
    bid: int    # block / layer index
    xid: int    # expert index
    kind: str   # "gate" | "up" | "down"


def canonical_gguf_name(mid: ManagedID) -> str:
    """Shared across all MoE adapters — every upstream arch uses
    this same canonical GGUF tensor naming for routed experts."""
    return f"blk.{mid.bid}.ffn_{mid.kind}_exps.weight"


@runtime_checkable
class MoEAdapter(Protocol):
    """Per-model surface the encoder pipeline calls into.

    Concrete adapters live in this package and are registered in
    ``adapters/__init__.py``.  The encoder selects one from the
    experiment config's ``model.adapter`` key.
    """

    name: str   # short identifier, matches the registry key

    # ── dpa_moe runtime executor name ───────────────────────────────
    # Recorded under dpa_moe.executor in the encoded GGUF.  The
    # dpa-moe-ext runtime looks this up in its executor registry at
    # install_for_gguf time.  Each adapter overrides with its
    # arch-specific name.
    dpa_moe_executor_name: str = "qwen3_ss_anybcq_v1"

    # ── Model structure ───────────────────────────────────────────────
    def get_layers(self, model) -> "nn.ModuleList":
        """Return the decoder ``ModuleList`` from an HF model.

        Default: ``model.model.layers`` (Qwen3-MoE, DeepSeek-MoE,
        plain LLaMA-style stacks).  Override for multimodal wrappers
        whose decoder lives elsewhere (e.g.
        ``model.model.language_model.layers``).
        """
        return model.model.layers

    # ── HF MoE class detection + unfusing ─────────────────────────────
    @abc.abstractmethod
    def unfuse_moe(self, model,
                   layer_range: "tuple[int, int] | None" = None) -> int:
        """Replace fused MoE blocks with per-expert ``nn.ModuleList[nn.Linear]``
        siblings so AnyBCQ.minimize sees per-Linear modules.  Returns the
        number of layers unfused.  Idempotent.

        If ``layer_range=(start, end)`` is given (inclusive), only layers
        in [start, end] are unfused; the rest remain fused and still produce
        correct activations during the calibration forward pass.  This is
        correct for AdaQuant-style (FP_CACHING=1) parallel partitioning
        because each range is calibrated independently.
        """
        ...

    # ── Managed-tensor predicates ─────────────────────────────────────
    @abc.abstractmethod
    def is_managed_linear(self, name: str) -> bool:
        """True iff ``name`` (block-relative OR fully-qualified) is a
        per-expert {gate,up,down}_proj nn.Linear that the encoder
        should calibrate through anybcq."""
        ...

    @abc.abstractmethod
    def parse_managed_name(self, name: str) -> ManagedID | None:
        """Extract (bid, xid, kind) from a fully-qualified managed
        nn.Linear name.  Return None if the name doesn't match this
        arch's managed-tensor pattern."""
        ...

    # ── HF → GGUF name map for UNMANAGED tensors ──────────────────────
    @abc.abstractmethod
    def hf_name_to_gguf(self, hf: str, num_layers: int) -> str | None:
        """Map an HF parameter name to its GGUF tensor name for the
        UNMANAGED pass (attention, embeddings, norms, router, shared
        experts).  Return None for names we intentionally drop (e.g.
        rotary inverse-freq cached buffers).  Managed expert names
        return None too — they're handled separately via packers +
        canonical_gguf_name above."""
        ...

    # ── GGUF metadata population ──────────────────────────────────────
    @abc.abstractmethod
    def populate_gguf_metadata(self, writer, config) -> None:
        """Write arch-specific KV metadata to the GGUFWriter.  Includes
        ``add_name``, ``add_block_count``, expert counts, head dims,
        and the ``arch`` field (which upstream llama.cpp dispatches
        the model class on)."""
        ...

    # ── HF → GGUF tensor value transform ──────────────────────────────
    def transform_tensor(self, hf_name, tensor, *, config):
        """Hook for arch-specific tensor reshuffling done by upstream
        convert_hf_to_gguf.py before writing.

        Default is identity — most arches store weights in the layout
        ggml expects.  DeepSeek-V1's custom modeling uses HF's
        split-half RoPE (``rotate_half``); GGUF expects the
        interleaved-pairs layout, so attn_q / attn_k need permutation.
        Adapters override this method to apply the permute (or any
        other write-time reshape).
        """
        return tensor

    # ── GGUF vocab population ─────────────────────────────────────────
    def populate_gguf_vocab(self, writer, model_dir, vocab_size) -> None:
        """Write tokenizer KV metadata to the GGUFWriter.

        Default implementation matches the Qwen3-MoE recipe: BPE-style
        ``tokenizer.ggml.model = "gpt2"`` with ``SpecialVocab(load_merges
        =True)``.  Adapters whose model requires a different tokenizer
        model string (so llama.cpp picks up the correct pre-tokenizer)
        override this method.
        """
        from .._vocab_default import populate_gpt2_bpe_vocab
        populate_gpt2_bpe_vocab(writer, model_dir, vocab_size)
