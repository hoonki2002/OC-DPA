"""Qwen3-MoE adapter.

Carries forward what used to live in ``model.py`` for the original
Qwen3-30B-A3B pipeline:

  * HF MoE class detection (``Qwen3MoeExperts``) and unfusing into a
    ModuleList of per-expert ``Qwen3MoeMLP``.
  * Managed-tensor predicates (regex over ``mlp.experts.experts.N.K_proj``).
  * Unmanaged HF→GGUF name map for Qwen3 attention/router/norms.
  * GGUF metadata population (``arch=qwen3moe``).

All Qwen3-specific knowledge now lives in this file; adding a new
MoE model means writing a sibling ``adapters/<arch>.py`` with the
same surface.
"""
from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

from .base import MoEAdapter, ManagedID


# -----------------------------------------------------------------------------
# Lazy HF class import
# -----------------------------------------------------------------------------
def _qwen3_classes():
    """Lazy import — avoids pulling transformers at module load."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeExperts, Qwen3MoeMLP,
    )
    return Qwen3MoeExperts, Qwen3MoeMLP


# -----------------------------------------------------------------------------
# Unfused replacement module
# -----------------------------------------------------------------------------
class Qwen3MoeExpertsUnfused(nn.Module):
    """Drop-in replacement for Qwen3MoeExperts with per-expert nn.Linear.

    Forward signature + numerical behaviour match the fused version.
    The split is ``gate_up_proj[i].chunk(2, dim=0)`` → (gate_w, up_w),
    matching the runtime split at ``modeling_qwen3_moe.py:245``.
    """

    def __init__(self, fused, config):
        super().__init__()
        _, Qwen3MoeMLP = _qwen3_classes()
        self.num_experts = fused.num_experts
        self.hidden_dim = fused.hidden_dim
        self.intermediate_dim = fused.intermediate_dim
        src_dtype = fused.gate_up_proj.dtype
        src_device = fused.gate_up_proj.device
        self.experts = nn.ModuleList([
            Qwen3MoeMLP(config, intermediate_size=fused.intermediate_dim)
            for _ in range(self.num_experts)
        ])
        self.experts.to(dtype=src_dtype)
        with torch.no_grad():
            gate_up = fused.gate_up_proj.to(device="cpu")
            down = fused.down_proj.to(device="cpu")
            for i, mlp in enumerate(self.experts):
                gate_w, up_w = gate_up[i].chunk(2, dim=0)
                mlp.gate_proj.weight.data = gate_w.contiguous().to(dtype=src_dtype)
                mlp.up_proj.weight.data = up_w.contiguous().to(dtype=src_dtype)
                mlp.down_proj.weight.data = down[i].contiguous().to(dtype=src_dtype)
        self.to(device=src_device)
        self.act_fn = fused.act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        device_jobs: dict[torch.device, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for expert_idx in expert_hit:
            xid = int(expert_idx[0].item())
            if xid == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[xid])
            mlp = self.experts[xid]
            try:
                expert_device = next(mlp.parameters()).device
            except StopIteration:
                expert_device = hidden_states.device
            device_jobs[expert_device].append((xid, top_k_pos, token_idx))

        def _run_device(device, jobs):
            device_results = []
            with torch.cuda.device(device):
                for xid, top_k_pos, token_idx in jobs:
                    mlp = self.experts[xid]
                    current_state = hidden_states[token_idx].to(device, non_blocking=True)
                    weights = top_k_weights[token_idx, top_k_pos, None].to(device, non_blocking=True)
                    out = mlp.down_proj(
                        self.act_fn(mlp.gate_proj(current_state)) * mlp.up_proj(current_state)
                    )
                    out = out * weights
                    device_results.append((token_idx, out))
            return device_results

        results = []
        if len(device_jobs) > 1:
            with ThreadPoolExecutor(max_workers=len(device_jobs)) as pool:
                futures = [
                    pool.submit(_run_device, device, jobs)
                    for device, jobs in device_jobs.items()
                ]
                for fut in futures:
                    results.extend(fut.result())
        else:
            for device, jobs in device_jobs.items():
                results.extend(_run_device(device, jobs))

        for token_idx, out in results:
            final_hidden_states.index_add_(
                0, token_idx,
                out.to(device=final_hidden_states.device,
                       dtype=final_hidden_states.dtype),
            )
        return final_hidden_states


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------
# Two name forms reach our predicates:
#   (a) fully-qualified, when walking model.named_modules() after
#       AnyBCQ.minimize() returns:
#           model.layers.<bid>.mlp.experts.experts.<xid>.<kind>_proj
#   (b) block-relative, while upstream's swap_quant_model walks one
#       decoder layer at a time during minimize():
#           mlp.experts.experts.<xid>.<kind>_proj
_REL_EXPERT_RE = re.compile(
    r"(?:^|.+\.)mlp\.experts\.experts\.(\d+)\.(gate|up|down)_proj$"
)
_FQ_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.experts\.(\d+)\.(gate|up|down)_proj$"
)

_HF_TO_GGUF: dict[str, str] = {
    "model.embed_tokens.weight":        "token_embd.weight",
    "model.norm.weight":                "output_norm.weight",
    "lm_head.weight":                   "output.weight",
}


class Qwen3MoEAdapter(MoEAdapter):
    name = "qwen3_moe"
    dpa_moe_executor_name = "qwen3_ss_anybcq_v1"

    def unfuse_moe(self, model,
                   layer_range: "tuple[int, int] | None" = None) -> int:
        Qwen3MoeExperts, _ = _qwen3_classes()
        n_done = 0
        for bid, layer in enumerate(model.model.layers):
            if layer_range is not None and not (layer_range[0] <= bid <= layer_range[1]):
                continue
            if not hasattr(layer.mlp, "experts"):
                continue
            if isinstance(layer.mlp.experts, Qwen3MoeExperts):
                layer.mlp.experts = Qwen3MoeExpertsUnfused(layer.mlp.experts, model.config)
                n_done += 1
        return n_done

    def is_managed_linear(self, name: str) -> bool:
        return _REL_EXPERT_RE.match(name) is not None

    def parse_managed_name(self, name: str) -> ManagedID | None:
        m = _FQ_EXPERT_RE.match(name)
        if m is None:
            return None
        return ManagedID(bid=int(m.group(1)), xid=int(m.group(2)), kind=m.group(3))

    def hf_name_to_gguf(self, hf: str, num_layers: int) -> str | None:
        if hf in _HF_TO_GGUF:
            return _HF_TO_GGUF[hf]
        m = re.match(r"^model\.layers\.(\d+)\.(.+)$", hf)
        if m is None:
            return None
        lid = int(m.group(1))
        rest = m.group(2)
        table = {
            "input_layernorm.weight":               f"blk.{lid}.attn_norm.weight",
            "post_attention_layernorm.weight":      f"blk.{lid}.ffn_norm.weight",
            "self_attn.q_proj.weight":              f"blk.{lid}.attn_q.weight",
            "self_attn.k_proj.weight":              f"blk.{lid}.attn_k.weight",
            "self_attn.v_proj.weight":              f"blk.{lid}.attn_v.weight",
            "self_attn.o_proj.weight":              f"blk.{lid}.attn_output.weight",
            "self_attn.q_norm.weight":              f"blk.{lid}.attn_q_norm.weight",
            "self_attn.k_norm.weight":              f"blk.{lid}.attn_k_norm.weight",
            "mlp.gate.weight":                      f"blk.{lid}.ffn_gate_inp.weight",
        }
        if rest in table:
            return table[rest]
        if ".experts.experts." in f"model.layers.{lid}.{rest}":
            return None
        if "experts.experts." in rest:
            return None
        if rest.startswith("self_attn.rotary_emb"):
            return None        # llama.cpp recomputes
        return None

    def populate_gguf_metadata(self, writer, config) -> None:
        """Emit Qwen3-MoE arch + hparams.  Mirrors the original
        ``_populate_qwen3_metadata`` from gguf_writer.py."""
        writer.add_name(getattr(config, "_name_or_path", "qwen3moe"))
        writer.add_context_length(int(config.max_position_embeddings))
        writer.add_embedding_length(int(config.hidden_size))
        writer.add_block_count(int(config.num_hidden_layers))
        writer.add_feed_forward_length(int(config.intermediate_size))
        writer.add_head_count(int(config.num_attention_heads))
        writer.add_head_count_kv(int(
            getattr(config, "num_key_value_heads", config.num_attention_heads)))
        # Operator-precedence bug: the previous form was
        #   (getattr(...) or (config.rope_parameters or {}).get(...)) if
        #   hasattr(config, "rope_parameters") else None
        # which evaluated to None whenever ``rope_parameters`` was absent
        # (transformers 5.x Qwen3-MoE), so no ``rope_freq_base`` was emitted
        # and the runtime fell back to 10_000.  Qwen3-30B-A3B's actual
        # rope_theta is 1_000_000 — a 100× scrambling of RoPE that produces
        # KLD ≈ 0.30 vs BF16 (the very symptom we're chasing).  Unconditionally
        # try top-level first, then ``rope_parameters`` dict.
        rope_theta = (
            getattr(config, "rope_theta", None)
            or (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
        )
        if rope_theta is not None:
            writer.add_rope_freq_base(float(rope_theta))
        if hasattr(config, "rms_norm_eps"):
            writer.add_layer_norm_rms_eps(float(config.rms_norm_eps))
        writer.add_expert_count(int(config.num_experts))
        writer.add_expert_used_count(int(config.num_experts_per_tok))
        writer.add_expert_feed_forward_length(int(config.moe_intermediate_size))
        if hasattr(config, "head_dim"):
            writer.add_key_length(int(config.head_dim))
            writer.add_value_length(int(config.head_dim))
            # Without ``rope.dimension_count`` the runtime defaults to a
            # wrong value and Qwen3-MoE attention produces garbage (cos
            # 0.43 vs HF post-attn-norm, KLD 0.30 vs 0.005 expected).
            # convert_hf_to_gguf.py writes this via Qwen2MoeModel base;
            # the chunked encoder must mirror it.
            writer.add_rope_dimension_count(int(config.head_dim))
