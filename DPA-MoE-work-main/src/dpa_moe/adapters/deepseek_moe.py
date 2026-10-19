"""DeepSeek-MoE-16B adapter (V1-era ``DeepseekForCausalLM``).

Target: ``deepseek-ai/deepseek-moe-16b-chat``.  Distinct from
DeepSeek-V2/V3 (which use MLA attention + selection bias + group
routing); upstream llama.cpp maps the V1 model to
``LLM_ARCH_DEEPSEEK`` → ``src/models/deepseek.cpp``.

DeepSeek-MoE layout (HF transformers ``DeepseekModel`` w/
``trust_remote_code=True``):

  * Per-routed-expert ``model.layers.{L}.mlp.experts.{X}.{gate,up,down}_proj``
  * Per-layer shared FFN ``model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj``
  * Per-layer router ``model.layers.{L}.mlp.gate.weight``
  * First ``first_k_dense_replace`` layers are DENSE (no MoE): they
    expose a single ``model.layers.{L}.mlp.{gate,up,down}_proj`` and
    no ``experts`` / ``gate`` siblings.

Routing: hard-coded SOFTMAX, ``norm_w=false`` (no top-k weight
renormalisation), no ``exp_probs_b`` bias, no group routing — those
are all V3 features.  ``expert_weights_scale`` is configurable.

Only the **routed** experts go through anybcq calibration.  Shared
experts + dense layers + the router itself flow through the
unmanaged-tensor pass (Q8_0 / Q4_K).  This matches the user's
directive: "for the shared experts -> don't uses anybcq -> treat as
same with the non anybcq managed thing".
"""
from __future__ import annotations

import re

import gguf
import torch.nn as nn

from .base import MoEAdapter, ManagedID


# -----------------------------------------------------------------------------
# Unfused replacement — DeepSeek's MoE block already exposes per-expert
# nn.Linear via ``self.experts: nn.ModuleList`` in the canonical HF
# class, so unfuse is largely a no-op.  We keep the entry point so the
# encoder pipeline can call ``adapter.unfuse_moe(model)`` uniformly;
# this function only confirms the per-expert ModuleList exists and
# returns the layer count.
# -----------------------------------------------------------------------------

_REL_EXPERT_RE = re.compile(
    r"(?:^|.+\.)mlp\.experts\.(\d+)\.(gate|up|down)_proj$"
)
_FQ_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$"
)

_HF_TO_GGUF: dict[str, str] = {
    "model.embed_tokens.weight":        "token_embd.weight",
    "model.norm.weight":                "output_norm.weight",
    "lm_head.weight":                   "output.weight",
}


class DeepSeekMoEAdapter(MoEAdapter):
    name = "deepseek_moe"
    dpa_moe_executor_name = "deepseek_anybcq_v1"

    def unfuse_moe(self, model,
                   layer_range: "tuple[int, int] | None" = None) -> int:
        """DeepSeek's MoE block already stores experts as nn.ModuleList;
        no fused 3-D parameter to unfold.  Confirm structure and
        return the number of MoE layers found."""
        n_done = 0
        for bid, layer in enumerate(model.model.layers):
            if layer_range is not None and not (layer_range[0] <= bid <= layer_range[1]):
                continue
            mlp = getattr(layer, "mlp", None)
            if mlp is None: continue
            experts = getattr(mlp, "experts", None)
            if experts is None: continue
            if not isinstance(experts, nn.ModuleList): continue
            # Sanity: each expert is a sub-module with the three
            # nn.Linear members our encoder expects.
            if len(experts) == 0: continue
            e0 = experts[0]
            if not (hasattr(e0, "gate_proj") and hasattr(e0, "up_proj")
                    and hasattr(e0, "down_proj")):
                raise ValueError(
                    f"DeepSeek-MoE adapter: expert[0] is missing "
                    f"gate/up/down_proj; got {type(e0).__name__}")
            n_done += 1
        return n_done

    def is_managed_linear(self, name: str) -> bool:
        return _REL_EXPERT_RE.match(name) is not None

    def parse_managed_name(self, name: str) -> ManagedID | None:
        m = _FQ_EXPERT_RE.match(name)
        if m is None: return None
        return ManagedID(bid=int(m.group(1)), xid=int(m.group(2)), kind=m.group(3))

    def transform_tensor(self, hf_name: str, tensor, *, config):
        """DeepSeek V1 uses HF's split-half ``rotate_half`` RoPE in its
        custom modeling code.  GGUF / llama.cpp ``ggml_rope`` expects
        the interleaved-pairs layout instead.  Upstream
        ``convert_hf_to_gguf.py`` permutes attn_q / attn_k accordingly
        (see DeciModel.permute at line ~3391).  Our chunked encoder
        wrote them as-is, which catastrophically corrupts attention
        (Mean KLD 5.16 on wikitext, garbage generation) — the
        per-tensor cos(BF16 ref, ss_anybcq) was 0.015 for attn_q and
        0.011 for attn_k.

        Apply the same permute so the GGUF matches the BF16 reference.
        """
        if not hf_name.endswith(("self_attn.q_proj.weight",
                                  "self_attn.k_proj.weight")):
            return tensor
        n_head = int(config.num_attention_heads)
        if hf_name.endswith("self_attn.k_proj.weight"):
            n_head = int(getattr(config, "num_key_value_heads", n_head))
        # Reshape to (n_head, 2, head_dim/2, in_features) and swap the
        # 2-axis with the head-pair axis, exactly as
        # convert_hf_to_gguf.py:DeciModel.permute does.
        w = tensor
        if w.shape[0] % (n_head * 2) != 0:
            raise ValueError(
                f"transform_tensor: {hf_name} shape {w.shape} not "
                f"divisible by n_head*2 = {n_head*2}")
        return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
                 .swapaxes(1, 2)
                 .reshape(w.shape)
                 .contiguous())

    def hf_name_to_gguf(self, hf: str, num_layers: int) -> str | None:
        if hf in _HF_TO_GGUF:
            return _HF_TO_GGUF[hf]
        m = re.match(r"^model\.layers\.(\d+)\.(.+)$", hf)
        if m is None: return None
        lid = int(m.group(1))
        rest = m.group(2)

        # Static per-layer tables.
        table = {
            "input_layernorm.weight":          f"blk.{lid}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{lid}.ffn_norm.weight",
            # Standard MHA — V1 doesn't have V2's MLA.
            "self_attn.q_proj.weight":         f"blk.{lid}.attn_q.weight",
            "self_attn.k_proj.weight":         f"blk.{lid}.attn_k.weight",
            "self_attn.v_proj.weight":         f"blk.{lid}.attn_v.weight",
            "self_attn.o_proj.weight":         f"blk.{lid}.attn_output.weight",
            # MoE router (unmanaged).
            "mlp.gate.weight":                 f"blk.{lid}.ffn_gate_inp.weight",
            # Shared-expert FFN (unmanaged — user's directive).
            "mlp.shared_experts.gate_proj.weight": f"blk.{lid}.ffn_gate_shexp.weight",
            "mlp.shared_experts.up_proj.weight":   f"blk.{lid}.ffn_up_shexp.weight",
            "mlp.shared_experts.down_proj.weight": f"blk.{lid}.ffn_down_shexp.weight",
            # First-k dense layers (first_k_dense_replace) have a
            # plain MLP instead of an MoE block.
            "mlp.gate_proj.weight":            f"blk.{lid}.ffn_gate.weight",
            "mlp.up_proj.weight":              f"blk.{lid}.ffn_up.weight",
            "mlp.down_proj.weight":            f"blk.{lid}.ffn_down.weight",
        }
        if rest in table:
            return table[rest]
        # Routed-expert nn.Linears are handled by the chunked path,
        # not this map. Filter them out so we don't double-emit.
        if rest.startswith("mlp.experts."):
            return None
        if rest.startswith("self_attn.rotary_emb"):
            return None
        return None

    def populate_gguf_metadata(self, writer, config) -> None:
        writer.add_name(getattr(config, "_name_or_path", "deepseek"))
        writer.add_context_length(int(config.max_position_embeddings))
        writer.add_embedding_length(int(config.hidden_size))
        writer.add_block_count(int(config.num_hidden_layers))
        writer.add_feed_forward_length(int(config.intermediate_size))
        writer.add_head_count(int(config.num_attention_heads))
        writer.add_head_count_kv(int(
            getattr(config, "num_key_value_heads", config.num_attention_heads)))
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is not None:
            writer.add_rope_freq_base(float(rope_theta))
        # ``deepseek.rope.dimension_count`` tells llama.cpp how many
        # head-dim positions get rotary applied (DeepSeek-V1 rotates
        # the full head_dim — partial-rope is a V2/V3 feature).
        # Missing the field defaults the loader to the wrong width and
        # produces attention outputs uncorrelated with the BF16
        # reference (Mean KLD ~5 vs ~0.05 expected).  Upstream
        # convert_hf_to_gguf.py writes it via ``add_rope_dimension_count``;
        # mirror that here.
        head_dim = int(getattr(config, "head_dim",
                                 config.hidden_size // config.num_attention_heads))
        writer.add_rope_dimension_count(head_dim)
        # Match upstream's "no rope scaling" KV so the loader doesn't
        # fall through to a (potentially mis-configured) default.
        try:
            writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        except Exception:
            writer.add_string("deepseek.rope.scaling.type", "none")
        if hasattr(config, "rms_norm_eps"):
            writer.add_layer_norm_rms_eps(float(config.rms_norm_eps))
        # DeepSeek-specific MoE knobs.
        writer.add_expert_count(int(config.n_routed_experts))
        writer.add_expert_used_count(int(config.num_experts_per_tok))
        writer.add_expert_feed_forward_length(int(config.moe_intermediate_size))
        # Expert routing knobs.  llama.cpp's LLM_ARCH_DEEPSEEK loader
        # reads ``expert_weights_scale`` and ``expert_gating_func`` to
        # decide how top-k routing weights are normalised; if the
        # ``gating_func`` field is missing the loader silently defaults
        # to 0 (NONE) and the resulting probabilities are wrong, which
        # in turn produces a huge KLD vs the BF16 reference even
        # though single-token greedy generation can still produce
        # coherent text.  DeepSeek-V1 uses SOFTMAX (=1) with
        # weights_scale = 1.0.
        try:
            writer.add_expert_weights_scale(
                float(getattr(config, "routed_scaling_factor", 1.0)))
        except AttributeError:
            writer.add_float32("deepseek.expert_weights_scale",
                                float(getattr(config, "routed_scaling_factor", 1.0)))
        # 1 = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX (DeepSeek V1).
        # 2 = SIGMOID (V2/V3 group routing).
        try:
            writer.add_expert_gating_func(1)
        except AttributeError:
            writer.add_uint32("deepseek.expert_gating_func", 1)
        # Shared-expert count is a model.config attribute; upstream
        # gguf-py exposes ``add_expert_shared_count``.
        if hasattr(config, "n_shared_experts"):
            try:
                writer.add_expert_shared_count(int(config.n_shared_experts))
            except AttributeError:
                # Older gguf-py: no such helper; write the KV manually.
                writer.add_uint32("deepseek.expert_shared_count",
                                  int(config.n_shared_experts))
        # First-k-dense (some DS variants put their first few layers as
        # plain MLPs instead of MoE).
        if hasattr(config, "first_k_dense_replace"):
            try:
                writer.add_leading_dense_block_count(
                    int(config.first_k_dense_replace))
            except AttributeError:
                writer.add_uint32("deepseek.leading_dense_block_count",
                                  int(config.first_k_dense_replace))
        if hasattr(config, "head_dim"):
            writer.add_key_length(int(config.head_dim))
            writer.add_value_length(int(config.head_dim))

    def populate_gguf_vocab(self, writer, model_dir, vocab_size) -> None:
        """DeepSeek-MoE uses the gpt2 BPE family but with a custom
        pre-tokenizer regex; llama-vocab.cpp gates the right regex on
        ``tokenizer.ggml.pre == "deepseek-llm"``.  Default
        ``populate_gpt2_bpe_vocab`` writes the token list but doesn't
        emit a pre-tokenizer KV, which silently degrades to the GPT-2
        BPE default and causes a tokenization mismatch against the
        BF16 reference (mean KLD ~5 vs ~0.005 expected).  Run the
        shared writer, then stamp the right pre-tokenizer name.
        """
        from .._vocab_default import populate_gpt2_bpe_vocab
        populate_gpt2_bpe_vocab(writer, model_dir, vocab_size)
        writer.add_tokenizer_pre("deepseek-llm")
