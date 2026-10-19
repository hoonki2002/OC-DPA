"""Load a MoE model without its expert tensors.

The experts are replaced by bit-planes, so reading the released expert weights
would waste the memory twice over -- for Qwen3-30B-A3B they are 29 of the 30
billion parameters. This builds the module tree from the config and fills in
only the tensors the analog path does not provide.
"""
from __future__ import annotations

import json, re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

EXPERT = re.compile(r"\.mlp\.(experts\.\d+|experts)\.(gate_proj|up_proj|down_proj|gate_up_proj)")


def snapshot(model_id):
    folder = Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--"))
    return sorted((folder / "snapshots").glob("*"))[-1]


def tokenizer(model_id):
    """AutoTokenizer, except for checkpoints whose config needs remote code:
    their tokenizer.json is read directly so nothing prompts for trust."""
    if "deepseek-moe" in model_id:
        import json as _json
        from transformers import PreTrainedTokenizerFast
        d = snapshot(model_id)
        conf = _json.load(open(d / "tokenizer_config.json"))
        tok_name = lambda t: t["content"] if isinstance(t, dict) else t
        return PreTrainedTokenizerFast(tokenizer_file=str(d / "tokenizer.json"),
                                       bos_token=tok_name(conf.get("bos_token")),
                                       eos_token=tok_name(conf.get("eos_token")),
                                       chat_template=conf.get("chat_template"))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def deepseek_as_qwen2moe(model_id):
    """DeepSeek-MoE (v1) expressed as a Qwen2-MoE config, plus a key remap.

    The checkpoint ships remote code written for transformers 4.x. Its
    architecture is Qwen2-MoE's minus two details, both matched exactly here:
    no QKV bias, and an ungated shared expert. Qwen2-MoE multiplies the
    shared expert by sigmoid(gate . h); with that gate's weight at zero the
    factor is exactly 0.5, so the shared down projection is loaded doubled.
    Layer 0 is a dense MLP (first_k_dense_replace = 1) -> mlp_only_layers.
    """
    from transformers import Qwen2MoeConfig
    src = json.load(open(snapshot(model_id) / "config.json"))
    dense = list(range(src["first_k_dense_replace"]))
    cfg = Qwen2MoeConfig(
        vocab_size=src["vocab_size"], hidden_size=src["hidden_size"],
        intermediate_size=src["intermediate_size"],
        moe_intermediate_size=src["moe_intermediate_size"],
        shared_expert_intermediate_size=src["moe_intermediate_size"] * src["n_shared_experts"],
        num_hidden_layers=src["num_hidden_layers"],
        num_attention_heads=src["num_attention_heads"],
        num_key_value_heads=src["num_key_value_heads"],
        num_experts=src["n_routed_experts"], num_experts_per_tok=src["num_experts_per_tok"],
        norm_topk_prob=src["norm_topk_prob"], mlp_only_layers=dense, decoder_sparse_step=1,
        rms_norm_eps=src["rms_norm_eps"], rope_theta=src.get("rope_theta", 10000.0),
        max_position_embeddings=src["max_position_embeddings"], qkv_bias=False,
        tie_word_embeddings=src.get("tie_word_embeddings", False), hidden_act=src["hidden_act"],
        bos_token_id=src.get("bos_token_id"), eos_token_id=src.get("eos_token_id"))

    def remap(key, tensor):
        if ".mlp.shared_experts." in key:
            key = key.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
            if key.endswith("down_proj.weight"):
                tensor = tensor * 2
        elif key.endswith(".mlp.gate.weight"):
            pass                                    # router: same name, same shape
        return key, tensor
    extra = {f"model.layers.{i}.mlp.shared_expert_gate.weight": torch.zeros(1, src["hidden_size"])
             for i in range(src["num_hidden_layers"]) if i not in dense}
    return cfg, remap, extra


def load(model_id, device="cuda", dtype=torch.bfloat16, patch=None):
    """patch(layer_index, experts_module) -> replacement module.

    Called before the model is moved off meta, because the expert parameters
    are never materialised and `.to(device)` refuses to copy them.
    """
    remap, extra = (lambda k, t: (k, t)), {}
    if "deepseek-moe" in model_id:
        cfg, remap, extra = deepseek_as_qwen2moe(model_id)
    else:
        cfg = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    root = snapshot(model_id)
    index = root / "model.safetensors.index.json"
    shards = (sorted({v for v in json.load(open(index))["weight_map"].values()})
              if index.exists() else ["model.safetensors"])
    state, skipped = {}, 0
    for shard in shards:
        with safe_open(root / shard, framework="pt") as handle:
            for key in handle.keys():
                if EXPERT.search(key):
                    skipped += 1
                    continue
                k2, t2 = remap(key, handle.get_tensor(key))
                state[k2] = t2.to(dtype)
    for key, tensor in extra.items():
        state[key] = tensor.to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # Expert parameters stay on meta; they are replaced before the first forward.
    left = [k for k in missing if not EXPERT.search(k)]
    if left:
        raise RuntimeError(f"non-expert tensors missing: {left[:5]}")
    if patch is not None:
        for li, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "experts"):
                replacement = patch(li, mlp.experts)
                del mlp.experts
                mlp.experts = replacement
    # Rotary inv_freq is computed, not stored, so it is still on meta; rebuild
    # it from the config rather than copying.
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is not None and any(b.is_meta for b in rotary.buffers()):
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        rope_type = rotary.rope_type
        init_fn = (rotary.compute_default_rope_parameters if rope_type == "default"
                   else ROPE_INIT_FUNCTIONS[rope_type])
        inv_freq, rotary.attention_scaling = init_fn(rotary.config)
        rotary.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
    model = model.to(device).eval()
    print(f"skeleton: {sum(p.numel() for p in model.parameters() if not p.is_meta)/1e9:.2f} B "
          f"parameters loaded, {skipped} expert tensors skipped", flush=True)
    return model, cfg


# ---------------------------------------------------------------------------
# Raw tensors of one layer, for the encoder and the residual profiler
# ---------------------------------------------------------------------------
def load_layer(model_id, layer):
    """All `model.layers.<layer>.mlp.*` tensors of an already-downloaded snapshot."""
    root = snapshot(model_id)
    index = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    pattern = re.compile(rf"^model\.layers\.{layer}\.mlp\.")
    wanted = [k for k in index if pattern.match(k)]
    if not wanted:
        raise ValueError(f"layer {layer} has no mlp tensors")
    tensors = {}
    for shard in sorted({index[k] for k in wanted}):
        with safe_open(root / shard, framework="pt") as handle:
            for key in [k for k in wanted if index[k] == shard]:
                tensors[key] = handle.get_tensor(key)
    return tensors


def moe_layers(model_id):
    """Indices of the layers that carry routed experts."""
    root = snapshot(model_id)
    index = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    found = {int(m.group(1)) for m in
             (re.match(r"^model\.layers\.(\d+)\.mlp\.experts\.", k) for k in index) if m}
    return sorted(found)
