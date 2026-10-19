"""Hybrid GGUF writer.

Emits one GGUF containing:
  - Standard Qwen3-MoE metadata (arch=qwen3moe, layer count, etc.)
  - Standard floating tensors for unmanaged weights (attn, lm_head,
    token_embd, norms, router).
  - Per-managed-expert chunked tensors under
    `dpa_moe.bytes.<canonical>:e<xid>` (raw_dtype=I8).
  - Canonical placeholders (`<canonical>` as a 1-element f16=0) for
    every managed group.
  - Per-expert KV metadata under `dpa_moe.tensor.<canonical>:e<xid>.*`.
  - Global KV under `dpa_moe.*` including `managed_tensors`,
    `executor`, and `required_runtime`.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import torch

from .format import (
    KV_PREFIX,
    K_EXPERT_R, K_EXPERT_N, K_EXPERT_R_LAYERS, K_EXPERT_R_EXPERTS,
    K_EXPERT_R_K_MIN, K_EXPERT_R_K_MAX, K_EXPERT_R_METRIC,
    bundle_tensor_name, k_bundle_cids, k_bundle_format, k_bundle_offsets,
    k_bundle_sizes, k_bundle_tensors, k_bundle_wids,
    k_canonical_shape, k_chunk_bytes, k_d1, k_fixed_bytes,
    k_n_groups_per_row, k_padded_m, k_shape,
)
from .hp import HP
from .adapters import ManagedID, canonical_gguf_name
from .packers import PackedChunk


def _ss_anybcq_chunk_to_kernel(pc: PackedChunk, cid: int, *,
                               n: int, m: int, group_size: int) -> bytes:
    """Mirror ss_anybcq::plane_disk_to_kernel for offline bundle payloads."""
    ng = -(-m // group_size)
    padded_m = ng * group_size
    k_over_32 = padded_m // 32
    qw = np.zeros((k_over_32, n), dtype="<u4")

    off = pc.fixed_overhead_bytes + sum(pc.chunk_sizes[:cid])
    chunk = memoryview(pc.data)[off:off + pc.chunk_sizes[cid]]
    sign_nbytes = n * (padded_m // 8)
    signs = np.frombuffer(chunk[:sign_nbytes], dtype=np.uint8).reshape(
        n, padded_m // 8)
    bits = np.unpackbits(signs, axis=1, bitorder="big").reshape(n, k_over_32, 32)
    weights = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    qw[:, :] = (bits.astype(np.uint32) * weights).sum(axis=2).T

    alpha = np.frombuffer(chunk[sign_nbytes:], dtype="<f2").reshape(n, ng)
    return qw.tobytes() + np.ascontiguousarray(alpha.T).tobytes()


def _populate_qwen3_metadata(writer, config) -> None:
    """Emit the standard Qwen3 / Qwen3-MoE GGUF metadata keys from a HF
    config.
    """
    writer.add_name(getattr(config, "_name_or_path", "qwen3moe"))
    writer.add_context_length(int(config.max_position_embeddings))
    writer.add_embedding_length(int(config.hidden_size))
    writer.add_block_count(int(config.num_hidden_layers))
    writer.add_feed_forward_length(int(config.intermediate_size))
    writer.add_head_count(int(config.num_attention_heads))
    writer.add_head_count_kv(int(
        getattr(config, "num_key_value_heads", config.num_attention_heads)))
    # Qwen3-MoE stores rope under `rope_parameters` dict in newer
    # transformers; legacy attribute name was `rope_theta`.
    rope_theta = (
        getattr(config, "rope_theta", None)
        or (config.rope_parameters or {}).get("rope_theta") if hasattr(config, "rope_parameters") else None
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


def write_hybrid_gguf(
    out_path: pathlib.Path,
    model,
    packed: dict[tuple[int, int, str], tuple[PackedChunk, int, int]],
    hp: HP,
    *,
    required_runtime: bool,
    limit_layers: int | None = None,
    unmanaged_qtype: str = "Q8_0",
    adapter,
    gguf_arch: str,
    residuals_npz: pathlib.Path | None = None,
):
    """Phase 1 GGUF emitter.

    `unmanaged_qtype` selects the inline ggml type used for unmanaged
    matmul tensors (attention Q/K/V/O, lm_head, ffn_gate_inp). Norms
    stay F32; token_embd stays F16 regardless (llama.cpp embed-lookup
    constraint).  Supported: `"Q8_0"` (default), `"Q4_K_M"`.

    ``adapter`` is the per-arch MoEAdapter (``dpa_moe.adapters``) and
    ``gguf_arch`` the matching llama.cpp architecture string.
    """
    from gguf import GGUFWriter, GGMLQuantizationType
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(out_path), arch=gguf_arch, use_temp_file=True)
    adapter.populate_gguf_metadata(writer, model.config)
    # Resolve text-side config (nested text_config for multimodal models).
    _text_cfg = model.config
    if not hasattr(_text_cfg, "hidden_size") and hasattr(_text_cfg, "text_config"):
        _text_cfg = _text_cfg.text_config
    adapter.populate_gguf_vocab(
        writer,
        model.config._name_or_path or pathlib.Path(out_path).parent,
        vocab_size=int(getattr(_text_cfg, "vocab_size", 0)
                       or getattr(model.config, "vocab_size", 0) or 0) or None,
    )

    # Multimodal configs nest language hparams under config.text_config;
    # other arches keep them at the top level.  Resolve once.
    _cfg = model.config
    if not hasattr(_cfg, "hidden_size") and hasattr(_cfg, "text_config"):
        _cfg = _cfg.text_config
    num_layers = int(_cfg.num_hidden_layers)
    # Per-arch expert count: Qwen3-MoE → ``num_experts``;
    # DeepSeek-MoE → ``n_routed_experts``; Mixtral/Gemma 4 may use
    # ``num_local_experts``.  Try each in priority order.
    _n_experts = (
        getattr(_cfg, "num_experts", None)
        or getattr(_cfg, "n_routed_experts", None)
        or getattr(_cfg, "num_local_experts", None)
    )
    if _n_experts is None:
        raise AttributeError(
            "gguf_writer: model.config has no expert-count attribute "
            "(checked: num_experts, n_routed_experts, num_local_experts)")
    n_experts = int(_n_experts)
    hidden = int(_cfg.hidden_size)
    intermediate = int(_cfg.moe_intermediate_size)

    managed_names: list[str] = []

    # Walk HF state dict for unmanaged tensors.
    sd = model.state_dict()
    for hf_name, tensor in sd.items():
        if ".mlp.experts.experts." in hf_name:
            continue
        gguf_name = adapter.hf_name_to_gguf(hf_name, num_layers)
        if gguf_name is None:
            continue
        # Per-arch HF → GGUF tensor reshape (e.g. RoPE Q/K permute on
        # DeepSeek V1).  Default is identity for most archs.
        tensor = adapter.transform_tensor(hf_name, tensor, config=_text_cfg)
        # Norms come in three flavors:
        #   blk.X.<name>_norm.weight          (most arches)
        #   blk.X.<name>_norm_1.weight        (Gemma 4 MoE: parallel-dense post-norm)
        #   blk.X.<name>_norm_2.weight        (Gemma 4 MoE: MoE pre/post norms)
        # All must stay F32 (ggml-cuda assertion on src1 of ggml_mul).
        is_norm = bool(re.search(r"_norm(?:_\d+)?\.weight$", gguf_name))
        # token_embd kept as F16 — llama.cpp embedding lookup mismatches on
        # Q8_0 token tables. Norms stay F32 (ggml-cuda assertion).
        is_embd_or_norm = is_norm or gguf_name == "token_embd.weight"
        # Small / scalar tensors (e.g. Gemma 4's per-layer
        # ``layer_out_scale`` shape (1,)) can't go through Q8_0 — it
        # requires at least one row's worth of values.  Keep them F32
        # to preserve the scalar exactly.
        is_scalar = tensor.numel() < 64
        # Scale tensors (Gemma 4's ``ffn_gate_inp.scale``,
        # ``ffn_down_exps.scale``, attention scaling tables, etc.)
        # are consumed as src1 in ggml_mul which requires F32/F16.
        # Q8_0 trips ``binbcast.cu``'s dtype assertion.  Keep them F32.
        is_scale = gguf_name.endswith(".scale") or "_s.weight" in gguf_name \
            or gguf_name.endswith("_scale.weight")
        # Router projection ``blk.X.ffn_gate_inp.weight`` is the MoE
        # router's [n_embd, n_expert] matmul output.  Upstream
        # convert_hf_to_gguf.py keeps this F32; quantising to Q8_0
        # introduces ~1/256 noise per logit which is catastrophic for
        # narrow expert distributions (DeepSeek-MoE-16B's 64-expert
        # top-6 router: Mean KLD 5.16 with Q8_0 router vs ~0.05
        # expected with F32 — Q8_0 noise flips top-k expert selections
        # on long contexts).  Keep routers in F32 to match BF16
        # behaviour and avoid scoring divergence.
        is_router = bool(re.search(r"\.ffn_gate_inp\.weight$", gguf_name))
        # LM head ``output.weight`` falls through to the inline-quantize
        # branch (Q8_0 by default — see ``unmanaged_qtype``).  Earlier
        # builds kept it F32 (+0.6 GB/token) so the KLD measurement
        # isolated MoE-quality noise from logit-quantization noise;
        # for the production bytes/token comparison we want lm_head
        # at the same precision as the Q8_0 baseline.  Quantizing here
        # adds ~1/256 noise per logit, which empirically lifts Mean KLD
        # by a small constant offset — acceptable for the bytes/token
        # axis.  Set ``unmanaged_qtype=F32`` to recover the old behaviour.
        is_lm_head = False    # handled by the default inline-quantize path
        # Shared-expert FFN tensors (DeepSeek-MoE-16B's ``ffn_*_shexp``)
        # are evaluated UNCONDITIONALLY for every token — their
        # contribution isn't scaled by routing weights, so quantisation
        # noise here propagates with full weight per layer.  Qwen3 has
        # no shared experts so its Q8_0 unmanaged path is fine; DeepSeek
        # at Q8_0 shared experts likely explains the catastrophic Mean
        # KLD = 5.16 vs the BF16 reference (which keeps these BF16).
        # Pin shared-expert weights to F16 to match the BF16 ref's
        # quality envelope.  Re-encoded artifacts are larger but the
        # correctness gain outweighs the size cost.
        is_shexp = bool(re.search(r"\.ffn_(gate|up|down)_shexp\.weight$",
                                    gguf_name))
        m = re.match(r"^blk\.(\d+)\.", gguf_name)
        if limit_layers is not None and m and int(m.group(1)) >= limit_layers:
            continue
        if is_norm or is_scale or is_router or is_lm_head:
            arr = tensor.detach().to(dtype=torch.float32).contiguous().cpu().numpy()
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        elif is_embd_or_norm or is_shexp:
            arr = tensor.detach().to(dtype=torch.float16).contiguous().cpu().numpy()
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F16)
        elif is_scalar:
            arr = tensor.detach().to(dtype=torch.float32).contiguous().cpu().numpy()
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            # Inline-quantize attention / lm_head / ffn_gate_inp. Sidesteps
            # llama-quantize's validator NaN bug on the managed-canonical
            # placeholders. Type is configurable via `unmanaged_qtype`.
            arr = tensor.detach().to(dtype=torch.float32).contiguous().cpu().numpy()
            from gguf import quantize as _gguf_quant
            # Python `gguf` lib implements only legacy types (Q4_0,
            # Q4_1, Q5_0, Q5_1, Q8_0, BF16). K-quants raise
            # NotImplementedError. Stick to legacy types here; if real
            # K-quant is needed, route through llama-quantize externally.
            _qtype_map = {
                "Q8_0": GGMLQuantizationType.Q8_0,
                "Q5_1": GGMLQuantizationType.Q5_1,
                "Q5_0": GGMLQuantizationType.Q5_0,
                "Q4_1": GGMLQuantizationType.Q4_1,
                "Q4_0": GGMLQuantizationType.Q4_0,
            }
            if unmanaged_qtype not in _qtype_map:
                raise ValueError(
                    f"unmanaged_qtype={unmanaged_qtype!r} not in {sorted(_qtype_map)}")
            ggml_type = _qtype_map[unmanaged_qtype]
            q = _gguf_quant(arr, qtype=ggml_type)
            writer.add_tensor(gguf_name, q, raw_dtype=ggml_type)

    # Managed experts: flush per (bid, kind) group when all xid arrive.
    accum: dict[tuple[int, str], dict[int, tuple[PackedChunk, int, int]]] = {}
    for (bid, xid, kind), entry in packed.items():
        if limit_layers is not None and bid >= limit_layers:
            continue
        accum.setdefault((bid, kind), {})[xid] = entry

    for (bid, kind), per_expert in sorted(accum.items()):
        if len(per_expert) != n_experts:
            raise RuntimeError(
                f"incomplete MoE group blk.{bid}.ffn_{kind}_exps: "
                f"{len(per_expert)}/{n_experts} experts present")
        canonical = canonical_gguf_name(ManagedID(bid, 0, kind))
        # 1-element f16 placeholder for the canonical tensor.
        stub = np.zeros((1,), dtype=np.float16)
        writer.add_tensor(canonical, stub, raw_dtype=GGMLQuantizationType.F16)
        # Canonical shape KV — ggml ne order: innermost first.
        if kind == "down":
            n_rows, m_cols = hidden, intermediate
        else:
            n_rows, m_cols = intermediate, hidden
        writer.add_array(k_canonical_shape(canonical),
                          [int(m_cols), int(n_rows), int(n_experts)])

        for xid, (pc, n_out_e, m_in_e) in sorted(per_expert.items()):
            wid = f"{canonical}:e{xid}"
            fixed = pc.data[:pc.fixed_overhead_bytes]
            byte_arr = np.frombuffer(fixed, dtype=np.int8)
            writer.add_tensor(
                f"{KV_PREFIX}bytes.{wid}",
                byte_arr,
                raw_shape=(byte_arr.size,),
                raw_dtype=GGMLQuantizationType.I8,
            )
            n_input_groups = -(-m_in_e // hp.group_size)
            d1 = n_out_e * n_input_groups
            writer.add_array(k_shape(wid), [int(n_out_e), int(m_in_e)])
            writer.add_array(k_chunk_bytes(wid), [int(b) for b in pc.chunk_sizes])
            writer.add_uint32(k_fixed_bytes(wid), int(pc.fixed_overhead_bytes))
            writer.add_uint32(k_d1(wid), int(d1))
            writer.add_uint32(k_n_groups_per_row(wid), int(n_input_groups))
            writer.add_uint32(k_padded_m(wid), int(n_input_groups * hp.group_size))
            managed_names.append(wid)
        managed_names.append(canonical)

    bundle_names: list[str] = []
    kinds = ("gate", "up", "down")
    for bid in sorted({layer for (layer, _kind) in accum}):
        if limit_layers is not None and bid >= limit_layers:
            continue
        per_kind = {kind: accum.get((bid, kind), {}) for kind in kinds}
        if any(len(per_kind[kind]) != n_experts for kind in kinds):
            continue
        canonicals = {
            kind: canonical_gguf_name(ManagedID(bid, 0, kind))
            for kind in kinds
        }
        for xid in range(n_experts):
            bundle_name = bundle_tensor_name(bid, xid)
            bundle_names.append(bundle_name)
            body = bytearray()
            wids: list[str] = []
            cids: list[int] = []
            offsets: list[int] = []
            sizes: list[int] = []
            n_chunks = len(per_kind["gate"][xid][0].chunk_sizes)
            for cid in range(n_chunks):
                for kind in kinds:
                    pc, n_out_e, m_in_e = per_kind[kind][xid]
                    payload = _ss_anybcq_chunk_to_kernel(
                        pc, cid, n=n_out_e, m=m_in_e,
                        group_size=hp.group_size)
                    offsets.append(len(body))
                    sizes.append(len(payload))
                    body += payload
                    wids.append(f"{canonicals[kind]}:e{xid}")
                    cids.append(100 + cid)
            arr = np.frombuffer(bytes(body), dtype=np.int8)
            writer.add_tensor(
                bundle_name, arr,
                raw_shape=(arr.size,),
                raw_dtype=GGMLQuantizationType.I8,
            )
            writer.add_array(k_bundle_wids(bundle_name), wids)
            writer.add_array(k_bundle_cids(bundle_name), cids)
            writer.add_array(k_bundle_offsets(bundle_name), offsets)
            writer.add_array(k_bundle_sizes(bundle_name), sizes)
            writer.add_string(k_bundle_format(bundle_name), "kernel_ready")
    if bundle_names:
        writer.add_array(k_bundle_tensors(), bundle_names)

    # Global KV.
    writer.add_uint32(f"{KV_PREFIX}version", 1)
    writer.add_string(f"{KV_PREFIX}encoder", "anybcq")
    writer.add_uint32(f"{KV_PREFIX}group_size", hp.group_size)
    writer.add_uint32(f"{KV_PREFIX}base_precision", hp.base_precision)
    writer.add_uint32(f"{KV_PREFIX}target_precision", hp.target_precision)
    writer.add_array(f"{KV_PREFIX}managed_tensors", managed_names)
    writer.add_bool(f"{KV_PREFIX}required_runtime", bool(required_runtime))
    writer.add_string(f"{KV_PREFIX}executor",
                       getattr(adapter, "dpa_moe_executor_name",
                                "qwen3_ss_anybcq_v1"))

    # ── Expert-output residuals (gate²·‖ΔE‖²) ────────────────────────────
    # Optional whole-model KV pack of the per-(layer, expert, K)
    # gate-weighted output-L2² residuals + routed-token count, produced
    # by ``experiments/3_dynamic_precision/profiling/profile_residuals.py``.
    # Lets the runtime/server compute a static layout for any budget at
    # install time without an external ``expert_residuals.npz``.
    if residuals_npz is not None and residuals_npz.exists():
        import numpy as _np
        d = _np.load(residuals_npz, allow_pickle=True)
        R = d["R"].astype(_np.float32)
        N = d["N"].astype(_np.int64)
        K_min = int(d["K_min"]); K_max = int(d["K_max"])
        n_layers_R, n_experts_R, n_K = R.shape
        # NaN → 0.0 sentinel; loader uses N==0 as "no signal, fall back
        # to per-layer mean".  Writing NaN as-is would break some
        # consumer libraries.
        R_flat = _np.nan_to_num(R, nan=0.0).reshape(-1).tolist()
        N_flat = N.reshape(-1).tolist()
        writer.add_array(K_EXPERT_R, [float(v) for v in R_flat])
        writer.add_array(K_EXPERT_N, [int(v)   for v in N_flat])
        writer.add_uint32(K_EXPERT_R_LAYERS,  int(n_layers_R))
        writer.add_uint32(K_EXPERT_R_EXPERTS, int(n_experts_R))
        writer.add_uint32(K_EXPERT_R_K_MIN,   K_min)
        writer.add_uint32(K_EXPERT_R_K_MAX,   K_max)
        writer.add_string(K_EXPERT_R_METRIC,
                           "gate_sq_l2_sq_mean_over_routed_tokens")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
