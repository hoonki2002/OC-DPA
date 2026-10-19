"""Calibration entry points.

  * `do_calibrate_ss_anybcq`  — **The** ss_anybcq calibration:
    incremental block-reconstruction in the spirit of Park et al. 2024
    (Any-Precision LLM, ICML 2024) adapted to BCQ. Active α planes are
    jointly retrained at each precision tier with a previous-K
    preservation loss. No post-hoc sort/BST.

  * `do_calibrate_anybcq`     — Per-precision-α (FLAG_ANY_PREC=1)
    calibrator: standard AnyBCQ with per-tier scales, used for the
    storage-overhead comparison against SharedScale AnyBCQ.

All write one safetensors per decoder block under
`codebooks_dir/layer_<bid>.safetensors`, with keys
`e{xid:03d}.{kind}.bytes` containing the disk-format byte stream.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file as _save_file

from .dataset import build_calibration_loader
from .hp import HP
from .packers import build_ss_anybcq_stream
from .upstream_shims import patch_upstream_skip


# -----------------------------------------------------------------------------
# Shared HF loader
# -----------------------------------------------------------------------------
def _load_hf_model(
    hf_dir: pathlib.Path,
    *,
    device: str = "cuda:0",
    attn_implementation: str = "eager",
):
    """Shared HF-model loader: fp16, eager attention, KV cache off.

    Includes a compatibility patch for older trust_remote_code modeling
    files (e.g. ``modeling_deepseek.py`` shipped with
    ``deepseek-moe-16b-chat``): transformers 5.x auto-populates
    ``config.rope_scaling`` from the new ``rope_parameters`` block
    with ``{rope_type: 'default'}``, but those models expect either
    ``None`` (= use default rotary, no scaling) or the legacy
    ``{type, factor}`` schema.  Normalise the "default-only" form
    back to ``None`` before model construction.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    # Capture rope_theta BEFORE any normalization.  Transformers 5.x
    # treats ``rope_scaling`` and ``rope_parameters`` as aliases of one
    # backing field; setting ``cfg.rope_scaling = None`` below
    # implicitly clears ``rope_parameters`` too, after which our
    # ``rope_parameters`` re-injection block falls back to
    # ``getattr(cfg, "rope_theta", 10000.0)`` and silently bakes the
    # wrong default (10_000) into the GGUF's ``rope.freq_base``.
    # Stash here so the subsequent re-injection has the true value.
    _captured_rope_theta = (
        getattr(cfg, "rope_theta", None)
        or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
    )
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict) and "type" not in rs:
        # transformers 5.x auto-populates ``rope_scaling`` from the new
        # ``rope_parameters`` block to shapes like ``{rope_theta,
        # rope_type: 'default'}`` (no 'type' key).  Legacy custom-modeling
        # files (e.g. DeepSeek-MoE-16B's modeling_deepseek.py:472) read
        # ``rope_scaling['type']`` unconditionally and crash with
        # KeyError.  Normalise the new-format / partial dicts back to
        # None — none of the supported checkpoints actually use rope
        # scaling, and the legacy code falls through to the default
        # rotary path when the field is None.
        cfg.rope_scaling = None
    # transformers 5.x rotary-embedding paths (Qwen3MoeRotaryEmbedding
    # etc.) read ``config.rope_parameters[...]`` unconditionally at
    # module __init__ — both ``rope_type`` and ``rope_theta`` keys are
    # required by ``compute_default_rope_parameters``.  Older HF
    # configs carry ``rope_theta`` at the top level only; build the new
    # block from those when missing so legacy checkpoints load on 5.x.
    #
    # SCOPE: only for in-tree transformers modeling files that need the
    # new block (Qwen3-MoE etc.).  Custom-modeling checkpoints loaded
    # via ``trust_remote_code`` (e.g. DeepSeek-MoE's
    # ``modeling_deepseek.py``) ship their own rotary code that uses
    # the legacy ``rope_scaling`` schema; injecting ``rope_parameters``
    # makes transformers back-populate ``rope_scaling`` to a dict
    # without a ``type`` key, which the custom modeling then crashes
    # on (KeyError 'type').  Skip the injection for ``trust_remote_code``
    # configs.
    _model_type = getattr(cfg, "model_type", "")
    _is_custom = bool(getattr(cfg, "auto_map", None))
    if not _is_custom and getattr(cfg, "rope_parameters", None) is None:
        cfg.rope_parameters = {
            "rope_type": "default",
            "rope_theta": float(
                _captured_rope_theta
                if _captured_rope_theta is not None
                else getattr(cfg, "rope_theta", 10000.0)
            ),
        }
    # ``_captured_rope_theta`` (captured pre-normalization above) is
    # the authoritative source for what the GGUF adapter should emit.
    # Reuse it here.
    _orig_rope_theta = _captured_rope_theta
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_dir, config=cfg,
            torch_dtype=torch.float16, device_map=device,
            trust_remote_code=True, attn_implementation=attn_implementation)
    except (TypeError, ValueError) as exc:
        if attn_implementation != "eager":
            print(
                f"[calibrate] attn_implementation={attn_implementation!r} "
                f"unsupported for {hf_dir.name}; falling back to eager "
                f"({exc.__class__.__name__})",
                flush=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                hf_dir, config=cfg,
                torch_dtype=torch.float16, device_map=device,
                trust_remote_code=True, attn_implementation="eager")
        else:
            raise
    model.config.use_cache = False
    try:
        model.config._attn_implementation = attn_implementation
    except Exception:
        pass
    # Re-instate the genuine rope_theta on both ``model.config`` and
    # its ``rope_parameters`` dict so downstream readers (the GGUF
    # adapter, in particular) see the source value not the
    # post-construction default.
    if _orig_rope_theta is not None:
        model.config.rope_theta = float(_orig_rope_theta)
        rp = getattr(model.config, "rope_parameters", None)
        if isinstance(rp, dict):
            rp["rope_theta"] = float(_orig_rope_theta)

    # Stash the model's rotary_emb for the DataCacheWrapper shim's
    # per-layer-type position_embeddings fixup (Gemma 4 has
    # sliding=256 vs full=512 head_dim, so cached kwargs reused
    # across layer-type boundaries land with wrong-sized cos/sin).
    from . import upstream_shims as _shims
    _rot = None
    _lm = getattr(model.model, "language_model", None)
    if _lm is not None and hasattr(_lm, "rotary_emb"):
        _rot = _lm.rotary_emb
    elif hasattr(model.model, "rotary_emb"):
        _rot = model.model.rotary_emb
    _shims.register_rotary_emb(_rot)

    return tok, model


def _sanitize_fp16(t: torch.Tensor) -> torch.Tensor:
    FP16_MAX = 65504.0
    return t.clamp_(-FP16_MAX, FP16_MAX).nan_to_num_(
        nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)


def _flatten_cached_batch(t):
    if t is None or not torch.is_tensor(t):
        return t
    if t.dim() >= 2:
        return t.flatten(0, 1)
    return t


def _repeat_metadata_to_batch(t, batch_size: int):
    if t is None or not torch.is_tensor(t) or t.dim() == 0:
        return t
    if t.shape[0] == batch_size:
        return t
    if t.shape[0] == 1:
        return t.expand(batch_size, *t.shape[1:])
    if batch_size % t.shape[0] == 0:
        return t.repeat_interleave(batch_size // t.shape[0], dim=0)
    return t


# -----------------------------------------------------------------------------
# ss_anybcq — incremental block-reconstruction
# -----------------------------------------------------------------------------
#
# Algorithm (Park et al. 2024, Any-Precision LLM, adapted to BCQ):
#
#   1. Train at base precision (base_p planes) — Adam tunes α[0..base_p-1]
#      and β jointly. Same as upstream's add_bit==0 step.
#   2. For each new plane k = base_p .. target_p - 1:
#        a. `add_onebit_model` (upstream) extracts the new sign plane
#           B[k] from the residual via alternating refinement.
#        b. Our overridden `blockReconstruction` jointly retrains all
#           active alpha planes and adds a previous-K preservation loss.
#   3. No post-hoc sort / BST; the ss_anybcq disk/runtime layout stays
#      one shared alpha/beta table plus the original bitplanes.
#
# Storage format unchanged: `build_ss_anybcq_stream` (FLAG_ANY_PREC=0).
# Runtime needs no change.
#
# Cost vs the legacy joint+sort+BST path: roughly the same training time
# (`add_bits + 1` calls to blockReconstruction either way); no separate
# sort+BST repack step at the end.
def do_calibrate_ss_anybcq(
    hf_dir: pathlib.Path, calib_files: list[pathlib.Path],
    hp: HP, codebooks_dir: pathlib.Path,
    limit_layers: int | None,
    adapter,
    calibration_mode: str = "sequential",
):
    """Production ss_anybcq calibration with joint active-alpha retraining."""

    if calibration_mode not in ("sequential", "layer_fp_parallel"):
        raise ValueError(
            f"unknown calibration_mode={calibration_mode!r}; expected "
            "sequential|layer_fp_parallel")
    _is_managed = adapter.is_managed_linear
    _parse_mid  = adapter.parse_managed_name

    import types

    from anybcq.inference.AnyBCQLinear import AnyBCQLinear
    from anybcq.quantization.anybcq import AnyBCQ, LossFunction
    from anybcq.quantization.bcq_linear import BCQLinear
    from anybcq.utils.swap_linear import save_alpha_and_beta_in_bcqlinear
    from tqdm.auto import tqdm

    t0 = time.monotonic()
    print("[calibrate] ss_anybcq incremental block-reconstruction "
          "(Park 2024 scheme adapted to BCQ)", flush=True)
    print(f"[calibrate] loading {hf_dir}", flush=True)
    tok, model = _load_hf_model(hf_dir, device="cuda:0")
    print(f"[calibrate] model loaded in {time.monotonic()-t0:.1f}s",
          flush=True)

    if limit_layers is not None:
        model.model.layers = nn.ModuleList(list(model.model.layers)[:limit_layers])
        model.config.num_hidden_layers = limit_layers
        print(f"[calibrate] DEBUG: limited to {limit_layers} layers",
              flush=True)

    import os as _os_early_ss
    _lr_early_ss = _os_early_ss.environ.get("DPA_MOE_LAYER_RANGE")
    _unfuse_range_ss = None
    if _lr_early_ss:
        _a, _b = _lr_early_ss.split(":")
        _unfuse_range_ss = (int(_a), int(_b))
    n_unfused = adapter.unfuse_moe(model, layer_range=_unfuse_range_ss)
    print(f"[calibrate] unfused {n_unfused} layers via "
          f"{type(adapter).__name__} (range={_unfuse_range_ss})", flush=True)

    # RTN-quantize unmanaged weights (attention, lm_head, embed) to Q8_0.
    # This matches the inference GGUF environment so block-recon calibration
    # sees the same quantization noise as deployment.
    # BCQLinear expert weights do not exist yet at this point, so mod.weight
    # in every BCQLinear created later will hold the original BF16 weight.
    _rtn_q8_0_n = 0
    def _rtn_q8_0_ss(W: torch.Tensor) -> torch.Tensor:
        bs = 32
        orig_shape = W.shape
        W_f = W.detach().float().reshape(-1)
        n = W_f.shape[0]
        n_pad = (-n) % bs
        if n_pad > 0:
            W_f = torch.cat([W_f, torch.zeros(n_pad, device=W_f.device)])
        blk = W_f.reshape(-1, bs)
        sc = blk.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
        W_dq = ((blk / sc).round_().clamp_(-127, 127) * sc).reshape(-1)[:n]
        return W_dq.reshape(orig_shape).to(W.dtype)
    for _nm_ss, _mod_ss in model.named_modules():
        if isinstance(_mod_ss, nn.Linear) and not _is_managed(_nm_ss):
            _mod_ss.weight.data = _rtn_q8_0_ss(_mod_ss.weight.data)
            _rtn_q8_0_n += 1
    print(f"[calibrate] RTN Q8_0-quantized {_rtn_q8_0_n} unmanaged Linears", flush=True)

    _iters_env = _os_early_ss.environ.get("DPA_MOE_ITERS")
    if _iters_env:
        hp = hp.__replace__(iters=int(_iters_env))
        print(f"[calibrate] DPA_MOE_ITERS override: iters={hp.iters}", flush=True)

    dl = build_calibration_loader(
        tok, calib_files, hp.seq_len, hp.num_samples, hp.batch_size)
    print(f"[calibrate] calibration: {hp.num_samples} samples × "
          f"{hp.seq_len} tokens; iters={hp.iters} batch={hp.batch_size}",
          flush=True)

    recon_dtype = torch.float32 if hp.recon_dtype == "float32" else torch.bfloat16

    # Build the upstream AnyBCQ instance and bind the ss_anybcq
    # preserve-joint block-reconstruction override onto this instance only.
    bcq = AnyBCQ(
        model, dl,
        wq_params={"n_bits": hp.base_precision, "group_size": hp.group_size},
        aq_params={"n_bits": 16},
        iters=hp.iters, num_samples=hp.num_samples, batch_size=hp.batch_size,
        add_bits=hp.add_bits,
        w_lr1=hp.w_lr1, w_lr2=hp.w_lr2, w_lr3=hp.w_lr3,
        torch_dtype=torch.float16, recon_dtype=recon_dtype,
        asymmetric=hp.asymmetric, train_beta=hp.train_beta,
    )
    def compute_w_k(mod: BCQLinear, k: int) -> torch.Tensor:
        """Reconstruct weight at precision k from BCQLinear.fpBinary.

        Returns (O, I_grp, G) float32.  Valid only during block-recon
        (while mod is still BCQLinear; fpBinary disappears after
        replace_bcq_with_lutgemm).
        """
        B_k = mod.fpBinary[:, :, :, :k].float()                   # (O, I_grp, G, k)
        a_k = torch.stack(
            [getattr(mod, f"alpha{i}") for i in range(k)], dim=-1
        ).float()                                                   # (O, I_grp, k)
        return (mod.beta[:, :, None].float()
                + (a_k[:, :, None, :] * B_k).sum(dim=-1))          # (O, I_grp, G)

    def blockReconstruction_preserve_joint(
        self, recon_block, cached_dataloader, add_bit,
        layer_idx=None, num_layers=None,
    ):
        """Joint active-alpha block reconstruction for ss_anybcq."""
        assert hp.target_precision == hp.base_precision + hp.add_bits, (
            f"target_precision={hp.target_precision} != "
            f"base_precision={hp.base_precision} + add_bits={hp.add_bits}")

        # K_new == mod.num_bit here (add_onebit_model already ran for i>0).
        K_new = hp.base_precision + add_bit   # NO +1

        if add_bit == 0:
            lr_alpha = self.w_lr1
        elif add_bit == 1:
            lr_alpha = self.w_lr2
        else:
            lr_alpha = self.w_lr3

        all_params: list = []
        trainable_params: list = []   # for clip_grad_norm_
        train_indices = list(range(K_new))
        for mod in recon_block.modules():
            if isinstance(mod, BCQLinear):
                for idx in train_indices:
                    p = getattr(mod, f"alpha{idx}")
                    all_params.append({"params": [p], "lr": lr_alpha})
                    trainable_params.append(p)
                # Beta stays frozen in the block-reconstruction override.

        optimizer = torch.optim.Adam(all_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=0.0,
        )

        print("*" * 42)
        print(f"[ss_anybcq preserve-joint] add_bit={add_bit}  K_new={K_new}")
        print(f"  lr_alpha={lr_alpha}  beta=FROZEN  iters={self.iters}")
        if hp.train_beta and add_bit == 0 and layer_idx in (None, 0):
            print("  note: train_beta is ignored for ss_anybcq; the shared β keeps its "
                  "least-squares init so every prefix uses the same offset")
        print(f"  trainable alpha planes: {train_indices}")
        print("*" * 42)

        # DeepSeek-MoE's eval forward dispatches routed experts through
        # ``moe_infer()``, which is decorated with ``@torch.no_grad()`` in
        # the checkpoint's custom modeling code.  Calibration must use the
        # differentiable training forward so alpha planes receive gradients.
        was_training = recon_block.training
        if adapter.name == "deepseek_moe":
            recon_block.train()

        loss_func = LossFunction(recon_block, max_count=self.iters)
        epochs = int(self.iters / len(cached_dataloader))
        remainder = self.iters - len(cached_dataloader) * epochs

        # Freeze all params; re-enable only selected α planes.
        for _name, param in recon_block.named_parameters():
            param.requires_grad = False
        for mod in recon_block.modules():
            if isinstance(mod, BCQLinear):
                for idx in train_indices:
                    getattr(mod, f"alpha{idx}").requires_grad = True
                # mod.beta stays frozen in this override.

        lam = 0.5

        optimizer.zero_grad()
        loss = None
        for epoch in tqdm(range(epochs + 1)):
            for step, batch in enumerate(cached_dataloader):
                if epoch == epochs and step == remainder:
                    break

                input_q = _flatten_cached_batch(batch["q_input"])
                input_fp = _flatten_cached_batch(batch["fp_input"])
                output_fp = _flatten_cached_batch(batch["fp_output"])
                self.attention_mask = batch.get("attention_mask", None)
                self.position_ids   = batch.get("position_ids", None)
                cache_position      = batch.get("cache_position", None)
                if self.attention_mask is not None:
                    self.attention_mask = _flatten_cached_batch(self.attention_mask)
                if self.position_ids is not None:
                    self.position_ids = _flatten_cached_batch(self.position_ids)
                if cache_position is not None:
                    cache_position = _flatten_cached_batch(cache_position)
                self.attention_mask = _repeat_metadata_to_batch(
                    self.attention_mask, input_q.shape[0])
                self.position_ids = _repeat_metadata_to_batch(
                    self.position_ids, input_q.shape[0])

                if self.input_prob < 1.0:
                    input_q = torch.where(
                        torch.rand_like(input_q, dtype=input_q.dtype) < self.input_prob,
                        input_q,
                        input_fp,
                    )

                input_q = self.type_cast(input_q, self.recon_dtype)
                self.attention_mask = self.type_cast(
                    self.attention_mask, self.recon_dtype)

                rope = None
                _lang = getattr(self.model.model, "language_model", None)
                if _lang is not None and hasattr(_lang, "rotary_emb"):
                    rope = _lang.rotary_emb
                elif hasattr(self.model.model, "rotary_emb"):
                    rope = self.model.model.rotary_emb
                elif hasattr(recon_block.self_attn, "rotary_emb"):
                    rope = recon_block.self_attn.rotary_emb

                if self.position_ids is None and input_q is not None:
                    if input_q.dim() == 3:
                        _bsz, _seq, _ = input_q.shape
                    elif input_q.dim() == 2:
                        _bsz, _seq = 1, input_q.shape[0]
                    else:
                        _bsz, _seq = 1, input_q.shape[-2] if input_q.dim() >= 2 else 1
                    self.position_ids = torch.arange(
                        _seq, dtype=torch.long, device=input_q.device
                    ).unsqueeze(0).expand(_bsz, -1)

                if not getattr(self, "_rope_debug_logged", False):
                    import sys as _sys
                    print(
                        f"[calibrate-rope] rope={type(rope).__name__ if rope else None} "
                        f"position_ids.shape="
                        f"{None if self.position_ids is None else tuple(self.position_ids.shape)}",
                        file=_sys.stderr, flush=True)
                    self._rope_debug_logged = True

                position_embeddings = None
                if rope is not None and self.position_ids is not None:
                    _layer_type = None
                    _sa = getattr(recon_block, "self_attn", None)
                    if _sa is not None:
                        _layer_type = getattr(_sa, "layer_type", None)
                        if _layer_type is None and hasattr(_sa, "is_sliding"):
                            _layer_type = ("sliding_attention" if _sa.is_sliding
                                           else "full_attention")
                    if _layer_type is None:
                        _cfg = getattr(recon_block, "config", None)
                        _lidx = getattr(recon_block, "layer_idx", None)
                        if _cfg is not None and _lidx is not None:
                            _lts = getattr(_cfg, "layer_types", None)
                            if _lts is not None and 0 <= _lidx < len(_lts):
                                _layer_type = _lts[_lidx]

                    for _attempt in (
                        (lambda: rope(input_q, self.position_ids, layer_type=_layer_type)
                                  if _layer_type else None),
                        (lambda: rope(input_q, self.position_ids)),
                    ):
                        try:
                            res = _attempt()
                            if res is None: continue
                            cos, sin = res
                            position_embeddings = (cos, sin)
                            break
                        except (TypeError, RuntimeError, AttributeError):
                            continue
                    _key = (_layer_type, position_embeddings is not None)
                    if getattr(self, "_rope_pe_last_key", None) != _key:
                        import sys as _sys
                        _hd = getattr(getattr(recon_block, "self_attn", None),
                                       "head_dim", None)
                        print(
                            f"[calibrate-rope] pe={'(cos,sin)' if position_embeddings else None} "
                            f"layer_type={_layer_type} head_dim={_hd}",
                            file=_sys.stderr, flush=True)
                        self._rope_pe_last_key = _key

                import inspect as _inspect
                _layer_params = set(
                    _inspect.signature(recon_block.forward).parameters)
                _extra_kwargs = {}
                if "shared_kv_states" in _layer_params:
                    _extra_kwargs["shared_kv_states"] = {}
                if "per_layer_input" in _layer_params:
                    _extra_kwargs["per_layer_input"] = None
                if position_embeddings is not None:
                    _extra_kwargs["position_embeddings"] = position_embeddings

                if not getattr(self, "_dtype_logged", False):
                    import sys as _sys
                    _pe_dt = (position_embeddings[0].dtype
                              if position_embeddings is not None else None)
                    _q_dt = None
                    if hasattr(recon_block, "self_attn") \
                            and hasattr(recon_block.self_attn, "q_proj"):
                        _q_dt = recon_block.self_attn.q_proj.weight.dtype
                    print(
                        f"[block-recon-dtype] input_q={input_q.dtype} "
                        f"q_proj.weight={_q_dt} "
                        f"position_embeddings[0]={_pe_dt} "
                        f"attn_mask="
                        f"{self.attention_mask.dtype if self.attention_mask is not None else None}",
                        file=_sys.stderr, flush=True)
                    self._dtype_logged = True

                if self.attention_mask is not None:
                    _kwargs = dict(
                        hidden_states=input_q,
                        attention_mask=self.attention_mask,
                        position_ids=self.position_ids,
                        cache_position=cache_position,
                    )
                    _kwargs.update(_extra_kwargs)
                    output_q = recon_block(**_kwargs)
                else:
                    output_q = recon_block(input_q, **_extra_kwargs)

                block_loss = loss_func(output_q, output_fp)

                # Previous-K BF16 auxiliary regularization:
                # push W_k → original BF16 for all k < K_new.
                # Mean over all (module × k) terms keeps λ tier-invariant.
                pres_terms = []
                if add_bit > 0 and lam > 0.0:
                    for mod in recon_block.modules():
                        if isinstance(mod, BCQLinear):
                            O, I_grp, G, _ = mod.fpBinary.shape
                            W_fp = mod.weight.detach().float().reshape(O, I_grp, G)
                            for k in range(hp.base_precision, K_new):
                                W_k = compute_w_k(mod, k)
                                pres_terms.append(
                                    (W_k - W_fp).pow(2).mean().to(block_loss.device)
                                )
                pres_loss = (
                    torch.stack(pres_terms).mean()
                    if pres_terms
                    else torch.zeros((), device=block_loss.device)
                )

                loss = block_loss + lam * pres_loss

                global_step = epoch * len(cached_dataloader) + step
                if global_step % 50 == 0:
                    import sys as _sl
                    print(
                        f"  [step {global_step:4d}] block={block_loss.item():.2e}"
                        f"  pres={pres_loss.item():.2e}",
                        file=_sl.stderr, flush=True)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        del optimizer
        if loss is not None:
            del loss
        torch.cuda.empty_cache()
        recon_block.train(was_training)

        save_alpha_and_beta_in_bcqlinear(recon_block)
        print("=" * 30)
        return recon_block

    bcq.blockReconstruction = types.MethodType(
        blockReconstruction_preserve_joint, bcq)

    # ── Per-layer streaming hook ────────────────────────────────────────
    # Fires after each decoder block's quantization finishes (post
    # `replace_bcq_with_lutgemm`, layer on CPU). Extracts that one
    # block's AnyBCQLinear codebooks, packs them, and writes the
    # per-layer safetensors immediately. No in-memory accumulation
    # of the full model's codebook bytes.
    #
    # See third_party/anybcq/anybcq/quantization/anybcq.py for
    # the hook injection site (single line added by dpa_moe).
    P = hp.target_precision
    gs = hp.group_size
    codebooks_dir.mkdir(parents=True, exist_ok=True)
    n_packed_total = [0]   # boxed for closure write

    # `recon_block.named_modules()` gives block-relative names; the
    # exact form depends on the MoE arch:
    #   * Qwen3-MoE: ``mlp.experts.experts.<xid>.<kind>_proj``
    #   * DeepSeek / Gemma: ``mlp.experts.<xid>.<kind>_proj``
    # Synthesize a fake FQN by prepending ``model.layers.<bid>.`` and
    # feed it through the adapter's parse_managed_name — works
    # uniformly for any registered MoE arch.

    from ._qweight_unpack import _unpack_qweight

    # GPU device for the qweight bit-unpack. `_unpack_qweight` is
    # device-agnostic (uses input.device) so we just push qweight to
    # CUDA before calling it. The big permute+contiguous in there is
    # the bottleneck; on GPU it's ~50× faster than serial CPU.
    _pack_dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def stream_layer(bid: int, recon_block: nn.Module):
        t_layer = time.monotonic()
        if limit_layers is not None and bid >= limit_layers:
            return
        layer_dict: dict[str, torch.Tensor] = {}
        n_packed = 0
        for name, mod in recon_block.named_modules():
            if not isinstance(mod, AnyBCQLinear):
                continue
            _mid = _parse_mid(f"model.layers.{bid}.{name}")
            if _mid is None:
                continue
            xid = _mid.xid
            kind = _mid.kind
            in_features  = mod.in_features
            out_features = mod.out_features
            n_input_groups = in_features // gs
            d1 = n_input_groups * out_features

            a = getattr(mod, f"alpha_{P}").detach().float().cpu()
            a = a.permute(2, 0, 1).contiguous().view(d1, P)
            beta = getattr(mod, f"beta_{P}").detach().float().cpu()
            beta = beta.permute(1, 0).contiguous().view(d1)
            _sanitize_fp16(a)
            _sanitize_fp16(beta)

            # GPU-offloaded bit-unpack: move qweight to GPU, run the
            # permute+contiguous there, copy result back to CPU. The
            # qweight tensor is small (int32, ~few MB per expert) so
            # PCIe round-trip is dwarfed by the permute speedup.
            qw_gpu = mod.qweight.to(_pack_dev, non_blocking=True)
            B = _unpack_qweight(qw_gpu, P, gs).cpu().to(torch.float32)
            del qw_gpu

            pc = build_ss_anybcq_stream(
                B, a, beta,
                n=out_features, m=in_features,
                base_precision=hp.base_precision, group_size=gs,
            )
            prefix = f"e{xid:03d}.{kind}"
            layer_dict[f"{prefix}.bytes"] = torch.from_numpy(
                np.frombuffer(pc.data, dtype=np.uint8).copy())
            n_packed += 1
            del B

        if layer_dict:
            out = codebooks_dir / f"layer_{bid:02d}.safetensors"
            _save_file(layer_dict, str(out))
            n_packed_total[0] += n_packed
            print(f"[calibrate] layer {bid:02d} packed+saved "
                  f"({n_packed} experts × kinds, "
                  f"{out.stat().st_size/1e6:.0f} MB) in "
                  f"{time.monotonic()-t_layer:.1f}s", flush=True)

    with patch_upstream_skip(lambda n: not _is_managed(n)):
        # Multi-GPU layer-range partition: when DPA_MOE_LAYER_RANGE is
        # set (e.g. "0:5" -> layers 0..5 inclusive), only those layers
        # are calibrated.  Pair with DPA_MOE_FP_CACHING=1 so the
        # un-calibrated layers between the start and the range still
        # cache BF16 outputs to the next layer (AdaQuant-style), which
        # is required for independent partitions.  DPA_MOE_INNER_BS
        # tunes the inner DataLoader batch size (upstream default 1).
        import os as _os
        _lr = _os.environ.get("DPA_MOE_LAYER_RANGE")
        _layer_range = None
        if _lr:
            _a, _b = _lr.split(":")
            _layer_range = (int(_a), int(_b))
        _use_fp = (
            calibration_mode == "layer_fp_parallel"
            or bool(int(_os.environ.get("DPA_MOE_FP_CACHING", "0")))
        )
        _inner_bs_env = _os.environ.get("DPA_MOE_INNER_BS")
        _inner_bs = int(_inner_bs_env) if _inner_bs_env else None
        print(f"[calibrate] minimize() layer_range={_layer_range} "
              f"use_fp_caching={_use_fp} inner_batch_size={_inner_bs}", flush=True)
        bcq.minimize(per_layer_hook=stream_layer,
                     layer_range=_layer_range,
                     use_fp_caching=_use_fp,
                     inner_batch_size=_inner_bs)
    print(f"[calibrate] AnyBCQ.minimize (incremental) + per-layer pack "
          f"done in {time.monotonic()-t0:.1f}s; "
          f"packed {n_packed_total[0]} expert codebooks total", flush=True)

# -----------------------------------------------------------------------------
# anybcq (per-precision-α)
# -----------------------------------------------------------------------------
def do_calibrate_anybcq(
    hf_dir: pathlib.Path, calib_files: list[pathlib.Path],
    hp: HP, codebooks_dir: pathlib.Path,
    limit_layers: int | None,
    adapter,
    calibration_mode: str = "sequential",
):
    """Per-precision-α calibrator (paper-faithful any-prec, FLAG_ANY_PREC=1).

    Uses the upstream AnyBCQ.blockReconstruction unmodified (all α planes
    trained jointly at each precision tier — NO freeze-prev). Before minimize(),
    non-managed weights (attn, lm_head, embd) are RTN-quantized to Q8_0 so
    block-recon calibrates against the same quantization noise those parts
    will introduce at inference time.

    After minimize(), the per_layer_hook reads alpha_p / beta_p / qweight from
    each AnyBCQLinear module (which preserves ALL per-precision snapshots), and
    packs via build_anybcq_stream (FLAG_ANY_PREC=1, per-tier α/β tables).

    Eval: only uniform-K is meaningful (no per-(e,c,p) residuals for DP
    dispatch). Run the sweep with CONFIGS=Nk:uniform only.
    """

    if calibration_mode not in ("sequential", "layer_fp_parallel"):
        raise ValueError(
            f"unknown calibration_mode={calibration_mode!r}; expected "
            "sequential|layer_fp_parallel")
    _is_managed = adapter.is_managed_linear
    _parse_mid  = adapter.parse_managed_name

    from anybcq.inference.AnyBCQLinear import AnyBCQLinear
    from anybcq.quantization.anybcq import AnyBCQ

    from ._qweight_unpack import anybcq_pack_expert_fast

    t0 = time.monotonic()
    print("[calibrate] anybcq per-precision-α (FLAG_ANY_PREC=1)", flush=True)
    print(f"[calibrate] loading {hf_dir}", flush=True)
    tok, model = _load_hf_model(hf_dir, device="cuda:0")
    print(f"[calibrate] model loaded in {time.monotonic()-t0:.1f}s", flush=True)

    if limit_layers is not None:
        model.model.layers = nn.ModuleList(list(model.model.layers)[:limit_layers])
        model.config.num_hidden_layers = limit_layers

    # Read DPA_MOE_LAYER_RANGE early so unfuse_moe only allocates the
    # expert modules for the layers this worker actually calibrates.
    # For FP_CACHING=1 (AdaQuant-style), out-of-range layers are only
    # used for their BF16 forward pass — they don't need to be unfused.
    import os as _os_early
    _lr_early = _os_early.environ.get("DPA_MOE_LAYER_RANGE")
    _unfuse_range = None
    if _lr_early:
        _a, _b = _lr_early.split(":")
        _unfuse_range = (int(_a), int(_b))
    n_unfused = adapter.unfuse_moe(model, layer_range=_unfuse_range)
    print(f"[calibrate] unfused {n_unfused} layers "
          f"(range={_unfuse_range})", flush=True)

    # RTN-quantize non-managed weights to Q8_0 before block-recon.
    # This matches the inference GGUF environment (attn/lm_head/embd at Q8_0),
    # so the MoE calibration sees the same quantization noise as inference.
    def _rtn_q8_0(W: torch.Tensor) -> torch.Tensor:
        block_size = 32
        orig_shape = W.shape
        W_f = W.detach().float().reshape(-1)
        n = W_f.shape[0]
        n_pad = (-n) % block_size
        if n_pad > 0:
            W_f = torch.cat([W_f, torch.zeros(n_pad, device=W_f.device)])
        W_blocks = W_f.reshape(-1, block_size)
        scales = W_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
        W_q = (W_blocks / scales).round_().clamp_(-127, 127)
        W_dq = (W_q * scales).reshape(-1)[:n].reshape(orig_shape)
        return W_dq.to(W.dtype)

    n_unmanaged_q = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not _is_managed(name):
            mod.weight.data = _rtn_q8_0(mod.weight.data)
            n_unmanaged_q += 1
    print(f"[calibrate] RTN Q8_0-quantized {n_unmanaged_q} unmanaged Linears", flush=True)

    _iters_env = _os_early.environ.get("DPA_MOE_ITERS")
    if _iters_env:
        hp = hp.__replace__(iters=int(_iters_env))
        print(f"[calibrate] DPA_MOE_ITERS override: iters={hp.iters}", flush=True)

    dl = build_calibration_loader(
        tok, calib_files, hp.seq_len, hp.num_samples, hp.batch_size)
    print(f"[calibrate] calibration: {hp.num_samples} samples × "
          f"{hp.seq_len} tokens; iters={hp.iters} batch={hp.batch_size}",
          flush=True)

    recon_dtype = torch.float32 if hp.recon_dtype == "float32" else torch.bfloat16

    # Standard AnyBCQ — upstream blockReconstruction, no freeze-prev override.
    bcq = AnyBCQ(
        model, dl,
        wq_params={"n_bits": hp.base_precision, "group_size": hp.group_size},
        aq_params={"n_bits": 16},
        iters=hp.iters, num_samples=hp.num_samples, batch_size=hp.batch_size,
        add_bits=hp.add_bits,
        w_lr1=hp.w_lr1, w_lr2=hp.w_lr2, w_lr3=hp.w_lr3,
        torch_dtype=torch.float16, recon_dtype=recon_dtype,
        asymmetric=hp.asymmetric, train_beta=hp.train_beta,
    )

    P  = hp.target_precision
    bp = hp.base_precision
    gs = hp.group_size
    codebooks_dir.mkdir(parents=True, exist_ok=True)
    n_packed_total = [0]

    def stream_layer_anybcq(bid: int, recon_block: nn.Module):
        t_layer = time.monotonic()
        if limit_layers is not None and bid >= limit_layers:
            return
        layer_dict: dict[str, torch.Tensor] = {}
        n_packed = 0
        for name, mod in recon_block.named_modules():
            if not isinstance(mod, AnyBCQLinear):
                continue
            _mid = _parse_mid(f"model.layers.{bid}.{name}")
            if _mid is None:
                continue
            in_features  = mod.in_features
            out_features = mod.out_features
            n_input_groups = in_features // gs
            d1 = out_features * n_input_groups

            # Collect per-precision (α_p, β_p) tensors.
            # AnyBCQLinear stores alpha_p as (IC, p, OC) and beta_p as (IC, OC).
            # Permute and flatten to (d1, p) and (d1,) respectively.
            alphas: list[torch.Tensor] = []
            betas:  list[torch.Tensor] = []
            for p in range(bp, P + 1):
                if not hasattr(mod, f"alpha_{p}"):
                    continue
                A_p = getattr(mod, f"alpha_{p}").detach().float().cpu()
                A_p = A_p.permute(2, 0, 1).contiguous().view(d1, p)
                b_p = getattr(mod, f"beta_{p}").detach().float().cpu()
                b_p = b_p.permute(1, 0).contiguous().view(d1)
                _sanitize_fp16(A_p)
                _sanitize_fp16(b_p)
                alphas.append(A_p)
                betas.append(b_p)

            if not alphas:
                continue

            # Fast LUT-based packing: int32 qweight → MSB-first sign bytes.
            # ~5–10× faster than the float32 sign-expansion path.
            pc = anybcq_pack_expert_fast(
                qweight=mod.qweight.detach().cpu(),
                alphas=alphas, betas=betas,
                base_precision=bp, target_precision=P,
                n_out=out_features, m_in=in_features,
                group_size=gs,
            )
            xid  = _mid.xid
            kind = _mid.kind
            prefix = f"e{xid:03d}.{kind}"
            layer_dict[f"{prefix}.bytes"] = torch.from_numpy(
                np.frombuffer(pc.data, dtype=np.uint8).copy())
            n_packed += 1

        if layer_dict:
            out = codebooks_dir / f"layer_{bid:02d}.safetensors"
            _save_file(layer_dict, str(out))
            n_packed_total[0] += n_packed
            print(f"[calibrate] layer {bid:02d} anybcq packed+saved "
                  f"({n_packed} experts × kinds, "
                  f"{out.stat().st_size/1e6:.0f} MB) in "
                  f"{time.monotonic()-t_layer:.1f}s", flush=True)

    with patch_upstream_skip(lambda n: not _is_managed(n)):
        import os as _os
        _lr = _os.environ.get("DPA_MOE_LAYER_RANGE")
        _layer_range = None
        if _lr:
            _a, _b = _lr.split(":")
            _layer_range = (int(_a), int(_b))
        _use_fp = (
            calibration_mode == "layer_fp_parallel"
            or bool(int(_os.environ.get("DPA_MOE_FP_CACHING", "0")))
        )
        _inner_bs_env = _os.environ.get("DPA_MOE_INNER_BS")
        _inner_bs = int(_inner_bs_env) if _inner_bs_env else None
        print(f"[calibrate] minimize() layer_range={_layer_range} "
              f"use_fp_caching={_use_fp} inner_batch_size={_inner_bs}", flush=True)
        bcq.minimize(per_layer_hook=stream_layer_anybcq,
                     layer_range=_layer_range,
                     use_fp_caching=_use_fp,
                     inner_batch_size=_inner_bs)

    print(f"[calibrate] anybcq minimize + per-layer pack done in "
          f"{time.monotonic()-t0:.1f}s; "
          f"packed {n_packed_total[0]} expert codebooks total", flush=True)
