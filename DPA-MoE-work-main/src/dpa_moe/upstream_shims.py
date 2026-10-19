"""Upstream-packaging shims for vendored `anybcq`.

Run on import. Two effects:

  (1) Stub upstream's CUDA inference kernel `custom_kernel`. `plugin.py`
      does `import custom_kernel` at module load. The kernel is only
      needed for inference (LUT-GEMV); encode-only, so no-op stub keeps
      import alive.

  (2) Provide `patch_upstream_skip`, a context manager that replaces
      the hard-coded skip predicate in `swap_quant_model`.

Do NOT inject a stub for `anybcq.utils` — upstream's real `swap_linear`
must remain importable for block-reconstruction to work.
"""
from __future__ import annotations

import contextlib
import sys
import types
from typing import Callable



# (a) Stub upstream's CUDA inference kernel before any anybcq import chain.
sys.modules.setdefault("custom_kernel", types.ModuleType("custom_kernel"))

print("[upstream_shims] module imported, applying shims...",
      file=sys.stderr, flush=True)


# (a3) ``DataCacheWrapper.forward`` (upstream/cached_loader.py:183) is
# declared as ``forward(self, inp, **other)``.  Modern HF decoder layers
# (Gemma 4, transformers 5.x) pass attention_mask / position_ids /
# position_embeddings as *positional* arguments, which trips a
# TypeError before the activation snapshot is captured.  Replace
# ``forward`` with a permissive variant that round-trips both *args and
# **kwargs to the wrapped module.
# Global stash for the per-arch rotary embedding module.  Set by
# ``register_rotary_emb`` from ``_load_hf_model`` once the model is
# loaded; consumed by the DataCacheWrapper shim when fixing up
# per-layer-type rope mismatches (Gemma 4's sliding vs full attention
# layers have different head_dim, hence different rope output sizes,
# but AnyBCQ caches kwargs once per layer transition).
_ROTARY_EMB = {"module": None}


def register_rotary_emb(rotary_module):
    """Stash a model's rotary_emb so the DataCacheWrapper shim can
    recompute per-layer-type position_embeddings on demand.  No-op
    if called with None (i.e. arch without per-layer-type rope)."""
    _ROTARY_EMB["module"] = rotary_module


def _patch_anybcq_data_cache_wrapper():
    try:
        from anybcq.quantization.cached_loader import (
            DataCacheWrapper, StopForwardException,
        )
    except ImportError:
        return
    import inspect as _inspect
    def _fixup_position_embeddings(module, kwargs):
        """When AnyBCQ replays cached kwargs across layer transitions,
        a stale (cos, sin) sized for the previous layer's head_dim
        can land in a layer whose self_attn has a different head_dim
        (Gemma 4 sliding=256 vs full=512).  Detect the mismatch and
        recompute on the fly using the stashed rotary_emb."""
        pe = kwargs.get("position_embeddings", None)
        if pe is None: return
        sa = getattr(module, "self_attn", None)
        if sa is None: return
        expected_head_dim = getattr(sa, "head_dim", None)
        if expected_head_dim is None: return
        try:
            cos, sin = pe
        except (TypeError, ValueError):
            return
        if cos.shape[-1] == expected_head_dim:
            return  # already matches; nothing to fix.
        rotary = _ROTARY_EMB["module"]
        if rotary is None:
            return
        pos_ids = kwargs.get("position_ids", None)
        if pos_ids is None:
            return
        layer_type = getattr(sa, "layer_type", None) \
            or ("sliding_attention" if getattr(sa, "is_sliding", False)
                 else "full_attention")
        try:
            new_pe = rotary(cos, pos_ids, layer_type=layer_type)
        except (TypeError, RuntimeError, AttributeError):
            try:
                new_pe = rotary(cos, pos_ids)
            except (TypeError, RuntimeError, AttributeError):
                return
        kwargs["position_embeddings"] = new_pe

    _debug_once_key = ["forward"]
    def forward(self, inp, *args, **kwargs):
        self.inp_data = inp
        # Per-layer-type rope fixup (Gemma 4).  Mutates kwargs in place.
        _fixup_position_embeddings(self.module, kwargs)
        # One-shot dtype debug.
        if _debug_once_key:
            import sys as _sys
            _kw = {k: (str(v.dtype) if hasattr(v, "dtype") else type(v).__name__)
                   for k, v in kwargs.items()}
            print(f"[shim-debug] wrapping={type(self.module).__name__} "
                  f"inp.dtype={inp.dtype} kwargs={_kw}",
                  file=_sys.stderr, flush=True)
            _debug_once_key.clear()
        output = self.module(inp, *args, **kwargs)
        self.out_data = output
        # Upstream GetLayerInpOut_q replays via ``q_block(model_input,
        # **others)``, so ``other_data`` must be a flat kwargs dict.
        # Bind ``args`` back into kwargs using the wrapped module's
        # forward signature; drop the leading ``hidden_states`` slot
        # since it's the positional input.
        try:
            sig = _inspect.signature(self.module.forward)
            names = list(sig.parameters)
            if names and len(args) > 0:
                bind_names = names[1 : 1 + len(args)]
                args_as_kwargs = dict(zip(bind_names, args))
            else:
                args_as_kwargs = {}
        except (TypeError, ValueError):
            args_as_kwargs = {}
        self.other_data = {**args_as_kwargs, **kwargs}
        raise StopForwardException
    DataCacheWrapper.forward = forward
    import sys
    print("[upstream_shims] DataCacheWrapper.forward patched to accept *args, **kwargs",
          file=sys.stderr, flush=True)
_patch_anybcq_data_cache_wrapper()


# (a2) DeepSeek-MoE-16B's bundled `modeling_deepseek.py` (loaded via
# trust_remote_code=True) imports ``is_torch_fx_available`` from
# ``transformers.utils.import_utils``. transformers ≥ 5.x dropped that
# symbol. Inject a no-op replacement so the dynamic module load
# succeeds; the function is only used to gate optional torch.fx
# tracing, which encoding doesn't exercise.
try:
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: False
except ImportError:
    pass


@contextlib.contextmanager
def patch_upstream_skip(skip_predicate: Callable[[str], bool]):
    """During `minimize()`, only quantize nn.Linears for which
    `skip_predicate(name)` is False (i.e. NOT skip).

    Upstream `swap_quant_model` hard-codes the skip list (`lm_head`,
    `project`) at swap_linear.py:96-98. We monkey-patch the whole
    function for the duration of `minimize()` with an adapter-driven
    skip predicate, then restore on exit.

    Important: ``anybcq.quantization.anybcq`` binds the symbol via
    ``from anybcq.utils.swap_linear import swap_quant_model`` at
    module load, which captures the ORIGINAL reference.  Patching
    only ``anybcq.utils.swap_linear`` would leave the production
    call site (``anybcq.minimize()`` line 125) using the original.
    We patch BOTH module namespaces.
    """
    from anybcq.utils import swap_linear as _swap
    from anybcq.quantization import anybcq as _anybcq_mod
    original_in_swap = _swap.swap_quant_model
    original_in_anybcq = getattr(_anybcq_mod, "swap_quant_model", None)

    def patched(model, n_bits=32, n_rounds=0, group_size=-1, in_ch_wise=False,
                packing=True, asymmetric=True, swap_type='bcq',
                linear_type='bcq', wq_params=None, aq_params=None):
        return original_in_swap(
            model,
            n_bits=n_bits,
            n_rounds=n_rounds,
            group_size=group_size,
            in_ch_wise=in_ch_wise,
            packing=packing,
            asymmetric=asymmetric,
            swap_type=swap_type,
            linear_type=linear_type,
            wq_params=wq_params,
            aq_params=aq_params,
            skip_predicate=skip_predicate,
        )

    _swap.swap_quant_model = patched
    if original_in_anybcq is not None:
        _anybcq_mod.swap_quant_model = patched
    try:
        yield
    finally:
        _swap.swap_quant_model = original_in_swap
        if original_in_anybcq is not None:
            _anybcq_mod.swap_quant_model = original_in_anybcq
