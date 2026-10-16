"""The any-precision (SharedScale AnyBCQ) encoder of DPA-MoE.

DPA-MoE's optimizer is used as is; it is loaded by path so that its GGUF
adapters and the vendored `anybcq` packaging chain (lm_eval, peft) are not
needed to encode one tensor. Point DPAMOE_DIR at a checkout of DPA-MoE
(https://github.com/kimGreg/DPA-MoE).
"""
from __future__ import annotations

import importlib.util
import sys
import types

from ocdpa.paths import DPAMOE

VENDOR = DPAMOE / "src" / "third-party" / "anybcq" / "anybcq"


def _load():
    if "optimizer" in sys.modules:
        return sys.modules["optimizer"].compute_codebook
    if not VENDOR.exists():
        raise FileNotFoundError(f"DPA-MoE not found at {DPAMOE}; set DPAMOE_DIR to a checkout of it")
    pkg = types.ModuleType("anybcq"); pkg.__path__ = [str(VENDOR)]
    sub = types.ModuleType("anybcq.quantization"); sub.__path__ = [str(VENDOR / "quantization")]
    sys.modules["anybcq"], sys.modules["anybcq.quantization"] = pkg, sub
    for name, path in [("anybcq.quantization.bcq_asym", VENDOR / "quantization" / "bcq_asym.py"),
                       ("optimizer", DPAMOE / "src" / "chunked_encoder" / "optimizer.py")]:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["optimizer"].compute_codebook


def compute_codebook(*args, **kwargs):
    """compute_codebook(W, base_precision=2, target_precision=8, group_size=128) -> (B, alpha, beta, n, m, J)"""
    return _load()(*args, **kwargs)
