"""Config-backed paths for dynamic precision profiling and KLD benchmarks."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
EXP_ROOT = _HERE
REPO_ROOT = _HERE.parent.parent
SCRIPTS_DIR = _HERE
CACHE_EXP_ROOT = REPO_ROOT / "cache" / "experiment"
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
CONFIG = Path(os.environ.get(
    "CONFIG", str(REPO_ROOT / "experiments/config/Qwen3-30B-A3B/base.yaml")))


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _resolve(path: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def active_variant() -> dict[str, str]:
    """Return model paths from the active experiment config.

    Selecting a different model means selecting a different ``CONFIG``.
    """
    m = _config()
    model = m.get("model", {})
    profile = m.get("profile", {})
    family = str(model["family"])
    ggufs = model.get("ggufs", {})
    gguf_path = ggufs.get("ss_anybcq") or ggufs.get("anybcq")
    if not gguf_path:
        gguf_path = f"cache/models/{family}/chunked/ss_anybcq.gguf"
    return {
        "name": family,
        "gguf_path": _resolve(gguf_path),
        "bf16_gguf": _resolve(model.get(
            "bf16_gguf",
            f"cache/models/{family}/baseline/{family}-BF16.gguf",
        )),
        "residuals_npz": _resolve(profile.get(
            "residuals_npz",
            f"cache/experiment/{family}/layout_solver/expert_residuals.npz",
        )),
        "encoder_flag": str(profile.get("encoding", "ss_anybcq")),
        "bits_non_moe": str(profile.get("bits_non_moe", 8.0)),
    }


def variant_name() -> str:
    return active_variant()["name"]


def shared_root() -> Path:
    return CACHE_EXP_ROOT / f"_shared_{variant_name()}"


def sweep_dir() -> Path:
    d = CACHE_EXP_ROOT / variant_name() / "3_dynamic_precision"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    d = RESULTS_ROOT / variant_name()
    d.mkdir(parents=True, exist_ok=True)
    return d


def ppl_bin() -> Path:
    return REPO_ROOT / "src" / "llama.cpp" / "build-dpa-moe" / "bin" / "llama-perplexity"


def model_dims() -> tuple[int, int]:
    """Return ``(n_layers, n_experts)`` from the config profile residuals."""
    res = Path(active_variant()["residuals_npz"])
    if not res.exists():
        raise SystemExit(
            f"missing residuals npz for variant {variant_name()!r}: {res}\n"
            "Run experiments/3_dynamic_precision/run.sh first, or set "
            "profile.residuals_npz in the config.")
    import numpy as np
    d = np.load(res, allow_pickle=True)
    R = d["R"]
    return int(R.shape[0]), int(R.shape[1])
