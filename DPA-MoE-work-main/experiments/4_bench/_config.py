"""Shared paths, config registry, and benchmark settings for 4_bench.

All scripts import from here so paths and configs are defined exactly once.
Config controls:
  ROOT_CONFIG  root model/profile config.
  BENCH_CONFIG benchmark-only config; defaults to
               experiments/config/<model-family>/bench.yaml.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# ── root paths ────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[2]
ROOT_CONFIG = Path(os.environ.get(
    "ROOT_CONFIG", str(REPO / "experiments/config/Qwen3-30B-A3B/base.yaml")))


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


_root = _read_yaml(ROOT_CONFIG)
_root_model = _root.get("model", {}) if isinstance(_root.get("model"), dict) else {}
_root_model_family = str(_root_model.get("family", "Qwen3-30B-A3B"))
_default_bench_config = REPO / "experiments/config" / _root_model_family / "bench.yaml"
_bench_config_env = os.environ.get("BENCH_CONFIG")
BENCH_CONFIG = Path(_bench_config_env) if _bench_config_env else _default_bench_config

if not BENCH_CONFIG.exists():
    raise FileNotFoundError(f"benchmark config not found: {BENCH_CONFIG}")

_bench_config = _read_yaml(BENCH_CONFIG)
_m = _deep_merge(_root, _bench_config)

_bench   = _m["benchmark"]
_profile = _m.get("profile", {})
MODEL    = str(_m.get("model", {}).get("family", _root_model_family))
BENCHMARK_SETTINGS = _bench


def _as_list(value, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x) for x in value]


def _k_suffix(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", "_")


def _expand_config_groups(value, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if not isinstance(value, dict):
        return _as_list(value, default)
    configs: list[str] = []
    for name in _as_list(value.get("baseline")):
        configs.append(name)
    for k in _as_list(value.get("uniform_k")):
        configs.append(f"uniform_k{_k_suffix(k)}")
    for k in _as_list(value.get("dynamic_k")):
        configs.append(f"dynamic_k{_k_suffix(k)}")
    for name in _as_list(value.get("extra")):
        configs.append(name)
    return configs


def _benchmarks_for(name: str, default: list[str]) -> list[str]:
    benchmarks = _bench.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        return list(default)
    return _as_list(benchmarks.get(name), default)

# ── file paths ────────────────────────────────────────────────────────────────
BENCH_RESULTS_DIR = REPO / "experiments/results" / MODEL / "bench"
KLD_RESULTS_DIR   = REPO / "experiments/results" / MODEL / "kld"
SPEED_RESULTS_DIR = REPO / "experiments/results" / MODEL / "speed"
RESULTS_DIR       = BENCH_RESULTS_DIR
BASELINE_DIR = REPO / "cache/models" / MODEL / "baseline"
SS_MODEL     = REPO / "cache/models" / MODEL / "chunked/ss_anybcq.gguf"
HF_DIR       = REPO / "cache/models" / MODEL / "hf"
DATA         = REPO / "cache/datasets"
BIN_SERVER   = Path(os.environ.get(
    "LLAMA_SERVER_BIN", str(REPO / "third_party/llama.cpp/build-dpa-moe/bin/llama-server")))
PYTHON       = Path(os.environ.get("PYTHON", str(REPO / ".venv/bin/python")))
PROFILE_FILE = REPO / str(_bench.get(
    "dynamic_profile_file",
    _profile.get(
        "runtime_bin",
        f"cache/experiment/{MODEL}/3_dynamic_precision/bin/expert_residuals.bin",
    ),
))

# ── bench hyperparams (from config.benchmark) ────────────────────────────────
MAX_TOKENS = int(_bench.get("max_tokens", 8192))
SEED       = int(_bench.get("seed", _m.get("seed", 0)))
FAST_MODE  = bool(_bench.get("fast_mode", True))
VRAM_CAP   = str(_bench.get("full_pin_vram_cap_mb" if FAST_MODE else "vram_cap_mb",
                            40000))
ALLOW_CAPTURE = "1" if _bench.get("allow_capture", True) else "0"
PIN_ALL       = "1" if _bench.get("pin_all", True) else "0"
BENCHMARK_GPUS = _as_list(_bench.get("gpu"), [str(_m.get("hardware", {}).get("gpu_pin", 0))])
RESULT_SUFFIX = str(_bench.get("result_suffix", ""))
LLAMA_MC_TASKS = _benchmarks_for("llama-perplexity", ["arc_c", "arc_e", "piqa", "winogrande", "hellaswag"])
_llama_ppl = _bench.get("llama_perplexity", {})
LLAMA_MC_CTX = int(_llama_ppl.get("ctx_size", 4096))
LLAMA_MC_BATCH = int(_llama_ppl.get("batch_size", 2048))
LLAMA_MC_UBATCH = int(_llama_ppl.get("ubatch_size", 512))
LLAMA_MC_SERIAL_NP1 = bool(_llama_ppl.get("serial_np1", False))
_n_parallel_cfg = _llama_ppl.get("n_parallel", {})
N_PARALLEL_BASELINE = int(_n_parallel_cfg.get("baseline", 8))
N_PARALLEL_UNIFORM = int(_n_parallel_cfg.get("uniform", 8))
N_PARALLEL_DYNAMIC = int(_n_parallel_cfg.get("dynamic", 1))
N_PARALLEL_SCHED = int(_n_parallel_cfg.get("sched", 1))
_wikitext = _bench.get("wikitext_ppl", {})
WIKITEXT_FILE = REPO / str(_wikitext.get("dataset", "cache/datasets/wikitext-2-raw-test.txt"))
WIKITEXT_PROFILE_FILE = REPO / str(_wikitext.get("dynamic_profile_file", PROFILE_FILE))
WIKITEXT_CTX = int(_wikitext.get("ctx_size", 4096))
WIKITEXT_CHUNKS = int(_wikitext.get("chunks", 10))
WIKITEXT_BATCH = int(_wikitext.get("batch_size", 1))
WIKITEXT_UBATCH = int(_wikitext.get("ubatch_size", 1))
WIKITEXT_SEED = int(_wikitext.get("seed", SEED))

# ── config registry ───────────────────────────────────────────────────────────
# (model_path, extra_env, n_parallel)
# n_parallel=8  → baselines: independent requests, fully safe to batch
# n_parallel=4  → uniform: fixed K for all experts/tokens, concurrent requests safe
# n_parallel=1  → dynamic/sched: dynamic allocator assigns per-expert K per generation;
#                  only one live generation at a time.  -b/-ub>1 is fine (prefill
#                  batching within a single request) but multiple concurrent requests
#                  would race on the allocator state.
_CLEAR = {k: "" for k in [
    "DPA_MOE_KBAR", "DPA_MOE_KBAR_ALLOCATOR", "DPA_MOE_KBAR_PROFILE_FILE",
    "DPA_MOE_KBAR_SCHEDULE_TOKENS", "DPA_MOE_KBAR_SCHEDULE_VALUES",
    "DPA_MOE_KBAR_PP_BEGIN", "DPA_MOE_KBAR_PP_END",
    "DPA_MOE_KBAR_TG_BEGIN", "DPA_MOE_KBAR_TG_END",
    "DPA_MOE_KBAR_TG_DECAY_START", "DPA_MOE_KBAR_TG_DECAY_END",
    "DPA_MOE_KBAR_PP_DECAY_END",
    "DPA_MOE_LOG_KBAR",
    "DPA_MOE_PHASE_REASONING_KBAR", "DPA_MOE_PHASE_GENERATION_KBAR",
    "DPA_MOE_PHASE_OPEN_TOKEN_IDS", "DPA_MOE_PHASE_CLOSE_TOKEN_IDS",
]}

def _ss(kbar: str, alloc: str) -> dict:
    env = {**_CLEAR,
           "DPA_MOE_KBAR": kbar,
            "DPA_MOE_KBAR_ALLOCATOR": alloc,
            "DPA_MOE_ALLOW_CAPTURE": ALLOW_CAPTURE,
            "DPA_MOE_PIN_ALL": PIN_ALL,
            "DPA_MOE_VRAM_CAP_MB": VRAM_CAP}
    if alloc == "profile":
        env["DPA_MOE_KBAR_PROFILE_FILE"] = str(PROFILE_FILE)
    return env

def _sched(pp_begin: str, pp_end: str, tg_begin: str, tg_end: str,
           decay_start: str, decay_end: str) -> dict:
    return {**_CLEAR,
            "DPA_MOE_KBAR": tg_end,
            "DPA_MOE_KBAR_ALLOCATOR": "profile",
            "DPA_MOE_KBAR_PROFILE_FILE": str(PROFILE_FILE),
            "DPA_MOE_KBAR_PP_BEGIN": pp_begin,
            "DPA_MOE_KBAR_PP_END": pp_end,
            "DPA_MOE_KBAR_TG_BEGIN": tg_begin,
            "DPA_MOE_KBAR_TG_END": tg_end,
            "DPA_MOE_KBAR_TG_DECAY_START": decay_start,
            "DPA_MOE_KBAR_TG_DECAY_END": decay_end,
            "DPA_MOE_ALLOW_CAPTURE": ALLOW_CAPTURE,
            "DPA_MOE_PIN_ALL": PIN_ALL,
            "DPA_MOE_VRAM_CAP_MB": VRAM_CAP}


def _baseline_quant(quant: str) -> Path:
    """Return the local baseline GGUF path, accepting dash and dot names."""
    candidates = [
        BASELINE_DIR / f"{MODEL}-{quant}.gguf",
        BASELINE_DIR / f"{MODEL}.{quant}.gguf",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


CONFIGS: dict[str, tuple[Path, dict, int]] = {
    "Q2_K":        (_baseline_quant("Q2_K"),   _CLEAR, N_PARALLEL_BASELINE),
    "Q5_K_M":      (_baseline_quant("Q5_K_M"), _CLEAR, N_PARALLEL_BASELINE),
    "Q8_0":        (_baseline_quant("Q8_0"),   _CLEAR, N_PARALLEL_BASELINE),
    "uniform_k2":  (SS_MODEL, _ss("2", "uniform"), N_PARALLEL_UNIFORM),
    "uniform_k3":  (SS_MODEL, _ss("3", "uniform"), N_PARALLEL_UNIFORM),
    "uniform_k4":  (SS_MODEL, _ss("4", "uniform"), N_PARALLEL_UNIFORM),
    "uniform_k5":  (SS_MODEL, _ss("5", "uniform"), N_PARALLEL_UNIFORM),
    "uniform_k6":  (SS_MODEL, _ss("6", "uniform"), N_PARALLEL_UNIFORM),
    "uniform_k7":  (SS_MODEL, _ss("7", "uniform"), N_PARALLEL_UNIFORM),
    "uniform_k8":  (SS_MODEL, _ss("8", "uniform"), N_PARALLEL_UNIFORM),
    "dynamic_k2":  (SS_MODEL, _ss("2", "profile"),  N_PARALLEL_DYNAMIC),
    "dynamic_k2_5": (SS_MODEL, _ss("2.5", "profile"), N_PARALLEL_DYNAMIC),
    "dynamic_k3":  (SS_MODEL, _ss("3", "profile"),  N_PARALLEL_DYNAMIC),
    "dynamic_k3_5": (SS_MODEL, _ss("3.5", "profile"), N_PARALLEL_DYNAMIC),
    "dynamic_k4":  (SS_MODEL, _ss("4", "profile"),  N_PARALLEL_DYNAMIC),
    "dynamic_k4_5": (SS_MODEL, _ss("4.5", "profile"), N_PARALLEL_DYNAMIC),
    "dynamic_k5":  (SS_MODEL, _ss("5", "profile"),  N_PARALLEL_DYNAMIC),
    "dynamic_k5_5": (SS_MODEL, _ss("5.5", "profile"), N_PARALLEL_DYNAMIC),
    "dynamic_k6":  (SS_MODEL, _ss("6", "profile"),  N_PARALLEL_DYNAMIC),
    "dynamic_k6_5": (SS_MODEL, _ss("6.5", "profile"), N_PARALLEL_DYNAMIC),
    "dynamic_k7":  (SS_MODEL, _ss("7", "profile"),  N_PARALLEL_DYNAMIC),
    "dynamic_k7_5": (SS_MODEL, _ss("7.5", "profile"), N_PARALLEL_DYNAMIC),
    "sched_k2_k4": (SS_MODEL, _sched("2", "2", "2", "4", "0", "100"), N_PARALLEL_SCHED),
    "sched_k2_k6": (SS_MODEL, _sched("2", "2", "2", "6", "0", "100"), N_PARALLEL_SCHED),
    "sched_k4_k8": (SS_MODEL, _sched("4", "4", "4", "8", "0", "100"), N_PARALLEL_SCHED),
    # ARC-C prefill-only budget drop experiment.  llama-perplexity now calls
    # dpa_moe_set_prefill_decay_end(seq_len, reset=1) before each MC decode,
    # so PP decay ends at the exact tokenized input sequence length instead of
    # a fixed batch-size horizon.
    "sched_pp_k5_k2_b256": (
        SS_MODEL,
        {**_sched("5", "2", "2", "2", "0", "0"),
         "DPA_MOE_LOG_KBAR": "1"},
        N_PARALLEL_SCHED,
    ),
    # Primary scheduled-prefill benchmark.  PP decays from K5 to K2.5 over
    # the exact tokenized input sequence; TG stays fixed at the PP-end budget.
    "sched_pp_k5_k2_5": (
        SS_MODEL,
        {**_sched("5", "2.5", "2.5", "2.5", "0", "0"),
         "DPA_MOE_LOG_KBAR": "1"},
        N_PARALLEL_SCHED,
    ),
    "sched_pp_k6_k2_5": (
        SS_MODEL,
        {**_sched("6", "2.5", "2.5", "2.5", "0", "0"),
         "DPA_MOE_LOG_KBAR": "1"},
        N_PARALLEL_SCHED,
    ),
    # Historical alias for the first validation run; keep readable results
    # reproducible without making it the primary config name.
    "sched_pp_k6_k2_5_b256": (
        SS_MODEL,
        {**_sched("6", "2.5", "2.5", "2.5", "0", "0"),
         "DPA_MOE_LOG_KBAR": "1"},
        N_PARALLEL_SCHED,
    ),
    # MMMLU-sw case-study schedules: uniform PP, then TG decays to K2 from
    # generated token 64 through 128.
    "sched_pp_k6_tg_k6_k2_64_128": (
        SS_MODEL,
        {**_sched("6", "6", "6", "2", "64", "128"),
         "DPA_MOE_LOG_KBAR": "1"},
        N_PARALLEL_SCHED,
    ),
    "sched_pp_k5_tg_k5_k2_64_128": (
        SS_MODEL,
        {**_sched("5", "5", "5", "2", "64", "128"),
         "DPA_MOE_LOG_KBAR": "1"},
        N_PARALLEL_SCHED,
    ),
    "sched_pp_k7_tg_k5_k2_64_128": (
        SS_MODEL,
        {**_sched("7", "7", "5", "2", "64", "128"),
         "DPA_MOE_LOG_KBAR": "1"},
        N_PARALLEL_SCHED,
    ),
    "sched_pp_k7_tg_k4_k2_64_128": (
        SS_MODEL,
        {**_sched("7", "7", "4", "2", "64", "128"),
         "DPA_MOE_LOG_KBAR": "1"},
        N_PARALLEL_SCHED,
    ),
}

BENCHMARK_CONFIGS = _expand_config_groups(_bench.get("config"), list(CONFIGS))
_unknown_configs = [cfg for cfg in BENCHMARK_CONFIGS if cfg not in CONFIGS]
if _unknown_configs:
    raise ValueError(f"config benchmark.configs contains unknown configs: {_unknown_configs}")


def _build_gpu_map() -> dict[str, str]:
    explicit = _bench.get("gpu_map", {})
    if isinstance(explicit, dict) and explicit:
        return {str(k): str(v) for k, v in explicit.items()}
    return {
        cfg: BENCHMARK_GPUS[i % len(BENCHMARK_GPUS)]
        for i, cfg in enumerate(BENCHMARK_CONFIGS)
    }


GPU_MAP: dict[str, str] = _build_gpu_map()

def is_single_flight_config(config: str, extra_env: dict | None = None) -> bool:
    """Configs whose precision state is not safe across concurrent requests."""
    if config.startswith(("dynamic_", "sched_")):
        return True
    if extra_env:
        allocator = extra_env.get("DPA_MOE_KBAR_ALLOCATOR", "")
        if allocator == "profile" or extra_env.get("DPA_MOE_KBAR_TG_DECAY_END"):
            return True
    return False


def validate_n_parallel(config: str, extra_env: dict, n_parallel: int) -> int:
    if is_single_flight_config(config, extra_env) and n_parallel != 1:
        raise ValueError(
            f"{config} must run with n_parallel=1: dynamic/sched precision "
            "state is per dispatch and would mix concurrent requests"
        )
    return n_parallel
