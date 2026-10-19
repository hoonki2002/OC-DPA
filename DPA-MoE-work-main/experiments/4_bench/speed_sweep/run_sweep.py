#!/usr/bin/env python3
"""Standalone text-prompt speed sweep for DPA-MoE.

This intentionally does not use downstream benchmark prompts. It measures
single-request serving speed on controlled text lengths with prompt cache off.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
REPO = HERE.parents[2]

sys.path.insert(0, str(BENCH_ROOT))
import _config as cfg

MODEL_FAMILY = os.environ.get("MODEL", cfg.MODEL)
ROOT_CONFIG = cfg.ROOT_CONFIG
BENCH_CONFIG = cfg.BENCH_CONFIG
BENCH_CFG = cfg.BENCHMARK_SETTINGS
SPEED_CFG = BENCH_CFG.get("speed_sweep", {})
if not isinstance(SPEED_CFG, dict):
    SPEED_CFG = {}

SPEED_RESULT_SUBDIR = os.environ.get("SPEED_RESULT_SUBDIR", "speed")
SPEED_RESULTS_ROOT = REPO / "experiments/results" / MODEL_FAMILY / SPEED_RESULT_SUBDIR
SPEED_RUN_ROOT = REPO / "cache/experiment" / MODEL_FAMILY / SPEED_RESULT_SUBDIR
RESULTS_DIR = SPEED_RESULTS_ROOT
RUN_ROOT = SPEED_RUN_ROOT
REQUESTS_JSONL = RESULTS_DIR / "requests.jsonl"
EXAMPLES_JSONL = RESULTS_DIR / "generation_examples.jsonl"
SUMMARY_CSV = RESULTS_DIR / "speed_summary.csv"
PLOT_POINTS_CSV = RESULTS_DIR / "speed_plot_points.csv"
PLOTS_HTML = RESULTS_DIR / "speed_plots.html"

CLEAR_DPA_MOE = {
    "DPA_MOE_KBAR", "DPA_MOE_KBAR_ALLOCATOR", "DPA_MOE_KBAR_PROFILE_FILE",
    "DPA_MOE_KBAR_SCHEDULE_TOKENS", "DPA_MOE_KBAR_SCHEDULE_VALUES",
    "DPA_MOE_KBAR_PP_BEGIN", "DPA_MOE_KBAR_PP_END",
    "DPA_MOE_KBAR_TG_BEGIN", "DPA_MOE_KBAR_TG_END",
    "DPA_MOE_KBAR_TG_DECAY_START", "DPA_MOE_KBAR_TG_DECAY_END",
    "DPA_MOE_KBAR_PP_DECAY_END", "DPA_MOE_ALLOW_CAPTURE", "DPA_MOE_PIN_ALL",
    "DPA_MOE_BATCHED_BACKEND", "DPA_MOE_STATS", "DPA_MOE_LOG_KBAR",
    "DPA_MOE_HOST_CACHE", "DPA_MOE_DISABLE_HOST_CACHE", "DPA_MOE_HOST_CAP_MB",
    "DPA_MOE_DIRECT_REFRESH_EVERY_DECODE",
    "DPA_MOE_KERNEL_OPTIMIZED", "DPA_MOE_OPTIMIZED_BATCH_KERNEL",
    "DPA_MOE_KERNEL_OPTIMIZED_BATCH", "DPA_MOE_KERNEL_OPTIMIZED_SPAN",
    "DPA_MOE_FUSED_MIXED_MODE", "DPA_MOE_FUSED_PLANE_GROUP",
    "DPA_MOE_FUSED_FLAT_PLANES", "DPA_MOE_DYNAMIC_GPU_PLANNER",
    "DPA_MOE_COMPARE_GPU_CPU_PLANNER", "DPA_MOE_PROFILE_PLANNER",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: pathlib.Path
    n_samples: int
    chat_template: bool = False


@dataclass(frozen=True)
class LengthSpec:
    name: str
    prompt_tokens: int
    generate_tokens: int


@dataclass(frozen=True)
class SpeedMode:
    name: str
    series: str
    model_path: pathlib.Path
    vram_cap_mb: int
    allocator: str
    kbar: str
    allow_capture: bool
    pin_all: bool
    env: dict[str, str]


def _as_path(value: str) -> pathlib.Path:
    p = pathlib.Path(value)
    return p if p.is_absolute() else REPO / p


def _format_k(k: float) -> str:
    return f"{k:.1f}" if abs(k - round(k)) > 1e-6 else str(int(round(k)))


def _mode_name(prefix: str, k: float) -> str:
    return f"{prefix}_k{_format_k(k).replace('.', '_')}"


def _k_from_config_name(name: str, prefix: str) -> float | None:
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix):].replace("_", "."))
    except ValueError:
        return None


def _cap_label(vram_cap_mb: int) -> str:
    if vram_cap_mb % 1000 == 0:
        return f"cap{vram_cap_mb // 1000}g"
    if vram_cap_mb % 1024 == 0:
        return f"cap{vram_cap_mb // 1024}g"
    return f"cap{vram_cap_mb}mb"


def _mode_with_cap(name: str, vram_cap_mb: int, multi_cap: bool) -> str:
    return f"{name}_{_cap_label(vram_cap_mb)}" if multi_cap else name


def _setting_name(mode: SpeedMode, length: LengthSpec, ubatch: int) -> str:
    host_cap_mb = SPEED_CFG.get("host_cap_mb")
    if host_cap_mb is None:
        return f"{length.name}_ub{ubatch}_{_cap_label(mode.vram_cap_mb)}"
    return f"{length.name}_ub{ubatch}_{_cap_label(mode.vram_cap_mb)}_host{_cap_label(int(host_cap_mb))}"


def _run_label(mode: SpeedMode, ubatch: int) -> str:
    host_cap_mb = SPEED_CFG.get("host_cap_mb")
    if host_cap_mb is None:
        return f"ub{ubatch}_{_cap_label(mode.vram_cap_mb)}"
    return f"ub{ubatch}_{_cap_label(mode.vram_cap_mb)}_host{_cap_label(int(host_cap_mb))}"


def _set_output_paths(mode: SpeedMode, length: LengthSpec, ubatch: int) -> None:
    global RESULTS_DIR, RUN_ROOT, REQUESTS_JSONL, EXAMPLES_JSONL, SUMMARY_CSV
    global PLOT_POINTS_CSV, PLOTS_HTML
    setting = _setting_name(mode, length, ubatch)
    RESULTS_DIR = SPEED_RESULTS_ROOT / mode.name / setting
    RUN_ROOT = SPEED_RUN_ROOT / mode.name / setting
    REQUESTS_JSONL = RESULTS_DIR / "requests.jsonl"
    EXAMPLES_JSONL = RESULTS_DIR / "generation_examples.jsonl"
    SUMMARY_CSV = RESULTS_DIR / "speed_summary.csv"
    PLOT_POINTS_CSV = RESULTS_DIR / "speed_plot_points.csv"
    PLOTS_HTML = RESULTS_DIR / "speed_plots.html"


def _vram_caps() -> list[int]:
    raw = SPEED_CFG.get("vram_cap_mbs", SPEED_CFG.get("vram_cap_mb", BENCH_CFG.get("vram_cap_mb", 40000)))
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return [int(raw)]


def _profile_file() -> pathlib.Path:
    p = SPEED_CFG.get("dynamic_profile_file") or BENCH_CFG.get("dynamic_profile_file")
    if p:
        return _as_path(str(p))
    return cfg.PROFILE_FILE


def _flag(value: Any) -> str:
    return "1" if bool(value) else "0"


def _speed_env() -> dict[str, str]:
    env = {str(k): str(v) for k, v in (SPEED_CFG.get("env", {}) or {}).items()}

    kernel = SPEED_CFG.get("kernel", {})
    if isinstance(kernel, dict):
        if "optimized" in kernel:
            env["DPA_MOE_KERNEL_OPTIMIZED"] = _flag(kernel["optimized"])
        if "optimized_batch" in kernel:
            batch = _flag(kernel["optimized_batch"])
            env["DPA_MOE_OPTIMIZED_BATCH_KERNEL"] = batch
            env["DPA_MOE_KERNEL_OPTIMIZED_BATCH"] = batch
        if "optimized_span" in kernel:
            env["DPA_MOE_KERNEL_OPTIMIZED_SPAN"] = str(kernel["optimized_span"])
        if "fused_mixed_mode" in kernel:
            env["DPA_MOE_FUSED_MIXED_MODE"] = str(kernel["fused_mixed_mode"])
        if "fused_plane_group" in kernel:
            env["DPA_MOE_FUSED_PLANE_GROUP"] = str(kernel["fused_plane_group"])
        if "batched_backend" in kernel:
            env["DPA_MOE_BATCHED_BACKEND"] = str(kernel["batched_backend"])

    scheduler = SPEED_CFG.get("scheduler", {})
    if isinstance(scheduler, dict):
        if "dynamic_gpu_planner" in scheduler:
            env["DPA_MOE_DYNAMIC_GPU_PLANNER"] = _flag(scheduler["dynamic_gpu_planner"])
        if "compare_gpu_cpu_planner" in scheduler:
            env["DPA_MOE_COMPARE_GPU_CPU_PLANNER"] = _flag(scheduler["compare_gpu_cpu_planner"])
        if "profile_planner" in scheduler:
            env["DPA_MOE_PROFILE_PLANNER"] = _flag(scheduler["profile_planner"])

    return env


def _ignore_eos() -> bool:
    return bool(SPEED_CFG.get("ignore_eos", False))


def _base_env(gpu: str, mode: SpeedMode, diag: bool) -> dict[str, str]:
    env = {**os.environ}
    for key in CLEAR_DPA_MOE:
        env.pop(key, None)
    env.update(mode.env)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["DPA_MOE_VRAM_CAP_MB"] = str(mode.vram_cap_mb)
    env["DPA_MOE_ALLOW_CAPTURE"] = "1" if mode.allow_capture else "0"
    env["DPA_MOE_PIN_ALL"] = "1" if mode.pin_all else "0"
    env["DPA_MOE_HOST_CACHE"] = str(SPEED_CFG.get("host_cache", 0))
    host_cap_mb = int(SPEED_CFG.get("host_cap_mb", 0) or 0)
    if str(SPEED_CFG.get("host_cache", 0)) != "0" and host_cap_mb > 0:
        env["DPA_MOE_HOST_CAP_MB"] = str(host_cap_mb)
    # Speed sweeps always need request-level cache hit/miss counters.
    # Without this, /dpa_moe/stats cannot be audited from the run logs.
    env["DPA_MOE_STATS"] = "1"
    if diag:
        env["DPA_MOE_LOG_KBAR"] = "1"
    env.update(_speed_env())
    return env


def _ss_env(kbar: str, allocator: str, vram_cap_mb: int,
            allow_capture: bool, pin_all: bool) -> dict[str, str]:
    env = {
        "DPA_MOE_KBAR": kbar,
        "DPA_MOE_KBAR_ALLOCATOR": allocator,
        "DPA_MOE_VRAM_CAP_MB": str(vram_cap_mb),
        "DPA_MOE_ALLOW_CAPTURE": "1" if allow_capture else "0",
        "DPA_MOE_PIN_ALL": "1" if pin_all else "0",
    }
    if allocator == "profile":
        env["DPA_MOE_KBAR_PROFILE_FILE"] = str(_profile_file())
    return env


def _configured_modes() -> list[SpeedMode]:
    raw_modes = SPEED_CFG.get("modes")
    if raw_modes:
        modes: list[SpeedMode] = []
        for raw in raw_modes:
            base_name = str(raw["name"])
            series = str(raw.get("series", "other"))
            model_path = _as_path(str(raw.get("model_path", cfg.SS_MODEL)))
            allocator = str(raw.get("allocator", "uniform"))
            kbar = str(raw.get("kbar", ""))
            allow_capture = bool(raw.get("allow_capture", SPEED_CFG.get("allow_capture", False)))
            pin_all = bool(raw.get("pin_all", SPEED_CFG.get("pin_all", False)))
            raw_caps = raw.get("vram_cap_mbs", raw.get("vram_cap_mb"))
            if raw_caps is None:
                caps = _vram_caps()
            elif isinstance(raw_caps, list):
                caps = [int(x) for x in raw_caps]
            else:
                caps = [int(raw_caps)]
            multi_cap = len(caps) > 1
            for vram in caps:
                name = _mode_with_cap(base_name, vram, multi_cap)
                env = {str(k): str(v) for k, v in (raw.get("env", {}) or {}).items()}
                if allocator in {"uniform", "profile"}:
                    env.update(_ss_env(kbar, allocator, vram, allow_capture, pin_all))
                modes.append(SpeedMode(name, series, model_path, vram, allocator, kbar, allow_capture, pin_all, env))
        return modes

    caps = _vram_caps()
    multi_cap = len(caps) > 1
    allow_capture = bool(SPEED_CFG.get("allow_capture", False))
    pin_all = bool(SPEED_CFG.get("pin_all", False))
    modes: list[SpeedMode] = []
    for vram in caps:
        for quant in [c for c in cfg.BENCHMARK_CONFIGS if c in {"Q2_K", "Q5_K_M", "Q8_0"}]:
            path = cfg.CONFIGS[quant][0] if quant in cfg.CONFIGS else cfg.SS_MODEL
            name = _mode_with_cap(quant, vram, multi_cap)
            modes.append(SpeedMode(name, "baseline", path, vram, "baseline", "", allow_capture, pin_all, {}))
        for config_name in [c for c in cfg.BENCHMARK_CONFIGS if c.startswith("uniform_k")]:
            kf = _k_from_config_name(config_name, "uniform_k")
            if kf is None:
                continue
            base_name = _mode_name("uniform", kf)
            name = _mode_with_cap(base_name, vram, multi_cap)
            modes.append(SpeedMode(name, "uniform", cfg.SS_MODEL, vram, "uniform", _format_k(kf), allow_capture, pin_all,
                                   _ss_env(_format_k(kf), "uniform", vram, allow_capture, pin_all)))
        for config_name in [c for c in cfg.BENCHMARK_CONFIGS if c.startswith("dynamic_k")]:
            kf = _k_from_config_name(config_name, "dynamic_k")
            if kf is None:
                continue
            base_name = _mode_name("dynamic", kf)
            name = _mode_with_cap(base_name, vram, multi_cap)
            modes.append(SpeedMode(name, "dynamic", cfg.SS_MODEL, vram, "profile", _format_k(kf), allow_capture, pin_all,
                                   _ss_env(_format_k(kf), "profile", vram, allow_capture, pin_all)))
    return modes


def _mode_from_config_name(config_name: str, vram_cap_mb: int | None = None) -> SpeedMode | None:
    if config_name not in cfg.CONFIGS:
        return None
    model_path, env_cfg, _ = cfg.CONFIGS[config_name]
    caps = [vram_cap_mb] if vram_cap_mb is not None else _vram_caps()
    vram = int(caps[0])
    allow_capture = bool(SPEED_CFG.get("allow_capture", False))
    pin_all = bool(SPEED_CFG.get("pin_all", False))
    if config_name.startswith("uniform_k"):
        kf = _k_from_config_name(config_name, "uniform_k")
        if kf is None:
            return None
        return SpeedMode(
            _mode_with_cap(_mode_name("uniform", kf), vram, False),
            "uniform", cfg.SS_MODEL, vram, "uniform", _format_k(kf),
            allow_capture, pin_all, _ss_env(_format_k(kf), "uniform", vram, allow_capture, pin_all),
        )
    if config_name.startswith("dynamic_k"):
        kf = _k_from_config_name(config_name, "dynamic_k")
        if kf is None:
            return None
        return SpeedMode(
            _mode_with_cap(_mode_name("dynamic", kf), vram, False),
            "dynamic", cfg.SS_MODEL, vram, "profile", _format_k(kf),
            allow_capture, pin_all, _ss_env(_format_k(kf), "profile", vram, allow_capture, pin_all),
        )
    series = "baseline" if config_name in {"Q2_K", "Q5_K_M", "Q8_0"} else "other"
    env = {str(k): str(v) for k, v in env_cfg.items() if v}
    return SpeedMode(config_name, series, model_path, vram, "baseline", "", allow_capture, pin_all, env)


def _datasets() -> list[DatasetSpec]:
    raw = SPEED_CFG.get("datasets") or [
        {"name": "wikitext_train", "path": "cache/datasets/wikitext-2-raw-train.txt", "n_samples": 50},
        {"name": "c4", "path": "cache/datasets/c4_text.txt", "n_samples": 50},
        {"name": "slimpajama", "path": "cache/datasets/slimpajama_text.txt", "n_samples": 50},
        {"name": "sharegpt", "path": "cache/datasets/sharegpt_prompts.txt", "n_samples": 50},
    ]
    specs = []
    sample_override = os.environ.get("SAMPLES_PER_DATASET")
    for item in raw:
        if isinstance(item, str):
            n_samples = int(sample_override) if sample_override else 50
            specs.append(DatasetSpec(pathlib.Path(item).stem, _as_path(item), n_samples, False))
        else:
            if "n_samples" not in item:
                raise ValueError(f"speed_sweep.datasets entry requires n_samples: {item}")
            n_samples = int(sample_override) if sample_override else int(item["n_samples"])
            specs.append(DatasetSpec(str(item["name"]), _as_path(str(item["path"])),
                                     n_samples,
                                     bool(item.get("chat_template", item.get("chat", False)))))
    return specs


def _sample_seed(dataset_name: str) -> int:
    base = int(SPEED_CFG.get("sample_seed", SPEED_CFG.get("seed", BENCH_CFG.get("seed", 0))))
    digest = hashlib.blake2b(dataset_name.encode("utf-8"), digest_size=4).digest()
    return (base + int.from_bytes(digest, "little")) & 0x7fffffff


def _lengths() -> list[LengthSpec]:
    raw = SPEED_CFG.get("lengths")
    if not raw:
        return [LengthSpec("direct", 0, int(SPEED_CFG.get("generate_tokens", 128)))]
    return [
        LengthSpec(str(x.get("name", f"p{x['prompt_tokens']}_g{x['generate_tokens']}")),
                   int(x["prompt_tokens"]), int(x["generate_tokens"]))
        for x in raw
    ]


def _read_texts(spec: DatasetSpec, extra_samples: int = 0) -> list[str]:
    if not spec.path.exists():
        raise FileNotFoundError(f"missing speed dataset {spec.name}: {spec.path}")
    texts: list[str] = []
    if spec.chat_template and spec.path.suffix == ".json":
        data = json.loads(spec.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"ShareGPT speed dataset must be a JSON list: {spec.path}")
        for row in data:
            if isinstance(row, dict):
                conv = row.get("conversations") or row.get("conversation") or row.get("messages") or row
            else:
                conv = row
            if conv:
                texts.append(json.dumps(conv, ensure_ascii=False))
    else:
        with spec.path.open(errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    texts.append(line)
    if not texts:
        raise ValueError(f"empty speed dataset {spec.name}: {spec.path}")
    rng = random.Random(_sample_seed(spec.name))
    rng.shuffle(texts)
    return texts[:spec.n_samples + max(0, extra_samples)]


def _post_json(port: int, path: str, body: dict[str, Any],
               timeout_s: int = 1800) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_json(port: int, path: str, timeout_s: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def _dpa_moe_stats(port: int) -> dict[str, float]:
    try:
        data = _get_json(port, "/dpa_moe/stats", timeout_s=10)
    except Exception:
        return {}
    pool = data.get("pool", {}) if isinstance(data, dict) else {}
    hook = data.get("hook", {}) if isinstance(data, dict) else {}
    residency = data.get("residency", {}) if isinstance(data, dict) else {}
    tier = data.get("logical_tier_hits", data.get("tier_hits", {})) if isinstance(data, dict) else {}
    return {
        "h2d_mb": float(pool.get("h2d_mb", 0.0) or 0.0),
        "h2d_calls": float(pool.get("h2d_calls", 0.0) or 0.0),
        "make_room_calls": float(pool.get("make_room_calls", 0.0) or 0.0),
        "host_dram_mb": float(pool.get("host_dram_mb", 0.0) or 0.0),
        "mc_pread_ns": float(hook.get("mc_pread_ns", 0.0) or 0.0),
        "mc_calls": float(hook.get("mc_calls", 0.0) or 0.0),
        "vram_residency_required_chunks": float(residency.get("required_chunks", 0.0) or 0.0),
        "vram_residency_hits": float(residency.get("resident_hits", 0.0) or 0.0),
        "vram_residency_load_misses": float(residency.get("load_misses", 0.0) or 0.0),
        "prefill_vram_residency_required_chunks": float(residency.get("prefill_required_chunks", 0.0) or 0.0),
        "prefill_vram_residency_hits": float(residency.get("prefill_resident_hits", 0.0) or 0.0),
        "decode_vram_residency_required_chunks": float(residency.get("decode_required_chunks", 0.0) or 0.0),
        "decode_vram_residency_hits": float(residency.get("decode_resident_hits", 0.0) or 0.0),
        "tier_attempts": float(tier.get("attempts", 0.0) or 0.0),
        "tier_vram_hits": float(tier.get("vram_hits", 0.0) or 0.0),
        "tier_dram_hits": float(tier.get("dram_hits", 0.0) or 0.0),
        "tier_ssd_misses": float(tier.get("ssd_misses", 0.0) or 0.0),
        "tier_diag_enabled": 1.0 if tier.get("diag_enabled", False) else 0.0,
    }


def _hit_pct(hits: float, attempts: float) -> float:
    return 100.0 * hits / attempts if attempts > 0.0 else 0.0


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_ready(port: int, proc: subprocess.Popen, timeout_s: int = 900) -> None:
    t0 = time.monotonic()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early ({proc.returncode})")
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"server did not become ready in {timeout_s}s")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)


def _start_server(port: int, gpu: str, mode: SpeedMode, ubatch: int,
                  ctx_size: int, diag: bool, log_path: pathlib.Path) -> subprocess.Popen:
    batch_size = int(SPEED_CFG.get("batch_size", 2048))
    args = [
        str(cfg.BIN_SERVER),
        "-m", str(mode.model_path),
        "-c", str(ctx_size),
        "-np", "1",
        "-b", str(batch_size),
        "-ub", str(ubatch),
        "-ngl", "999",
        "--no-warmup",
        "--no-cache-prompt",
        "--cache-ram", "0",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--seed", str(SPEED_CFG.get("seed", BENCH_CFG.get("seed", 0))),
        "--reasoning", "off",
        "--fit", "off",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("wb")
    proc = subprocess.Popen(
        args,
        env=_base_env(gpu, mode, diag),
        stdin=subprocess.DEVNULL,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_fh = fh  # type: ignore[attr-defined]
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)
    try:
        proc._log_fh.close()  # type: ignore[attr-defined]
    except Exception:
        pass


def _materialize_prompt(port: int, text: str, target_tokens: int) -> str:
    repeated = text
    while True:
        toks = _post_json(port, "/tokenize", {
            "content": repeated,
            "add_special": False,
            "parse_special": False,
        })["tokens"]
        if len(toks) >= target_tokens:
            break
        repeated = (repeated + "\n\n" + text).strip()
    return str(_post_json(port, "/detokenize", {"tokens": toks[:target_tokens]})["content"])


def _tokenize(port: int, text: str) -> list[int]:
    return _post_json(port, "/tokenize", {
        "content": text,
        "add_special": False,
        "parse_special": False,
    })["tokens"]


def _detokenize(port: int, tokens: list[int]) -> str:
    return str(_post_json(port, "/detokenize", {"tokens": tokens})["content"])


def _parse_sharegpt_messages(text: str) -> list[dict[str, str]]:
    def convert(obj: Any) -> dict[str, str] | None:
        if not isinstance(obj, dict):
            return None
        role = str(obj.get("role") or obj.get("from") or "").lower()
        content = obj.get("content", obj.get("value", obj.get("text", "")))
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }
        mapped = role_map.get(role)
        if not mapped or not content:
            return None
        return {"role": mapped, "content": str(content)}

    parsed: Any | None = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    raw_messages: list[Any] = []
    if isinstance(parsed, dict):
        raw = parsed.get("conversations") or parsed.get("conversation") or parsed.get("messages")
        raw_messages = raw if isinstance(raw, list) else [parsed]
    elif isinstance(parsed, list):
        raw_messages = parsed
    else:
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            try:
                obj, pos = decoder.raw_decode(text, pos)
            except json.JSONDecodeError:
                raw_messages = []
                break
            raw_messages.append(obj)

    messages = [m for m in (convert(obj) for obj in raw_messages) if m is not None]
    if not messages:
        return [{"role": "user", "content": text}]
    while messages and messages[-1]["role"] != "user":
        messages.pop()
    return messages or [{"role": "user", "content": text}]


def _apply_chat_template(port: int, messages: list[dict[str, str]]) -> str:
    return str(_post_json(port, "/apply-template", {
        "messages": messages,
        "add_generation_prompt": True,
        "cache_prompt": False,
    })["prompt"])


def _last_user_index(messages: list[dict[str, str]]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    messages.append({"role": "user", "content": ""})
    return len(messages) - 1


def _materialize_chat_messages(port: int, text: str, target_tokens: int) -> list[dict[str, str]]:
    messages = _parse_sharegpt_messages(text)
    source = messages[_last_user_index(messages)]["content"]
    if not source.strip():
        source = text

    def rendered_len(msgs: list[dict[str, str]]) -> int:
        return len(_tokenize(port, _apply_chat_template(port, msgs)))

    while rendered_len(messages) < target_tokens:
        idx = _last_user_index(messages)
        messages[idx] = {**messages[idx], "content": (messages[idx]["content"] + "\n\n" + source).strip()}

    if rendered_len(messages) <= target_tokens:
        return messages

    idx = _last_user_index(messages)
    candidate = [messages[idx]]
    while rendered_len(candidate) < target_tokens:
        candidate[0] = {**candidate[0], "content": (candidate[0]["content"] + "\n\n" + source).strip()}
    content_tokens = _tokenize(port, candidate[0]["content"])
    lo, hi = 1, max(1, len(content_tokens))
    best = candidate[0]["content"]
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = [{**candidate[0], "content": _detokenize(port, content_tokens[:mid])}]
        if rendered_len(trial) <= target_tokens:
            best = trial[0]["content"]
            lo = mid + 1
        else:
            hi = mid - 1
    return [{**candidate[0], "content": best}]


def _completion(port: int, prompt: str, n_predict: int) -> dict[str, Any]:
    stats0 = _dpa_moe_stats(port)
    t0 = time.monotonic()
    resp = _post_json(port, "/completion", {
        "prompt": prompt,
        "n_predict": n_predict,
        "ignore_eos": _ignore_eos(),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "cache_prompt": False,
        "stream": False,
    })
    resp["_wall_s"] = time.monotonic() - t0
    stats1 = _dpa_moe_stats(port)
    if stats0 or stats1:
        for key in ("h2d_mb", "h2d_calls", "make_room_calls", "mc_pread_ns", "mc_calls",
                    "vram_residency_required_chunks", "vram_residency_hits",
                    "vram_residency_load_misses",
                    "prefill_vram_residency_required_chunks", "prefill_vram_residency_hits",
                    "decode_vram_residency_required_chunks", "decode_vram_residency_hits",
                    "tier_attempts", "tier_vram_hits", "tier_dram_hits", "tier_ssd_misses"):
            resp[f"_dpa_moe_delta_{key}"] = max(0.0, stats1.get(key, 0.0) - stats0.get(key, 0.0))
        resp["_dpa_moe_host_dram_mb"] = stats1.get("host_dram_mb", 0.0)
        resp["_dpa_moe_tier_diag_enabled"] = stats1.get("tier_diag_enabled", 0.0)
    return resp


def _chat_completion(port: int, messages: list[dict[str, str]], n_predict: int) -> dict[str, Any]:
    stats0 = _dpa_moe_stats(port)
    t0 = time.monotonic()
    resp = _post_json(port, "/chat/completions", {
        "messages": messages,
        "max_tokens": n_predict,
        "ignore_eos": _ignore_eos(),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "cache_prompt": False,
        "stream": False,
    })
    resp["_wall_s"] = time.monotonic() - t0
    choices = resp.get("choices") or []
    if choices:
        msg = choices[0].get("message", {})
        if isinstance(msg, dict):
            resp["content"] = msg.get("content", "")
    stats1 = _dpa_moe_stats(port)
    if stats0 or stats1:
        for key in ("h2d_mb", "h2d_calls", "make_room_calls", "mc_pread_ns", "mc_calls",
                    "vram_residency_required_chunks", "vram_residency_hits",
                    "vram_residency_load_misses",
                    "prefill_vram_residency_required_chunks", "prefill_vram_residency_hits",
                    "decode_vram_residency_required_chunks", "decode_vram_residency_hits",
                    "tier_attempts", "tier_vram_hits", "tier_dram_hits", "tier_ssd_misses"):
            resp[f"_dpa_moe_delta_{key}"] = max(0.0, stats1.get(key, 0.0) - stats0.get(key, 0.0))
        resp["_dpa_moe_host_dram_mb"] = stats1.get("host_dram_mb", 0.0)
        resp["_dpa_moe_tier_diag_enabled"] = stats1.get("tier_diag_enabled", 0.0)
    return resp


def _record(run_kind: str, mode: SpeedMode, dataset: DatasetSpec,
            length: LengthSpec, ubatch: int, i: int, resp: dict[str, Any],
            effective_gb: str) -> dict[str, Any]:
    t = resp.get("timings", {})
    prompt_ms = float(t.get("prompt_ms", 0.0) or 0.0)
    decode_ms = float(t.get("predicted_ms", 0.0) or 0.0)
    wall_s = float(resp.get("_wall_s", 0.0) or 0.0)
    pred_n = int(t.get("predicted_n", 0) or 0)
    generated_answer = str(resp.get("content", ""))
    tier_attempts = float(resp.get("_dpa_moe_delta_tier_attempts", 0.0) or 0.0)
    tier_vram_hits = float(resp.get("_dpa_moe_delta_tier_vram_hits", 0.0) or 0.0)
    tier_dram_hits = float(resp.get("_dpa_moe_delta_tier_dram_hits", 0.0) or 0.0)
    tier_ssd_misses = float(resp.get("_dpa_moe_delta_tier_ssd_misses", 0.0) or 0.0)
    h2d_mb = float(resp.get("_dpa_moe_delta_h2d_mb", 0.0) or 0.0)
    h2d_calls = float(resp.get("_dpa_moe_delta_h2d_calls", 0.0) or 0.0)
    vram_required = float(resp.get("_dpa_moe_delta_vram_residency_required_chunks", 0.0) or 0.0)
    vram_hits = float(resp.get("_dpa_moe_delta_vram_residency_hits", 0.0) or 0.0)
    vram_load_misses = float(resp.get("_dpa_moe_delta_vram_residency_load_misses", 0.0) or 0.0)
    prefill_vram_required = float(resp.get("_dpa_moe_delta_prefill_vram_residency_required_chunks", 0.0) or 0.0)
    prefill_vram_hits = float(resp.get("_dpa_moe_delta_prefill_vram_residency_hits", 0.0) or 0.0)
    decode_vram_required = float(resp.get("_dpa_moe_delta_decode_vram_residency_required_chunks", 0.0) or 0.0)
    decode_vram_hits = float(resp.get("_dpa_moe_delta_decode_vram_residency_hits", 0.0) or 0.0)
    logical_non_vram = tier_dram_hits + tier_ssd_misses
    return {
        "run_kind": run_kind,
        "model": MODEL_FAMILY,
        "config": mode.name,
        "series": mode.series,
        "effective_gb": effective_gb,
        "allocator": mode.allocator,
        "kbar": mode.kbar,
        "vram_cap_mb": mode.vram_cap_mb,
        "dataset": dataset.name,
        "prompt_target_tokens": length.prompt_tokens,
        "generate_target_tokens": length.generate_tokens,
        "length": length.name,
        "ubatch_size": ubatch,
        "request_parallel": 1,
        "prompt_n": int(t.get("prompt_n", 0) or 0),
        "generated_n": pred_n,
        "generated_answer": generated_answer,
        "generated_answer_chars": len(generated_answer),
        "prefill_latency_ms": prompt_ms,
        "decode_latency_ms": decode_ms,
        "end_to_end_latency_ms": wall_s * 1000.0,
        "ttft_ms": prompt_ms,
        "prompt_tokens_per_sec": float(t.get("prompt_per_second", 0.0) or 0.0),
        "generated_tokens_per_sec": float(t.get("predicted_per_second", 0.0) or 0.0),
        "request_per_sec": (1.0 / wall_s) if wall_s > 0.0 else 0.0,
        "predicted_per_token_ms": float(t.get("predicted_per_token_ms", 0.0) or 0.0),
        "h2d_mb": h2d_mb,
        "h2d_calls": h2d_calls,
        "physical_h2d_mb": h2d_mb,
        "physical_h2d_calls": h2d_calls,
        "physical_mb_per_h2d_call": h2d_mb / h2d_calls if h2d_calls > 0.0 else 0.0,
        "make_room_calls": float(resp.get("_dpa_moe_delta_make_room_calls", 0.0) or 0.0),
        "mc_pread_ms": float(resp.get("_dpa_moe_delta_mc_pread_ns", 0.0) or 0.0) / 1.0e6,
        "mc_calls": float(resp.get("_dpa_moe_delta_mc_calls", 0.0) or 0.0),
        "host_dram_mb": float(resp.get("_dpa_moe_host_dram_mb", 0.0) or 0.0),
        "tier_diag_enabled": bool(resp.get("_dpa_moe_tier_diag_enabled", 0.0)),
        "vram_residency_required_chunks": vram_required,
        "vram_residency_hits": vram_hits,
        "vram_residency_load_misses": vram_load_misses,
        "vram_residency_hit_pct": _hit_pct(vram_hits, vram_required),
        "prefill_vram_residency_required_chunks": prefill_vram_required,
        "prefill_vram_residency_hits": prefill_vram_hits,
        "prefill_vram_residency_hit_pct": _hit_pct(prefill_vram_hits, prefill_vram_required),
        "decode_vram_residency_required_chunks": decode_vram_required,
        "decode_vram_residency_hits": decode_vram_hits,
        "decode_vram_residency_hit_pct": _hit_pct(decode_vram_hits, decode_vram_required),
        "tier_attempts": tier_attempts,
        "tier_vram_hits": tier_vram_hits,
        "tier_dram_hits": tier_dram_hits,
        "tier_ssd_misses": tier_ssd_misses,
        "tier_vram_hit_pct": _hit_pct(tier_vram_hits, tier_attempts),
        "tier_dram_hit_pct": _hit_pct(tier_dram_hits, tier_attempts),
        "tier_ssd_miss_pct": _hit_pct(tier_ssd_misses, tier_attempts),
        "tier_total_hit_pct": _hit_pct(tier_vram_hits + tier_dram_hits, tier_attempts),
        "logical_tier_attempts": tier_attempts,
        "logical_tier_vram_hits": tier_vram_hits,
        "logical_tier_dram_hits": tier_dram_hits,
        "logical_tier_ssd_misses": tier_ssd_misses,
        "logical_tier_vram_hit_pct": _hit_pct(tier_vram_hits, tier_attempts),
        "logical_tier_dram_hit_pct": _hit_pct(tier_dram_hits, tier_attempts),
        "logical_tier_ssd_miss_pct": _hit_pct(tier_ssd_misses, tier_attempts),
        "logical_tier_total_hit_pct": _hit_pct(tier_vram_hits + tier_dram_hits, tier_attempts),
        "logical_non_vram_attempts": logical_non_vram,
        "physical_load_dedupe_factor": logical_non_vram / h2d_calls if h2d_calls > 0.0 else 0.0,
        "sample_index": i,
        "sample_seed": _sample_seed(dataset.name),
        "cache_prompt": False,
        "ignore_eos": _ignore_eos(),
    }


def _maybe_record_example(seen: set[tuple[str, str, str]], run_kind: str,
                          mode: SpeedMode, dataset: DatasetSpec,
                          length: LengthSpec, prompt: Any,
                          resp: dict[str, Any]) -> None:
    if run_kind != "measured":
        return
    key = (mode.name, dataset.name, length.name)
    if key in seen:
        return
    content = str(resp.get("content", ""))
    if not content:
        return
    seen.add(key)
    EXAMPLES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": MODEL_FAMILY,
        "config": mode.name,
        "series": mode.series,
        "allocator": mode.allocator,
        "kbar": mode.kbar,
        "vram_cap_mb": mode.vram_cap_mb,
        "dataset": dataset.name,
        "length": length.name,
        "prompt_target_tokens": length.prompt_tokens,
        "generate_target_tokens": length.generate_tokens,
        "sample_seed": _sample_seed(dataset.name),
        "prompt_preview": json.dumps(prompt, ensure_ascii=False)[:1000] if isinstance(prompt, list) else str(prompt)[:1000],
        "completion": content,
    }
    with EXAMPLES_JSONL.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _stats(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return mean(xs), stdev(xs) if len(xs) > 1 else 0.0


def _effective_weights() -> dict[str, str]:
    weights: dict[str, str] = {}
    path = REPO / "experiments/results" / MODEL_FAMILY / "kld/bytes_vs_kld.csv"
    if not path.exists():
        path = REPO / "experiments/results" / MODEL_FAMILY / "kld_sweep/bytes_vs_kld.csv"
    if not path.exists():
        return weights
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            mode = row.get("mode", "")
            label = row.get("label", "")
            k = row.get("K_bar", "")
            gb = row.get("effective_gb", "")
            if not gb:
                continue
            if mode == "baseline":
                weights[label] = gb
            elif mode in {"uniform", "dynamic"} and k:
                weights[f"{mode}_k{k.replace('.', '_')}"] = gb
                try:
                    kf = float(k)
                    weights[f"{mode}_k{_format_k(kf).replace('.', '_')}"] = gb
                except ValueError:
                    pass
    return weights


def _write_summary(records: list[dict[str, Any]]) -> None:
    measured = [r for r in records if r["run_kind"] == "measured"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("model", "config", "series", "effective_gb", "allocator", "kbar",
            "vram_cap_mb", "dataset", "length", "prompt_target_tokens",
            "generate_target_tokens", "ubatch_size")
    for r in measured:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    fields = list(keys) + [
        "n_samples", "mean_prefill_latency_ms", "std_prefill_latency_ms",
        "mean_decode_latency_ms", "std_decode_latency_ms",
        "mean_end_to_end_latency_ms", "std_end_to_end_latency_ms",
        "mean_ttft_ms", "std_ttft_ms",
        "mean_generated_tokens_per_sec", "std_generated_tokens_per_sec",
        "mean_prompt_tokens_per_sec", "std_prompt_tokens_per_sec",
        "mean_request_per_sec", "std_request_per_sec",
        "mean_h2d_mb", "std_h2d_mb", "mean_h2d_calls", "std_h2d_calls",
        "mean_physical_h2d_mb", "std_physical_h2d_mb",
        "mean_physical_h2d_calls", "std_physical_h2d_calls",
        "mean_physical_mb_per_h2d_call", "std_physical_mb_per_h2d_call",
        "mean_mc_pread_ms", "std_mc_pread_ms", "mean_make_room_calls",
        "std_make_room_calls",
        "mean_host_dram_mb", "std_host_dram_mb",
        "mean_vram_residency_required_chunks", "std_vram_residency_required_chunks",
        "mean_vram_residency_hit_pct", "std_vram_residency_hit_pct",
        "mean_prefill_vram_residency_hit_pct", "std_prefill_vram_residency_hit_pct",
        "mean_decode_vram_residency_hit_pct", "std_decode_vram_residency_hit_pct",
        "mean_tier_attempts", "std_tier_attempts",
        "mean_tier_vram_hit_pct", "std_tier_vram_hit_pct",
        "mean_tier_dram_hit_pct", "std_tier_dram_hit_pct",
        "mean_tier_ssd_miss_pct", "std_tier_ssd_miss_pct",
        "mean_tier_total_hit_pct", "std_tier_total_hit_pct",
        "mean_logical_tier_attempts", "std_logical_tier_attempts",
        "mean_logical_tier_vram_hit_pct", "std_logical_tier_vram_hit_pct",
        "mean_logical_tier_dram_hit_pct", "std_logical_tier_dram_hit_pct",
        "mean_logical_tier_ssd_miss_pct", "std_logical_tier_ssd_miss_pct",
        "mean_logical_tier_total_hit_pct", "std_logical_tier_total_hit_pct",
        "mean_logical_non_vram_attempts", "std_logical_non_vram_attempts",
        "mean_physical_load_dedupe_factor", "std_physical_load_dedupe_factor",
        "mean_prompt_n", "mean_generated_n",
        "cache_prompt", "request_parallel",
    ]
    rows = []
    for key, rs in sorted(groups.items()):
        row = dict(zip(keys, key))
        for out, src in [
            ("prefill_latency_ms", "prefill_latency_ms"),
            ("decode_latency_ms", "decode_latency_ms"),
            ("end_to_end_latency_ms", "end_to_end_latency_ms"),
            ("ttft_ms", "ttft_ms"),
            ("generated_tokens_per_sec", "generated_tokens_per_sec"),
            ("prompt_tokens_per_sec", "prompt_tokens_per_sec"),
            ("request_per_sec", "request_per_sec"),
            ("h2d_mb", "h2d_mb"),
            ("h2d_calls", "h2d_calls"),
            ("physical_h2d_mb", "physical_h2d_mb"),
            ("physical_h2d_calls", "physical_h2d_calls"),
            ("physical_mb_per_h2d_call", "physical_mb_per_h2d_call"),
            ("mc_pread_ms", "mc_pread_ms"),
            ("make_room_calls", "make_room_calls"),
            ("host_dram_mb", "host_dram_mb"),
            ("vram_residency_required_chunks", "vram_residency_required_chunks"),
            ("vram_residency_hit_pct", "vram_residency_hit_pct"),
            ("prefill_vram_residency_hit_pct", "prefill_vram_residency_hit_pct"),
            ("decode_vram_residency_hit_pct", "decode_vram_residency_hit_pct"),
            ("tier_attempts", "tier_attempts"),
            ("tier_vram_hit_pct", "tier_vram_hit_pct"),
            ("tier_dram_hit_pct", "tier_dram_hit_pct"),
            ("tier_ssd_miss_pct", "tier_ssd_miss_pct"),
            ("tier_total_hit_pct", "tier_total_hit_pct"),
            ("logical_tier_attempts", "logical_tier_attempts"),
            ("logical_tier_vram_hit_pct", "logical_tier_vram_hit_pct"),
            ("logical_tier_dram_hit_pct", "logical_tier_dram_hit_pct"),
            ("logical_tier_ssd_miss_pct", "logical_tier_ssd_miss_pct"),
            ("logical_tier_total_hit_pct", "logical_tier_total_hit_pct"),
            ("logical_non_vram_attempts", "logical_non_vram_attempts"),
            ("physical_load_dedupe_factor", "physical_load_dedupe_factor"),
        ]:
            m, s = _stats([float(r.get(src, 0.0)) for r in rs if float(r.get(src, 0.0)) > 0.0])
            row[f"mean_{out}"] = f"{m:.6f}"
            row[f"std_{out}"] = f"{s:.6f}"
        row["n_samples"] = len(rs)
        row["mean_prompt_n"] = f"{mean(float(r['prompt_n']) for r in rs):.3f}"
        row["mean_generated_n"] = f"{mean(float(r['generated_n']) for r in rs):.3f}"
        row["cache_prompt"] = "false"
        row["request_parallel"] = "1"
        rows.append(row)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    _write_plot_points_csv()


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _plot_x(row: dict[str, str]) -> tuple[str, float | None]:
    effective = _float_or_none(row.get("effective_gb"))
    if effective is not None:
        return "effective_gb", effective
    kbar = _float_or_none(row.get("kbar"))
    if kbar is not None:
        return "kbar", kbar
    import re
    m = re.search(r"k(\d+(?:_\d+)?)", row.get("config", ""))
    if m:
        return "kbar", float(m.group(1).replace("_", "."))
    return "index", None


def _mean_field(rows: list[dict[str, str]], field: str) -> str:
    vals = [_float_or_none(r.get(field)) for r in rows]
    clean = [v for v in vals if v is not None]
    return f"{mean(clean):.6f}" if clean else ""


def _std_field(rows: list[dict[str, str]], field: str) -> str:
    vals = [_float_or_none(r.get(field)) for r in rows]
    clean = [v for v in vals if v is not None]
    return f"{stdev(clean):.6f}" if len(clean) > 1 else "0.000000"


def _write_plot_points_csv() -> None:
    if not SUMMARY_CSV.exists():
        return
    effective_weights = _effective_weights()
    rows = list(csv.DictReader(SUMMARY_CSV.open(newline="")))
    if not rows:
        return
    group_keys = ("model", "config", "series", "allocator", "kbar", "effective_gb",
                  "vram_cap_mb", "length", "prompt_target_tokens",
                  "generate_target_tokens", "ubatch_size")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in group_keys), []).append(row)

    fields = [
        "model", "config", "series", "allocator", "kbar", "effective_gb",
        "x_axis", "x_value", "vram_cap_mb", "length", "prompt_target_tokens",
        "generate_target_tokens", "ubatch_size", "n_datasets", "total_samples",
        "datasets", "mean_generated_tokens_per_sec",
        "std_generated_tokens_per_sec_across_datasets",
        "mean_prompt_tokens_per_sec", "std_prompt_tokens_per_sec_across_datasets",
        "mean_decode_latency_ms", "mean_prefill_latency_ms",
        "mean_end_to_end_latency_ms", "mean_generated_n", "mean_prompt_n",
        "mean_h2d_mb", "std_h2d_mb_across_datasets",
        "mean_h2d_calls", "std_h2d_calls_across_datasets",
        "mean_physical_h2d_mb", "std_physical_h2d_mb_across_datasets",
        "mean_physical_h2d_calls", "std_physical_h2d_calls_across_datasets",
        "mean_physical_mb_per_h2d_call",
        "mean_mc_pread_ms", "mean_make_room_calls",
        "mean_host_dram_mb",
        "mean_vram_residency_required_chunks", "mean_vram_residency_hit_pct",
        "mean_prefill_vram_residency_hit_pct", "mean_decode_vram_residency_hit_pct",
        "mean_tier_attempts",
        "mean_tier_vram_hit_pct", "mean_tier_dram_hit_pct",
        "mean_tier_ssd_miss_pct", "mean_tier_total_hit_pct",
        "mean_logical_tier_attempts", "mean_logical_non_vram_attempts",
        "mean_logical_tier_vram_hit_pct", "mean_logical_tier_dram_hit_pct",
        "mean_logical_tier_ssd_miss_pct", "mean_logical_tier_total_hit_pct",
        "mean_physical_load_dedupe_factor",
    ]
    out_rows: list[dict[str, str]] = []
    for key, rs in sorted(groups.items()):
        base = dict(zip(group_keys, key))
        if not base.get("effective_gb"):
            lookup_names = [base.get("config", "")]
            lookup_names.append(lookup_names[0].split("_cap", 1)[0])
            for name in lookup_names:
                if name in effective_weights:
                    base["effective_gb"] = effective_weights[name]
                    break
        x_axis, x_value = _plot_x(base)
        if x_value is None:
            x_value = float(len(out_rows))
        total_samples = sum(int(float(r.get("n_samples", "0") or 0)) for r in rs)
        out_rows.append({
            **base,
            "x_axis": x_axis,
            "x_value": f"{x_value:.6f}",
            "n_datasets": str(len(rs)),
            "total_samples": str(total_samples),
            "datasets": "|".join(sorted(r.get("dataset", "") for r in rs)),
            "mean_generated_tokens_per_sec": _mean_field(rs, "mean_generated_tokens_per_sec"),
            "std_generated_tokens_per_sec_across_datasets": _std_field(rs, "mean_generated_tokens_per_sec"),
            "mean_prompt_tokens_per_sec": _mean_field(rs, "mean_prompt_tokens_per_sec"),
            "std_prompt_tokens_per_sec_across_datasets": _std_field(rs, "mean_prompt_tokens_per_sec"),
            "mean_decode_latency_ms": _mean_field(rs, "mean_decode_latency_ms"),
            "mean_prefill_latency_ms": _mean_field(rs, "mean_prefill_latency_ms"),
            "mean_end_to_end_latency_ms": _mean_field(rs, "mean_end_to_end_latency_ms"),
            "mean_generated_n": _mean_field(rs, "mean_generated_n"),
            "mean_prompt_n": _mean_field(rs, "mean_prompt_n"),
            "mean_h2d_mb": _mean_field(rs, "mean_h2d_mb"),
            "std_h2d_mb_across_datasets": _std_field(rs, "mean_h2d_mb"),
            "mean_h2d_calls": _mean_field(rs, "mean_h2d_calls"),
            "std_h2d_calls_across_datasets": _std_field(rs, "mean_h2d_calls"),
            "mean_physical_h2d_mb": _mean_field(rs, "mean_physical_h2d_mb"),
            "std_physical_h2d_mb_across_datasets": _std_field(rs, "mean_physical_h2d_mb"),
            "mean_physical_h2d_calls": _mean_field(rs, "mean_physical_h2d_calls"),
            "std_physical_h2d_calls_across_datasets": _std_field(rs, "mean_physical_h2d_calls"),
            "mean_physical_mb_per_h2d_call": _mean_field(rs, "mean_physical_mb_per_h2d_call"),
            "mean_mc_pread_ms": _mean_field(rs, "mean_mc_pread_ms"),
            "mean_make_room_calls": _mean_field(rs, "mean_make_room_calls"),
            "mean_host_dram_mb": _mean_field(rs, "mean_host_dram_mb"),
            "mean_vram_residency_required_chunks": _mean_field(rs, "mean_vram_residency_required_chunks"),
            "mean_vram_residency_hit_pct": _mean_field(rs, "mean_vram_residency_hit_pct"),
            "mean_prefill_vram_residency_hit_pct": _mean_field(rs, "mean_prefill_vram_residency_hit_pct"),
            "mean_decode_vram_residency_hit_pct": _mean_field(rs, "mean_decode_vram_residency_hit_pct"),
            "mean_tier_attempts": _mean_field(rs, "mean_tier_attempts"),
            "mean_tier_vram_hit_pct": _mean_field(rs, "mean_tier_vram_hit_pct"),
            "mean_tier_dram_hit_pct": _mean_field(rs, "mean_tier_dram_hit_pct"),
            "mean_tier_ssd_miss_pct": _mean_field(rs, "mean_tier_ssd_miss_pct"),
            "mean_tier_total_hit_pct": _mean_field(rs, "mean_tier_total_hit_pct"),
            "mean_logical_tier_attempts": _mean_field(rs, "mean_logical_tier_attempts"),
            "mean_logical_non_vram_attempts": _mean_field(rs, "mean_logical_non_vram_attempts"),
            "mean_logical_tier_vram_hit_pct": _mean_field(rs, "mean_logical_tier_vram_hit_pct"),
            "mean_logical_tier_dram_hit_pct": _mean_field(rs, "mean_logical_tier_dram_hit_pct"),
            "mean_logical_tier_ssd_miss_pct": _mean_field(rs, "mean_logical_tier_ssd_miss_pct"),
            "mean_logical_tier_total_hit_pct": _mean_field(rs, "mean_logical_tier_total_hit_pct"),
            "mean_physical_load_dedupe_factor": _mean_field(rs, "mean_physical_load_dedupe_factor"),
        })
    with PLOT_POINTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)


def _write_plots_html() -> None:
    if not SUMMARY_CSV.exists():
        return
    _write_plot_points_csv()
    rows = list(csv.DictReader(PLOT_POINTS_CSV.open(newline=""))) if PLOT_POINTS_CSV.exists() else []
    rows_json = json.dumps(rows)
    PLOTS_HTML.write_text("""<!doctype html>
<html><head><meta charset="utf-8"><title>DPA-MoE Speed Sweep</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;color:#20242a}svg{max-width:860px;width:100%;height:auto;border:1px solid #d0d5dd;margin:10px 0 28px}.grid{stroke:#d0d5dd;opacity:.5}.axis{stroke:#667085}.u{stroke:#475467}.d{stroke:#2563eb}.b{stroke:#d92d20}.s{stroke:#079455}.note{color:#667085}</style></head>
<body><h1>DPA-MoE Speed Sweep</h1><p class="note">Generated from <code>speed_plot_points.csv</code>. Uses effective weight on the x-axis when present; TPS-only rows fall back to K-bar.</p>
<h2>Generated TPS</h2><svg id="plot-tg" viewBox="0 0 860 500"></svg>
<h2>Prefill TPS</h2><svg id="plot-pp" viewBox="0 0 860 500"></svg>
<h2>H2D MB / Request</h2><svg id="plot-h2d-mb" viewBox="0 0 860 500"></svg>
<h2>H2D Calls / Request</h2><svg id="plot-h2d-calls" viewBox="0 0 860 500"></svg>
<script>
const rows = __ROWS_JSON__;
function xValue(r){ return +r.x_value; }
function plot(svgId, metric, ylabel, stdMetric){
const svg=document.getElementById(svgId), W=860,H=500,m={l:75,r:28,t:28,b:58};
const firstLength = rows.length ? rows[0].length : '';
const data=rows.filter(r=>r.length===firstLength && Number.isFinite(xValue(r)) && r[metric]).map(r=>({x:xValue(r),y:+r[metric],e:+(r[stdMetric]||0),s:r.series,c:r.config,d:r.dataset}));
if(!data.length){ svg.replaceWith(document.createTextNode('No plotted rows yet.')); return; }
const xmin=Math.min(...data.map(d=>d.x)), xmax=Math.max(...data.map(d=>d.x)), ymin=0, ymax=Math.max(...data.map(d=>d.y+d.e))*1.1;
const sx=x=>m.l+(x-xmin)/(xmax-xmin||1)*(W-m.l-m.r), sy=y=>H-m.b-(y-ymin)/(ymax-ymin||1)*(H-m.t-m.b);
function el(n,a,t=''){const e=document.createElementNS('http://www.w3.org/2000/svg',n); for(const[k,v]of Object.entries(a))e.setAttribute(k,v); if(t)e.textContent=t; svg.appendChild(e); return e;}
for(let i=0;i<=5;i++){const x=xmin+(xmax-xmin)*i/5; el('line',{x1:sx(x),y1:m.t,x2:sx(x),y2:H-m.b,class:'grid'}); el('text',{x:sx(x),y:H-25,'text-anchor':'middle'},x.toFixed(1));}
for(let i=0;i<=5;i++){const y=ymax*i/5; el('line',{x1:m.l,y1:sy(y),x2:W-m.r,y2:sy(y),class:'grid'}); el('text',{x:m.l-8,y:sy(y)+4,'text-anchor':'end'},y.toFixed(1));}
el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}); el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'});
el('text',{x:W/2,y:H-8,'text-anchor':'middle'},rows.some(r=>r.x_axis==='effective_gb')?'Effective weight (GB)':'K-bar'); el('text',{x:18,y:H/2,transform:`rotate(-90 18 ${H/2})`,'text-anchor':'middle'},ylabel);
for(const s of ['baseline','uniform','dynamic','sched']){
const pts=data.filter(d=>d.s===s).sort((a,b)=>a.x-b.x);
if(pts.length>1)el('path',{d:pts.map((p,i)=>(i?'L':'M')+sx(p.x)+' '+sy(p.y)).join(' '),fill:'none','stroke-width':'2.2','stroke-linejoin':'round','stroke-linecap':'round',class:s[0]});
for(const p of pts){if(p.e>0){const y0=sy(Math.max(0,p.y-p.e)), y1=sy(p.y+p.e), x=sx(p.x); el('line',{x1:x,y1:y0,x2:x,y2:y1,'stroke-width':'1.2',class:s[0]}); el('line',{x1:x-4,y1:y0,x2:x+4,y2:y0,'stroke-width':'1.2',class:s[0]}); el('line',{x1:x-4,y1:y1,x2:x+4,y2:y1,'stroke-width':'1.2',class:s[0]});}}
if(pts.length===1){const p=pts[0]; const q=el('circle',{cx:sx(p.x),cy:sy(p.y),r:3.5,fill:'none','stroke-width':'2',class:s[0]}); q.appendChild(el('title',{},`${p.c}: ${p.y.toFixed(2)} tok/s @ ${p.x.toFixed(2)}`));}
}
}
plot('plot-tg', 'mean_generated_tokens_per_sec', 'Generated tokens/sec', 'std_generated_tokens_per_sec_across_datasets');
plot('plot-pp', 'mean_prompt_tokens_per_sec', 'Prefill tokens/sec', 'std_prompt_tokens_per_sec_across_datasets');
plot('plot-h2d-mb', 'mean_h2d_mb', 'H2D MB / request', 'std_h2d_mb_across_datasets');
plot('plot-h2d-calls', 'mean_h2d_calls', 'H2D calls / request', 'std_h2d_calls_across_datasets');
</script></body></html>
""".replace("__ROWS_JSON__", rows_json))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        _write_plots_html()
        print(f"wrote {PLOTS_HTML}")
        return 0

    gpu = os.environ.get("GPU", cfg.BENCHMARK_GPUS[0] if cfg.BENCHMARK_GPUS else "0")
    modes = _configured_modes()
    mode_filter = {x.strip() for x in os.environ.get("MODES", "").split(",") if x.strip()}
    if mode_filter:
        existing = {m.name for m in modes}
        for name in sorted(mode_filter - existing):
            extra = _mode_from_config_name(name)
            if extra is not None:
                modes.append(extra)
        modes = [m for m in modes if m.name in mode_filter]
    ubatches = [int(x) for x in os.environ.get(
        "UBATCHES", ",".join(str(x) for x in SPEED_CFG.get("ubatch_sizes", [1, 16, 64, 256]))).split(",") if x.strip()]
    warmup = int(os.environ.get("WARMUP", SPEED_CFG.get("warmup_runs", 10)))
    runs_per_sample = int(os.environ.get(
        "RUNS_PER_SAMPLE",
        os.environ.get("MEASURED", SPEED_CFG.get("runs_per_sample", SPEED_CFG.get("measured_runs", 50))),
    ))
    diag = os.environ.get("SPEED_DIAG", "0") == "1"
    datasets = _datasets()
    lengths = _lengths()

    missing = [str(d.path) for d in datasets if not d.path.exists()]
    print(f"[speed-sweep] model={MODEL_FAMILY} root_config={ROOT_CONFIG} bench_config={BENCH_CONFIG} gpu={gpu}")
    print(f"[speed-sweep] modes={len(modes)} ubatches={ubatches} warmup={warmup} runs_per_sample={runs_per_sample} diag={int(diag)}")
    print(f"[speed-sweep] request_parallel=1 cache_prompt=false --no-cache-prompt ignore_eos={int(_ignore_eos())} sample_seed={SPEED_CFG.get('sample_seed', SPEED_CFG.get('seed', BENCH_CFG.get('seed', 0)))}")
    for d in datasets:
        print(f"  dataset={d.name} path={d.path} n_samples={d.n_samples} sample_seed={_sample_seed(d.name)} exists={int(d.path.exists())}")
    for l in lengths:
        print(f"  length={l.name} prompt={l.prompt_tokens} generate={l.generate_tokens}")
    if missing:
        print("[speed-sweep] missing datasets:")
        for p in missing:
            print(f"  {p}")
    if args.dry_run:
        print("[speed-sweep] DRY_RUN=1; not starting servers")
        return 0 if not missing else 2
    if missing:
        return 2

    all_texts = {d.name: _read_texts(d, extra_samples=warmup) for d in datasets}
    effective = _effective_weights()
    max_ctx = int(SPEED_CFG.get("ctx_size", max(l.prompt_tokens + l.generate_tokens + 128 for l in lengths)))
    for mode in modes:
        if not mode.model_path.exists():
            raise FileNotFoundError(f"missing model for {mode.name}: {mode.model_path}")
        for ubatch in ubatches:
            port = _free_port()
            log_path = SPEED_RUN_ROOT / mode.name / _run_label(mode, ubatch) / "server.log"
            proc = _start_server(port, gpu, mode, ubatch, max_ctx, diag, log_path)
            print(f"[speed-sweep] mode={mode.name} ub={ubatch} pid={proc.pid} log={log_path}", flush=True)
            try:
                _wait_ready(port, proc)
                for length in lengths:
                    _set_output_paths(mode, length, ubatch)
                    records: list[dict[str, Any]] = []
                    example_keys: set[tuple[str, str, str]] = set()
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    REQUESTS_JSONL.unlink(missing_ok=True)
                    EXAMPLES_JSONL.unlink(missing_ok=True)
                    print(f"[speed-sweep] output={RESULTS_DIR}", flush=True)
                    for dataset in datasets:
                        texts = all_texts[dataset.name]
                        warmup_texts = texts[:warmup]
                        measured_texts = texts[warmup:warmup + dataset.n_samples]
                        if len(measured_texts) < dataset.n_samples:
                            raise ValueError(
                                f"dataset {dataset.name} has {len(measured_texts)} measured samples "
                                f"after reserving {warmup} warmup samples; need {dataset.n_samples}")
                        if dataset.chat_template:
                            if length.prompt_tokens > 0:
                                warmup_prompts = [_materialize_chat_messages(port, t, length.prompt_tokens) for t in warmup_texts]
                                prompts = [_materialize_chat_messages(port, t, length.prompt_tokens) for t in measured_texts]
                            else:
                                warmup_prompts = [_parse_sharegpt_messages(t) for t in warmup_texts]
                                prompts = [_parse_sharegpt_messages(t) for t in measured_texts]
                        else:
                            if length.prompt_tokens > 0:
                                warmup_prompts = [_materialize_prompt(port, t, length.prompt_tokens) for t in warmup_texts]
                                prompts = [_materialize_prompt(port, t, length.prompt_tokens) for t in measured_texts]
                            else:
                                warmup_prompts = warmup_texts
                                prompts = measured_texts
                        selected = [("warmup", i, prompt) for i, prompt in enumerate(warmup_prompts)]
                        selected.extend(
                            ("measured", sample_idx, prompt)
                            for sample_idx, prompt in enumerate(prompts)
                            for _ in range(runs_per_sample)
                        )
                        eff = effective.get(mode.name, "")
                        with REQUESTS_JSONL.open("a") as out:
                            for i, (kind, sample_idx, prompt) in enumerate(selected):
                                if dataset.chat_template:
                                    resp = _chat_completion(port, prompt, length.generate_tokens)
                                else:
                                    resp = _completion(port, prompt, length.generate_tokens)
                                _maybe_record_example(example_keys, kind, mode, dataset, length, prompt, resp)
                                rec = _record(kind, mode, dataset, length, ubatch, sample_idx, resp, eff)
                                rec["repeat_index"] = (
                                    0 if kind == "warmup"
                                    else (i - warmup) % runs_per_sample
                                )
                                rec["runs_per_sample"] = runs_per_sample
                                records.append(rec)
                                out.write(json.dumps(rec, sort_keys=True) + "\n")
                                out.flush()
                                print(f"  [{mode.name} ub={ubatch} {length.name} {dataset.name}]"
                                      f" {kind} {i + 1}/{len(selected)}"
                                      f" tg={rec['generated_tokens_per_sec']:.2f}"
                                      f" h2d_mb={rec['h2d_mb']:.2f} h2d_calls={rec['h2d_calls']:.0f}"
                                      f" vram_resident={rec['vram_residency_hit_pct']:.1f}%"
                                      f" decode_resident={rec['decode_vram_residency_hit_pct']:.1f}%"
                                      f" logical_hit={rec['tier_total_hit_pct']:.1f}%"
                                      f" logical_dram={rec['tier_dram_hit_pct']:.1f}%"
                                      f" logical_ssd={rec['tier_ssd_miss_pct']:.1f}%"
                                      f" prompt_n={rec['prompt_n']} gen_n={rec['generated_n']}",
                                      flush=True)
                        _write_summary(records)
                        _write_plots_html()
            finally:
                _stop_server(proc)
                print(f"[speed-sweep] stopped mode={mode.name} ub={ubatch}", flush=True)
    print(f"[speed-sweep] results root -> {SPEED_RESULTS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
