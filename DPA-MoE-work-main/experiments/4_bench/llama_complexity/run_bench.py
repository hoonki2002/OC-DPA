#!/usr/bin/env python3
"""Native llama.cpp multiple-choice benchmark for ARC-C and PIQA.

lm_eval's local-completions loglikelihood path needs prompt echo logprobs.
This llama-server build rejects echo, so ARC/PIQA collapse to all-zero ties.
Use llama-perplexity --multiple-choice instead; it computes continuation
logprobs directly from llama.cpp logits.

Results:
    experiments/results/<MODEL>/bench/<config>/llama_mc/
        arc_c.log, piqa.log
        arc_c.json, piqa.json
        summary.csv
"""
from __future__ import annotations

import csv
import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))
from _config import (
    CONFIGS, DATA, LLAMA_MC_BATCH, LLAMA_MC_CTX, LLAMA_MC_TASKS,
    LLAMA_MC_SERIAL_NP1, LLAMA_MC_UBATCH, MODEL, RESULTS_DIR, RESULT_SUFFIX,
    is_single_flight_config, validate_n_parallel,
)


REPO = Path(__file__).resolve().parents[3]
BIN_PERPLEXITY = REPO / "third_party/llama.cpp/build-dpa-moe/bin/llama-perplexity"

GPU = os.environ.get("GPU", "0")
CONFIG = os.environ.get("CONFIG", "Q2_K")
FORCE = os.environ.get("FORCE", "0") == "1"
TASKS = [
    x.strip()
    for x in os.environ.get("LLAMA_MC_TASKS", ",".join(LLAMA_MC_TASKS)).split(",")
    if x.strip()
]
CTX_SIZE = LLAMA_MC_CTX
BATCH_SIZE = LLAMA_MC_BATCH
UBATCH_SIZE = LLAMA_MC_UBATCH
N_PARALLEL_ENV = os.environ.get("LLAMA_MC_PARALLEL")
N_GPU_LAYERS = 999
FIT_MODE = "off"
NO_WARMUP = True

TASK_FILES = {
    "arc_c": DATA / "arc_c.jsonl",
    "arc_e": DATA / "arc_e.jsonl",
    "piqa": DATA / "piqa.jsonl",
    "winogrande": DATA / "winogrande.jsonl",
    "hellaswag": DATA / "hellaswag.jsonl",
}
DPA_MOE_ENV_KEYS = (
    "DPA_MOE_KBAR", "DPA_MOE_KBAR_ALLOCATOR", "DPA_MOE_KBAR_PROFILE_FILE",
    "DPA_MOE_KBAR_SCHEDULE_TOKENS", "DPA_MOE_KBAR_SCHEDULE_VALUES",
    "DPA_MOE_KBAR_PP_BEGIN", "DPA_MOE_KBAR_PP_END",
    "DPA_MOE_KBAR_TG_BEGIN", "DPA_MOE_KBAR_TG_END",
    "DPA_MOE_KBAR_TG_DECAY_START", "DPA_MOE_KBAR_TG_DECAY_END",
    "DPA_MOE_KBAR_PP_DECAY_END",
    "DPA_MOE_LOG_KBAR",
    "DPA_MOE_PHASE_REASONING_KBAR", "DPA_MOE_PHASE_GENERATION_KBAR",
    "DPA_MOE_PHASE_OPEN_TOKEN_IDS", "DPA_MOE_PHASE_CLOSE_TOKEN_IDS",
)


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def write_string(fh, text: str) -> None:
    data = text.encode("utf-8")
    fh.write(struct.pack("<I", len(data)))
    fh.write(data)


def load_task(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prompt_for(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer:"


def labels_for(row: dict) -> list[int]:
    correct = row["correct_letter"]
    return [1 if letter == correct else 0 for letter in row["choices"].keys()]


def write_mc_binary(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for row in rows:
        answers = list(row["choices"].values())
        labels = labels_for(row)
        if sum(labels) != 1:
            raise ValueError(f"expected exactly one correct label in {row.get('id')}")
        tasks.append((prompt_for(row), answers, labels))

    with out.open("wb") as fh:
        fh.write(struct.pack("<I", len(tasks)))
        positions_at = fh.tell()
        fh.write(b"\x00" * (4 * len(tasks)))

        positions = []
        for question, answers, labels in tasks:
            positions.append(fh.tell())
            write_string(fh, question)

            fh.write(struct.pack("<I", len(answers)))
            for answer in answers:
                write_string(fh, answer)
            fh.write(struct.pack(f"<{len(labels)}i", *labels))

            fh.write(struct.pack("<I", 0))  # mc2 answers, unused by llama.cpp

        end_at = fh.tell()
        fh.seek(positions_at)
        fh.write(struct.pack(f"<{len(positions)}I", *positions))
        fh.seek(end_at)


def parse_final(log_text: str) -> tuple[float, float]:
    m = re.search(r"Final result:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", log_text)
    if not m:
        raise RuntimeError("could not parse Final result from llama-perplexity output")
    return float(m.group(1)), float(m.group(2))


def required_parallel(rows: list[dict]) -> int:
    return max(len(row["choices"]) for row in rows)


def select_task_parallel(config: str, extra_env: dict, config_n_parallel: int,
                         rows: list[dict]) -> int:
    required = required_parallel(rows)
    if is_single_flight_config(config, extra_env):
        if N_PARALLEL_ENV and int(N_PARALLEL_ENV) != 1:
            log(f"[{config}] ignoring LLAMA_MC_PARALLEL={N_PARALLEL_ENV}; "
                "dynamic/sched runs are forced to n_parallel=1")
        return validate_n_parallel(config, extra_env, 1)

    requested = int(N_PARALLEL_ENV) if N_PARALLEL_ENV else config_n_parallel
    if requested < required:
        raise SystemExit(
            f"{config}/{required=} needs LLAMA_MC_PARALLEL >= {required}; got {requested}"
        )
    return validate_n_parallel(config, extra_env, requested)


def run_task(task: str, model: Path, extra_env: dict, out_dir: Path,
             config_n_parallel: int) -> dict:
    rows = load_task(TASK_FILES[task])
    n_parallel = select_task_parallel(CONFIG, extra_env, config_n_parallel, rows)
    single_flight = is_single_flight_config(CONFIG, extra_env)
    bin_path = out_dir / f"{task}.mc.bin"
    log_path = out_dir / f"{task}.log"
    json_path = out_dir / f"{task}.json"

    if json_path.exists() and not FORCE:
        return json.loads(json_path.read_text())

    write_mc_binary(rows, bin_path)
    cmd = [
        str(BIN_PERPLEXITY),
        "-m", str(model),
        "-f", str(bin_path),
        "--multiple-choice",
        "--multiple-choice-tasks", "0",
        "-c", str(CTX_SIZE),
        "-b", str(BATCH_SIZE),
        "-ub", str(UBATCH_SIZE),
        "-np", str(n_parallel),
        "-ngl", str(N_GPU_LAYERS),
        "--no-warmup",
        "--fit", FIT_MODE,
    ]

    log(f"[{CONFIG}/{task}] llama-perplexity starting ({len(rows)} tasks)")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU}
    for key in DPA_MOE_ENV_KEYS:
        env.pop(key, None)
    env.update({k: v for k, v in extra_env.items() if v})
    if LLAMA_MC_SERIAL_NP1 or single_flight:
        env["DPA_MOE_MC_SERIAL_NP1"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"{task} failed rc={proc.returncode}; see {log_path}")

    acc_pct, stderr_pct = parse_final(proc.stdout)
    result = {
        "config": CONFIG,
        "model": MODEL,
        "benchmark": task,
        "profile_file": extra_env.get("DPA_MOE_KBAR_PROFILE_FILE", ""),
        "n_total": len(rows),
        "n_parallel": n_parallel,
        "single_flight": single_flight,
        "runtime": {
            "binary": str(BIN_PERPLEXITY),
            "model_path": str(model),
            "input_bin": str(bin_path),
            "cuda_visible_devices": GPU,
            "ctx_size": CTX_SIZE,
            "batch_size": BATCH_SIZE,
            "ubatch_size": UBATCH_SIZE,
            "n_parallel": n_parallel,
            "n_gpu_layers": N_GPU_LAYERS,
            "no_warmup": NO_WARMUP,
            "fit": FIT_MODE,
            "multiple_choice": True,
            "multiple_choice_tasks": 0,
            "serial_np1": env.get("DPA_MOE_MC_SERIAL_NP1") == "1",
            "llama_mc_parallel_env": N_PARALLEL_ENV or "",
            "command": cmd,
        },
        "dpa_moe_env": {
            key: env[key]
            for key in sorted(DPA_MOE_ENV_KEYS + ("DPA_MOE_MC_SERIAL_NP1",))
            if env.get(key)
        },
        "acc_pct": acc_pct,
        "stderr_pct": stderr_pct,
        "method": "llama-perplexity --multiple-choice",
        "log": str(log_path),
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    log(f"[{CONFIG}/{task}] acc={acc_pct:.4f}% +/- {stderr_pct:.4f}%")
    return result


def write_summary(out_dir: Path, results: list[dict]) -> None:
    out = out_dir / "summary.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "config", "benchmark", "n_total", "n_parallel",
                "ctx_size", "batch_size", "ubatch_size", "n_gpu_layers",
                "single_flight", "serial_np1", "acc_pct", "stderr_pct",
                "profile_file", "method", "log",
            ],
        )
        writer.writeheader()
        for row in results:
            runtime = row.get("runtime", {})
            flat = {
                **row,
                "ctx_size": runtime.get("ctx_size", ""),
                "batch_size": runtime.get("batch_size", ""),
                "ubatch_size": runtime.get("ubatch_size", ""),
                "n_gpu_layers": runtime.get("n_gpu_layers", ""),
                "serial_np1": runtime.get("serial_np1", ""),
            }
            writer.writerow({k: flat.get(k, "") for k in writer.fieldnames})
    log(f"[{CONFIG}] summary -> {out}")


def main() -> None:
    if CONFIG not in CONFIGS:
        raise SystemExit(f"Unknown CONFIG={CONFIG!r}. Valid: {sorted(CONFIGS)}")
    unknown = [task for task in TASKS if task not in TASK_FILES]
    if unknown:
        raise SystemExit(f"Unknown TASKS={unknown}. Valid: {sorted(TASK_FILES)}")
    if not BIN_PERPLEXITY.exists():
        raise SystemExit(f"missing {BIN_PERPLEXITY}")

    model, extra_env, config_n_parallel = CONFIGS[CONFIG]
    out_dir = RESULTS_DIR / f"{CONFIG}{RESULT_SUFFIX}" / "llama_mc"
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_single_flight_config(CONFIG, extra_env):
        log(f"[{CONFIG}] single-flight generation mode; native MC uses per-task candidate slots")
    log(f"[{CONFIG}] MODEL={MODEL} GPU={GPU} model={model.name} tasks={TASKS}")

    results = [run_task(task, model, extra_env, out_dir, config_n_parallel) for task in TASKS]
    write_summary(out_dir, results)


if __name__ == "__main__":
    main()
