#!/usr/bin/env python3
"""Run Wikitext-2 perplexity/KLD for one 4_bench config."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))
from _config import (
    CONFIGS, KLD_RESULTS_DIR, MODEL, WIKITEXT_BATCH, WIKITEXT_CHUNKS,
    WIKITEXT_CTX, WIKITEXT_FILE, WIKITEXT_PROFILE_FILE, WIKITEXT_SEED,
    WIKITEXT_UBATCH,
)


REPO = Path(__file__).resolve().parents[3]
BIN_PERPLEXITY = REPO / "third_party/llama.cpp/build-dpa-moe/bin/llama-perplexity"
SHARED = REPO / "cache/experiment" / f"_shared_{MODEL}"

GPU = os.environ.get("GPU", "0")
CONFIG = os.environ.get("CONFIG", "Q5_K_M")
FORCE = os.environ.get("FORCE", "0") == "1"
CTX_SIZE = WIKITEXT_CTX
BATCH_SIZE = WIKITEXT_BATCH
UBATCH_SIZE = WIKITEXT_UBATCH
CHUNKS = WIKITEXT_CHUNKS
EVAL_FILE = WIKITEXT_FILE

DPA_MOE_ENV_KEYS = (
    "DPA_MOE_KBAR", "DPA_MOE_KBAR_ALLOCATOR", "DPA_MOE_KBAR_PROFILE_FILE",
    "DPA_MOE_KBAR_SCHEDULE_TOKENS", "DPA_MOE_KBAR_SCHEDULE_VALUES",
    "DPA_MOE_KBAR_PP_BEGIN", "DPA_MOE_KBAR_PP_END",
    "DPA_MOE_KBAR_TG_BEGIN", "DPA_MOE_KBAR_TG_END",
    "DPA_MOE_KBAR_TG_DECAY_START", "DPA_MOE_KBAR_TG_DECAY_END",
    "DPA_MOE_KBAR_PP_DECAY_END",
    "DPA_MOE_PHASE_REASONING_KBAR", "DPA_MOE_PHASE_GENERATION_KBAR",
    "DPA_MOE_PHASE_OPEN_TOKEN_IDS", "DPA_MOE_PHASE_CLOSE_TOKEN_IDS",
)

PATTERNS = {
    "mean_kld": r"^(?:Mean\s+KLD|Mean\s+KLD):\s+([\d.eE+-]+)",
    "median_kld": r"^(?:Median\s+KLD|Median\s+KLD):\s+([\d.eE+-]+)",
    "max_kld": r"^(?:Maximum\s+KLD|Maximum\s+KLD):\s+([\d.eE+-]+)",
    "ppl_q": r"^(?:q-ppl|Mean\s+PPL\(Q\))\s*:?\s+([\d.eE+-]+)",
    "ppl_base": r"^(?:base-ppl|Mean\s+PPL\(base\))\s*:?\s+([\d.eE+-]+)",
    "ppl_ratio": r"^(?:ppl\(q\)/ppl\(base\)|Mean\s+PPL\(Q\)/PPL\(base\))\s*:?\s+([\d.eE+-]+)",
    "same_top_p_pct": r"^Same top p:\s+([\d.]+)",
}


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            metrics[key] = float(match.group(1))
    if "ppl_q" not in metrics:
        raise RuntimeError("could not parse q-ppl from llama-perplexity output")
    return metrics


def make_result(metrics: dict[str, float], log_path: Path, profile_file: str = "",
                elapsed: float | str = "") -> dict:
    return {
        "config": CONFIG,
        "model": MODEL,
        "benchmark": "wikitext_ppl",
        "profile_file": profile_file,
        "eval_file": str(EVAL_FILE),
        "chunks": CHUNKS,
        "ctx_size": CTX_SIZE,
        "batch_size": BATCH_SIZE,
        "ubatch_size": UBATCH_SIZE,
        **metrics,
        "elapsed_sec": elapsed,
        "method": "llama-perplexity --kl-divergence",
        "log": str(log_path),
    }


def ref_logits_path() -> Path:
    path = SHARED / "ref_logits_b1.bin"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/4_bench/kld_sweep/gen_ref.sh first")
    return path


def write_summary(out_dir: Path, result: dict) -> None:
    out = out_dir / "summary.csv"
    fields = [
        "config", "model", "benchmark", "profile_file", "eval_file",
        "chunks", "ctx_size", "batch_size",
        "ubatch_size", "ppl_q", "ppl_base", "ppl_ratio", "mean_kld",
        "median_kld", "max_kld", "same_top_p_pct", "elapsed_sec", "method", "log",
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in fields})
    log(f"[{CONFIG}] summary -> {out}")


def main() -> int:
    if CONFIG not in CONFIGS:
        raise SystemExit(f"Unknown CONFIG={CONFIG!r}. Valid: {sorted(CONFIGS)}")
    if not BIN_PERPLEXITY.exists():
        raise SystemExit(f"missing {BIN_PERPLEXITY}")
    if not EVAL_FILE.exists():
        raise SystemExit(f"missing {EVAL_FILE}")

    model, extra_env, _n_parallel = CONFIGS[CONFIG]
    phase_env = dict(extra_env)
    if phase_env.get("DPA_MOE_KBAR_ALLOCATOR") == "profile":
        phase_env["DPA_MOE_KBAR_PROFILE_FILE"] = str(WIKITEXT_PROFILE_FILE)
    if not model.exists():
        raise SystemExit(f"missing model for {CONFIG}: {model}")

    out_dir = KLD_RESULTS_DIR / CONFIG
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "wikitext_ppl.log"
    json_path = out_dir / "wikitext_ppl.json"

    if json_path.exists() and not FORCE:
        write_summary(out_dir, json.loads(json_path.read_text()))
        return 0
    if log_path.exists() and not FORCE:
        try:
            profile_file = phase_env.get("DPA_MOE_KBAR_PROFILE_FILE", "")
            result = make_result(parse_metrics(log_path.read_text(errors="replace")), log_path, profile_file)
        except RuntimeError:
            pass
        else:
            json_path.write_text(json.dumps(result, indent=2) + "\n")
            write_summary(out_dir, result)
            log(f"[{CONFIG}] recovered q-ppl={result['ppl_q']:.6g} from existing log")
            return 0

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU}
    for key in DPA_MOE_ENV_KEYS:
        env.pop(key, None)
    env.update({k: v for k, v in phase_env.items() if v})

    cmd = [
        str(BIN_PERPLEXITY),
        "-m", str(model),
        "-f", str(EVAL_FILE),
        "--kl-divergence",
        "--kl-divergence-base", str(ref_logits_path()),
        "-c", str(CTX_SIZE),
        "-b", str(BATCH_SIZE),
        "-ub", str(UBATCH_SIZE),
        "-ngl", "999",
        "-s", str(WIKITEXT_SEED),
        "--no-warmup",
        "--chunks", str(CHUNKS),
        "--fit", "off",
    ]

    log(f"[{CONFIG}] wikitext PPL starting GPU={GPU} chunks={CHUNKS}")
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - started
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"wikitext PPL failed rc={proc.returncode}; see {log_path}")

    metrics = parse_metrics(proc.stdout)
    result = make_result(metrics, log_path, phase_env.get("DPA_MOE_KBAR_PROFILE_FILE", ""), elapsed)
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    write_summary(out_dir, result)
    log(f"[{CONFIG}] q-ppl={metrics['ppl_q']:.6g} ratio={metrics.get('ppl_ratio', 0.0):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
