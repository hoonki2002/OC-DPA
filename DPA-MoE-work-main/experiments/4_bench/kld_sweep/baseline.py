"""Run baseline GGUF KLD measurements and emit a plot-ready CSV.

The layout sweep compares ss_anybcq KBar points against plain GGUF
baselines.  This script is intentionally idempotent: existing valid
``kld_<quant>.out`` files are reused unless ``FORCE=1`` is set.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/3_dynamic_precision"))
sys.path.insert(0, str(REPO / "experiments/4_bench/analysis"))

from _config import REPO_ROOT, active_variant, ppl_bin, shared_root, variant_name
from _parse import parse_kld_output
from bytes_calc import full_model_bytes

FORCE = os.environ.get("FORCE", "0") == "1"


def kld_results_dir() -> pathlib.Path:
    out = REPO / "experiments/results" / variant_name() / "kld"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _quant_path(baseline_dir: pathlib.Path, prefix: str, quant: str) -> pathlib.Path:
    hyphen = baseline_dir / f"{prefix}-{quant}.gguf"
    if hyphen.exists():
        return hyphen
    dotted = baseline_dir / f"{prefix}.{quant}.gguf"
    if dotted.exists():
        return dotted
    return hyphen


def _run_one(*, quant: str, model: pathlib.Path, ref_logits: pathlib.Path,
             wiki: pathlib.Path, bin_path: pathlib.Path, ctx_size: int,
             chunks: int, batch: int, ubatch: int,
             gpu: str, outdir: pathlib.Path) -> dict[str, float] | None:
    out = outdir / f"kld_{quant}.out"
    err = outdir / f"kld_{quant}.err"
    outdir.mkdir(parents=True, exist_ok=True)

    if not FORCE and out.exists() and out.stat().st_size > 100:
        stats = parse_kld_output(out.read_text(errors="replace"))
        if stats is not None:
            return stats

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        str(bin_path),
        "-m", str(model),
        "-f", str(wiki),
        "--kl-divergence",
        "--kl-divergence-base", str(ref_logits),
        "-c", str(ctx_size),
        "-b", str(batch),
        "-ub", str(ubatch),
        "-ngl", "999",
        "-s", "0",
        "--no-warmup",
        "--chunks", str(chunks),
        "--fit", "off",
    ]
    print(f"[baseline] {quant}  llama-perplexity...", flush=True)
    t0 = time.time()
    with open(out, "w") as fout, open(err, "w") as ferr:
        proc = subprocess.run(cmd, env=env, stdout=fout, stderr=ferr)
    print(f"[baseline] {quant}  done in {(time.time() - t0)/60:.1f} min  "
          f"rc={proc.returncode}", flush=True)
    if proc.returncode != 0:
        return None
    return parse_kld_output(out.read_text(errors="replace"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", default="Q2_K,Q5_K_M,Q8_0",
                    help="comma-separated GGUF quant suffixes to measure")
    ap.add_argument("--baseline-dir", type=pathlib.Path, default=None)
    ap.add_argument("--baseline-prefix", default=None,
                    help="filename prefix before '-<quant>.gguf'")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ubatch", type=int, default=1)
    ap.add_argument("--eval-file", type=pathlib.Path, default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out-csv", type=pathlib.Path, default=None)
    args = ap.parse_args()

    variant = active_variant()
    name = variant["name"]
    baseline_dir = args.baseline_dir or (
        REPO_ROOT / "cache" / "models" / name / "baseline")
    prefix = args.baseline_prefix or name
    out_csv = args.out_csv or (kld_results_dir() / "baseline_quants.csv")

    ref_logits = shared_root() / "ref_logits_b1.bin"
    if not ref_logits.exists():
        legacy = shared_root() / "ref_logits.bin"
        if legacy.exists():
            ref_logits = legacy
    wiki = args.eval_file or (REPO_ROOT / "cache" / "datasets" / "wikitext-2-raw-test.txt")
    bin_path = ppl_bin()
    outdir = shared_root() / "baseline_kld"

    for needed in (ref_logits, wiki, bin_path):
        if not needed.exists():
            sys.exit(f"missing: {needed}")

    rows: list[dict[str, str]] = []
    for quant in [q.strip() for q in args.baselines.split(",") if q.strip()]:
        model = _quant_path(baseline_dir, prefix, quant)
        if not model.exists():
            sys.exit(f"missing baseline GGUF for {quant}: {model}")
        stats = _run_one(
            quant=quant, model=model, ref_logits=ref_logits, wiki=wiki,
            bin_path=bin_path, ctx_size=args.ctx_size, chunks=args.chunks,
            batch=args.batch, ubatch=args.ubatch,
            gpu=args.gpu, outdir=outdir)
        if stats is None:
            print(f"[baseline] FAILED {quant}", flush=True)
            continue
        rows.append({
            "name": quant,
            "gb": f"{full_model_bytes(model) / 1e9:.9f}",
            "mean_kld": f"{stats['mean_kld']:.6f}",
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "gb", "mean_kld"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[baseline] wrote {out_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
