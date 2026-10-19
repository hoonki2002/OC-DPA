#!/usr/bin/env python3
"""Plot multiple-choice accuracy vs effective model weight with 2-sigma bands."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from bytes_calc import effective_model_bytes, full_model_bytes


TASK_LABELS = {
    "arc_c": "ARC-C",
    "arc_e": "ARC-Easy",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "hellaswag": "HellaSwag",
}


def read_scores(llama_mc_dir: pathlib.Path) -> dict[str, tuple[float, float]]:
    """Return task -> (accuracy %, stderr %) from JSONs or summary.csv."""
    scores: dict[str, tuple[float, float]] = {}

    for task in TASK_LABELS:
        path = llama_mc_dir / f"{task}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        scores[task] = (float(data["acc_pct"]), float(data.get("stderr_pct", 0.0)))

    summary = llama_mc_dir / "summary.csv"
    if summary.exists():
        with summary.open() as fh:
            for row in csv.DictReader(fh):
                task = row.get("benchmark", "")
                if task in TASK_LABELS and task not in scores:
                    scores[task] = (
                        float(row["acc_pct"]),
                        float(row.get("stderr_pct", 0.0) or 0.0),
                    )
    return scores


def config_weight_gb(config: str, model_dir: pathlib.Path, ss_gguf: pathlib.Path,
                     n_experts: int, n_layers: int,
                     cache: dict[tuple[str, str], float]) -> tuple[str, float]:
    m = re.fullmatch(r"(uniform|dynamic)_k(\d+)", config)
    if m:
        key = ("stream", m.group(2))
        if key not in cache:
            cache[key] = effective_model_bytes(
                ss_gguf, float(m.group(2)), n_experts=n_experts, n_layers=n_layers) / 1e9
        return m.group(1), cache[key]

    baseline = model_dir / "baseline" / f"{model_dir.name}-{config}.gguf"
    if baseline.exists():
        key = ("baseline", str(baseline))
        if key not in cache:
            cache[key] = full_model_bytes(baseline) / 1e9
        return "baseline", cache[key]

    raise FileNotFoundError(f"cannot infer weight for {config}: missing {baseline}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen3-30B-A3B")
    ap.add_argument("--bench-root", type=pathlib.Path,
                    default=REPO / "experiments/results")
    ap.add_argument("--model-root", type=pathlib.Path,
                    default=REPO / "cache/models")
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "experiments/results/Qwen3-30B-A3B/bench/bench_score_vs_effective_weight.png")
    ap.add_argument("--csv-out", type=pathlib.Path, default=None)
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=48)
    args = ap.parse_args()

    bench_dir = args.bench_root / args.model / "bench"
    model_dir = args.model_root / args.model
    ss_gguf = model_dir / "chunked" / "ss_anybcq.gguf"
    csv_out = args.csv_out or args.out.with_suffix(".csv")
    weight_cache: dict[tuple[str, str], float] = {}

    rows: list[dict[str, str | float]] = []
    for llama_mc_dir in sorted(bench_dir.glob("*/llama_mc")):
        config = llama_mc_dir.parent.name
        try:
            series, gb = config_weight_gb(
                config, model_dir, ss_gguf, args.n_experts, args.n_layers, weight_cache)
        except FileNotFoundError:
            continue
        for task, (acc, stderr) in read_scores(llama_mc_dir).items():
            rows.append({
                "config": config,
                "series": series,
                "task": task,
                "effective_gb": gb,
                "acc_pct": acc,
                "stderr_pct": stderr,
                "err_2sigma_pct": 2.0 * stderr,
            })

    if not rows:
        raise SystemExit(f"no llama_mc scores found under {bench_dir}")

    rows.sort(key=lambda r: (str(r["task"]), str(r["series"]),
                             float(r["effective_gb"]), str(r["config"])))
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "config", "series", "task", "effective_gb",
                "acc_pct", "stderr_pct", "err_2sigma_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("config,series,task,effective_gb,acc_pct,stderr_pct,err_2sigma_pct")
    for r in rows:
        print(f"{r['config']},{r['series']},{r['task']},"
              f"{float(r['effective_gb']):.4f},{float(r['acc_pct']):.4f},"
              f"{float(r['stderr_pct']):.4f},{float(r['err_2sigma_pct']):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"wrote {csv_out}; matplotlib unavailable, skipped PNG", file=sys.stderr)
        return 0

    colors = {"baseline": "#1f77b4", "uniform": "#444444", "dynamic": "#d62728"}
    markers = {"baseline": "*", "uniform": "o", "dynamic": "v"}

    by_key: dict[tuple[str, str], list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["task"]), str(row["series"]))].append(row)

    tasks = [task for task in TASK_LABELS if any(r["task"] == task for r in rows)]
    ncols = min(3, len(tasks))
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows), sharex=True)
    axes = list(axes_grid.flat) if hasattr(axes_grid, "flat") else [axes_grid]
    for ax, task in zip(axes, tasks):
        for series in ["baseline", "uniform", "dynamic"]:
            vals = by_key.get((task, series), [])
            vals.sort(key=lambda r: float(r["effective_gb"]))
            if not vals:
                continue
            xs = [float(r["effective_gb"]) for r in vals]
            ys = [float(r["acc_pct"]) for r in vals]
            err = [float(r["err_2sigma_pct"]) for r in vals]
            lo = [y - e for y, e in zip(ys, err)]
            hi = [y + e for y, e in zip(ys, err)]
            color = colors[series]

            if len(vals) >= 2:
                ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.errorbar(xs, ys, yerr=err, marker=markers[series], color=color,
                        linewidth=1.4, markersize=8 if series != "baseline" else 11,
                        capsize=3, label=f"{series} (±2σ)")
            for x, y, row in zip(xs, ys, vals):
                label = str(row["config"]).replace("uniform_", "U").replace("dynamic_", "D")
                ax.annotate(label, (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=color)

        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Effective model weight (GB)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.legend(fontsize=8)

    for ax in axes[len(tasks):]:
        ax.axis("off")

    fig.suptitle(f"{args.model}: multiple-choice accuracy vs effective model weight, ±2σ")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"wrote {csv_out}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
