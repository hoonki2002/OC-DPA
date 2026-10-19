#!/usr/bin/env python3
"""Aggregate benchmark result files into one CSV.

The collector is intentionally file-based: it walks experiments/results and
keeps source_path/run_id columns so repeated or validation runs remain visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from typing import Any


REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = REPO / "experiments" / "results"
DEFAULT_OUT = DEFAULT_RESULTS / "bench_scores_all.csv"
DEFAULT_FINAL_OUT = DEFAULT_RESULTS / "bench_scores_final.csv"


FIELDNAMES = [
    "model",
    "config",
    "raw_config",
    "series",
    "benchmark_family",
    "benchmark",
    "metric",
    "value",
    "stderr",
    "n",
    "effective_gb",
    "k_bar",
    "ctx_size",
    "batch_size",
    "ubatch_size",
    "n_parallel",
    "gpu",
    "vram_cap_mb",
    "request_parallel",
    "method",
    "profile_file",
    "run_id",
    "source_path",
]


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def series_for_config(config: str) -> str:
    if config.startswith("uniform_"):
        return "uniform"
    if config.startswith("dynamic_"):
        return "dynamic"
    if config.startswith("sched_"):
        return "sched"
    if re.fullmatch(r"Q\d.*", config):
        return "baseline"
    return "other"


def canonical_config(config: str, family: str) -> str:
    if family == "tps":
        return config
    config = re.sub(r"_(arc_[ce]|piqa|winogrande|hellaswag)_serial$", "", config)
    config = re.sub(r"_arce_rerun$", "", config)
    return config


def kbar_for_config(config: str) -> str:
    sched = re.search(r"sched_pp_k(\d+(?:_\d+)?)_k(\d+(?:_\d+)?)", config)
    if sched:
        k0 = float(sched.group(1).replace("_", "."))
        k1 = float(sched.group(2).replace("_", "."))
        return f"{(k0 + k1) / 2.0:.4f}".rstrip("0").rstrip(".")
    match = re.search(r"_k(\d+(?:_\d+)?)", config)
    if not match:
        return ""
    return match.group(1).replace("_", ".")


def config_for_mode_kbar(mode: str, k_raw: str) -> str:
    try:
        k = float(k_raw)
    except ValueError:
        k_norm = k_raw.replace(".", "_")
    else:
        k_norm = f"{k:g}".replace(".", "_")
    return f"{mode}_k{k_norm}"


def find_effective_weights(results_root: pathlib.Path) -> dict[tuple[str, str], str]:
    weights: dict[tuple[str, str], str] = {}
    by_model_k: dict[str, dict[float, float]] = {}
    kld_paths = list(results_root.glob("*/kld/bytes_vs_kld.csv"))
    legacy_models = {path.parents[1].name for path in kld_paths}
    kld_paths.extend(
        path for path in results_root.glob("*/kld_sweep/bytes_vs_kld.csv")
        if path.parents[1].name not in legacy_models
    )
    for path in kld_paths:
        model = path.parents[1].name
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                label = row.get("label", "")
                mode = row.get("mode", "")
                gb = row.get("effective_gb", "")
                if not gb:
                    continue
                k_raw = row.get("K_bar", "")
                if k_raw:
                    by_model_k.setdefault(model, {})[float(k_raw)] = float(gb)
                if mode == "baseline":
                    cfg = label
                elif mode == "uniform":
                    cfg = config_for_mode_kbar("uniform", row.get("K_bar", ""))
                elif mode == "dynamic":
                    cfg = config_for_mode_kbar("dynamic", row.get("K_bar", ""))
                else:
                    continue
                weights[(model, cfg)] = gb
    for model, k_to_gb in by_model_k.items():
        if len(k_to_gb) < 2:
            continue
        ks = sorted(k_to_gb)
        slope = (k_to_gb[ks[-1]] - k_to_gb[ks[0]]) / (ks[-1] - ks[0])
        intercept = k_to_gb[ks[0]] - slope * ks[0]
        for cfg in ("sched_pp_k5_k2_5", "sched_pp_k6_k2_5"):
            m = re.search(r"sched_pp_k(\d+(?:_\d+)?)_k(\d+(?:_\d+)?)", cfg)
            if not m:
                continue
            k0 = float(m.group(1).replace("_", "."))
            k1 = float(m.group(2).replace("_", "."))
            weights[(model, cfg)] = f"{intercept + slope * ((k0 + k1) / 2.0):.9f}"
    return weights


def base_row(model: str, raw_config: str, family: str, benchmark: str,
             metric: str, source: pathlib.Path,
             weights: dict[tuple[str, str], str]) -> dict[str, Any]:
    config = canonical_config(raw_config, family)
    return {
        "model": model,
        "config": config,
        "raw_config": raw_config,
        "series": series_for_config(config),
        "benchmark_family": family,
        "benchmark": benchmark,
        "metric": metric,
        "value": "",
        "stderr": "",
        "n": "",
        "effective_gb": weights.get((model, config), ""),
        "k_bar": kbar_for_config(config),
        "ctx_size": "",
        "batch_size": "",
        "ubatch_size": "",
        "n_parallel": "",
        "gpu": "",
        "vram_cap_mb": "",
        "request_parallel": "",
        "method": "",
        "profile_file": "",
        "run_id": "",
        "source_path": rel(source),
        "_source_mtime": source.stat().st_mtime,
    }


def add_llama_mc(results_root: pathlib.Path, rows: list[dict[str, Any]],
                 weights: dict[tuple[str, str], str]) -> None:
    for path in sorted(results_root.glob("*/bench/*/llama_mc/*.json")):
        model = path.parents[3].name
        config = path.parents[1].name
        data = json.loads(path.read_text())
        task = str(data.get("benchmark") or path.stem)
        row = base_row(model, config, "llama_mc", task, "acc_pct", path, weights)
        runtime = data.get("runtime", {}) or {}
        env = data.get("dpa_moe_env", {}) or {}
        row.update({
            "value": data.get("acc_pct", ""),
            "stderr": data.get("stderr_pct", ""),
            "n": data.get("n_total", ""),
            "ctx_size": runtime.get("ctx_size", ""),
            "batch_size": runtime.get("batch_size", ""),
            "ubatch_size": runtime.get("ubatch_size", ""),
            "n_parallel": data.get("n_parallel", runtime.get("n_parallel", "")),
            "gpu": runtime.get("cuda_visible_devices", ""),
            "method": data.get("method", ""),
            "profile_file": data.get("profile_file") or env.get("DPA_MOE_KBAR_PROFILE_FILE", ""),
        })
        rows.append(row)


def add_wikitext_ppl(results_root: pathlib.Path, rows: list[dict[str, Any]],
                     weights: dict[tuple[str, str], str]) -> None:
    metrics = [
        "mean_kld", "median_kld", "max_kld", "ppl_q", "ppl_base",
        "ppl_ratio", "same_top_p_pct", "elapsed_sec",
    ]
    new_paths = list(results_root.glob("*/kld/*/wikitext_ppl.json"))
    new_keys = {(path.parents[2].name, path.parents[0].name) for path in new_paths}
    legacy_paths = [
        path for path in results_root.glob("*/bench/*/wikitext_ppl/wikitext_ppl.json")
        if (path.parents[3].name, path.parents[1].name) not in new_keys
    ]
    legacy_paths.extend(
        path for path in results_root.glob("*/kld_sweep/*/wikitext_ppl/wikitext_ppl.json")
        if (path.parents[3].name, path.parents[1].name) not in new_keys
    )
    paths = [*new_paths, *legacy_paths]
    root = results_root.resolve()
    for path in sorted({p.resolve() for p in paths}):
        rel = path.relative_to(root)
        model = rel.parts[0]
        config = rel.parts[2]
        data = json.loads(path.read_text())
        for metric in metrics:
            if data.get(metric, "") == "":
                continue
            row = base_row(model, config, "wikitext_ppl", "wikitext_ppl", metric, path, weights)
            row.update({
                "value": data.get(metric, ""),
                "n": data.get("chunks", ""),
                "ctx_size": data.get("ctx_size", ""),
                "batch_size": data.get("batch_size", ""),
                "ubatch_size": data.get("ubatch_size", ""),
                "method": data.get("method", ""),
                "profile_file": data.get("profile_file", ""),
            })
            rows.append(row)


def metric_stderr_key(metric: str) -> str:
    if "," in metric:
        base, suffix = metric.split(",", 1)
        return f"{base}_stderr,{suffix}"
    return f"{metric}_stderr"


def add_lm_eval(results_root: pathlib.Path, rows: list[dict[str, Any]],
                weights: dict[tuple[str, str], str]) -> None:
    for path in sorted(results_root.glob("*/bench/*/lm_eval/**/results_*.json")):
        model = path.parents[4].name
        config = path.parents[2].name
        data = json.loads(path.read_text())
        run_id = path.stem.removeprefix("results_")
        for task, result in (data.get("results", {}) or {}).items():
            if not isinstance(result, dict):
                continue
            for metric, value in result.items():
                if metric == "alias" or metric.endswith("_stderr") or "_stderr," in metric:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                stderr = result.get(metric_stderr_key(metric), "")
                row = base_row(model, config, "lm_eval", task, metric, path, weights)
                row.update({
                    "value": value,
                    "stderr": stderr,
                    "n": (data.get("n-samples", {}) or {}).get(task, ""),
                    "method": "lm_eval",
                    "run_id": run_id,
                })
                rows.append(row)


def add_kld_sweep(results_root: pathlib.Path, rows: list[dict[str, Any]],
                  weights: dict[tuple[str, str], str]) -> None:
    metrics = [
        "mean_kld", "median_kld", "max_kld", "ppl_q", "ppl_base",
        "ppl_ratio", "same_top_p_pct",
    ]
    paths = list(results_root.glob("*/kld/bytes_vs_kld.csv"))
    legacy_models = {path.parents[1].name for path in paths}
    paths.extend(
        path for path in results_root.glob("*/kld_sweep/bytes_vs_kld.csv")
        if path.parents[1].name not in legacy_models
    )
    for path in sorted(paths):
        model = path.parents[1].name
        with path.open(newline="") as fh:
            for src in csv.DictReader(fh):
                label = src.get("label", "")
                mode = src.get("mode", "")
                if mode == "baseline":
                    config = label
                elif mode in {"uniform", "dynamic"}:
                    config = config_for_mode_kbar(mode, src.get("K_bar", ""))
                else:
                    config = label
                for metric in metrics:
                    if not src.get(metric):
                        continue
                    row = base_row(model, config, "kld", "wikitext", metric, path, weights)
                    row.update({
                        "series": mode,
                        "value": src.get(metric, ""),
                        "effective_gb": src.get("effective_gb", ""),
                        "k_bar": src.get("K_bar", ""),
                        "method": "bytes_vs_kld",
                    })
                    rows.append(row)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("model", "")),
            str(row.get("config", "")),
            str(row.get("benchmark_family", "")),
            str(row.get("benchmark", "")),
            str(row.get("metric", "")),
        )
        prev = merged.get(key)
        if prev is None or float(row.get("_source_mtime", 0.0)) >= float(prev.get("_source_mtime", 0.0)):
            merged[key] = row
    return sorted(merged.values(), key=lambda r: (
        str(r.get("model", "")),
        str(r.get("config", "")),
        str(r.get("benchmark_family", "")),
        str(r.get("benchmark", "")),
        str(r.get("metric", "")),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--final-out", type=pathlib.Path, default=DEFAULT_FINAL_OUT)
    parser.add_argument(
        "--include-legacy-wikitext-ppl",
        action="store_true",
        help="Also collect legacy bench/*/wikitext_ppl rows. KLD sweep rows are always collected from */kld/bytes_vs_kld.csv.",
    )
    args = parser.parse_args()

    weights = find_effective_weights(args.results_root)
    rows: list[dict[str, Any]] = []
    add_llama_mc(args.results_root, rows, weights)
    if args.include_legacy_wikitext_ppl:
        add_wikitext_ppl(args.results_root, rows, weights)
    add_lm_eval(args.results_root, rows, weights)
    add_kld_sweep(args.results_root, rows, weights)
    rows.sort(key=lambda r: (
        str(r.get("model", "")),
        str(r.get("config", "")),
        str(r.get("benchmark_family", "")),
        str(r.get("benchmark", "")),
        str(r.get("metric", "")),
        str(r.get("run_id", "")),
    ))
    write_csv(args.out, rows)
    final = final_rows(rows)
    write_csv(args.final_out, final)
    print(f"wrote {args.out} ({len(rows)} rows)")
    print(f"wrote {args.final_out} ({len(final)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
