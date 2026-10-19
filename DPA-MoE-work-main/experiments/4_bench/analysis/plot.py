"""KLD vs effective-model-weight plot — final figure style.

X-axis: effective full model weight in GB (log scale).
  For baseline quants (Q3_K_M … Q8_0): total bytes of all tensors in the
    GGUF — the actual memory needed to load the model.
  For ours (dynamic/uniform): non-expert bytes + ALL expert bytes at K̄
    = non_expert + n_experts × n_layers × K̄ × chunk_bytes_per_expert.
    Equivalent to per_token_moe × (n_experts / top_k) + non_expert.
    Comparable to baseline full-model sizes.

Y-axis: Mean KLD vs BF16 reference (log scale).

Baseline quants (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) are
hard-coded from measurements on the same Qwen3-30B-A3B with
the unsloth GGUFs.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
from typing import Optional

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # bytes_calc in same dir (analysis/)


def make_effective_gb(gguf_path: pathlib.Path,
                       n_experts: Optional[int] = None,
                       n_layers: Optional[int] = None):
    """Read GGUF once; return a kbar→GB callable that reuses the cached bytes."""
    from bytes_calc import gguf_bytes
    b = gguf_bytes(gguf_path)
    managed_experts = int(b.get("moe_managed_experts", 0))
    managed_layers = int(b.get("moe_managed_layers", 0))
    if n_experts is None:
        n_experts = managed_experts
    if n_layers is None:
        n_layers = managed_layers
    # Effective model weight counts every fixed/unmanaged tensor once, plus
    # the managed MoE expert chunks at Kbar.  ``other`` includes architecture
    # fixed weights such as DeepSeek's dense layer-0 FFN and F16 shared experts.
    non_expert = (
        b["attn"] + b["lm_head"] + b["token_embd"] + b["router"]
        + b["norms"] + b["other"] + b["moe_meta"]
    )
    chunk_bytes = b["moe_chunk_per_expert"]
    def _gb(kbar: float) -> float:
        return (non_expert + int(n_experts * n_layers * kbar * chunk_bytes)) / 1e9
    return _gb


# Baselines — full model weight in GB (total bytes of all tensors).
# For Q3/Q4/Q8: measured directly from actual on-disk unsloth GGUF tensor sizes.
# For Q5/Q6: taken from unsloth/Qwen3-30B-A3B-GGUF HuggingFace file listing.
# KLD: measured with b=1 ub=1 chunks=10 wikitext-2-test vs BF16 ref.
QWEN3_BASELINES = [
    # (name, full_model_GB, mean_KLD)
    ("Q3_K_M", 14.863, 0.05131),
    ("Q4_K_M", 18.551, 0.01804),
    ("Q5_K_M", 21.7,   0.00831),
    ("Q6_K",   25.1,   0.00478),
    ("Q8_0",   32.478, 0.00235),
]

DEEPSEEK_BASELINES = [
    ("Q3_K_M", 8.5603, 0.097335),
    # Size-only estimate from local Q8_0 tensor shapes with Q8_0 tensors
    # projected to Q5_K byte density. No KLD is plotted until measured.
    ("Q5_K_M est.", 11.2704, None),
    ("Q8_0", 17.4099, 0.002947),
]


def _read_baselines(path: pathlib.Path) -> list[tuple[str, float, Optional[float]]]:
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            kld = r.get("mean_kld", "")
            rows.append((r["name"], float(r["gb"]), float(kld) if kld else None))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=pathlib.Path,
                     default=pathlib.Path("experiments/results/Qwen3-30B-A3B/kld/layout_summary.csv"))
    ap.add_argument("--summary-prev", type=pathlib.Path, default=None,
                     help="Previous sweep CSV to overlay (drawn lighter, dashed)")
    ap.add_argument("--bench-dir", type=pathlib.Path,
                     default=pathlib.Path("cache/experiment/Qwen3-30B-A3B/3_dynamic_precision/bench"))
    ap.add_argument("--out", type=pathlib.Path,
                     default=pathlib.Path("experiments/results/Qwen3-30B-A3B/kld/bytes_vs_kld.png"))
    ap.add_argument("--out-csv", type=pathlib.Path, default=None,
                    help="Plot-ready CSV. Defaults to bytes_vs_kld.csv next to --out.")
    ap.add_argument("--gguf", type=pathlib.Path,
                     default=pathlib.Path("cache/models/Qwen3-30B-A3B/chunked/ss_anybcq.gguf"))
    ap.add_argument("--n-experts", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None,
                     help="number of MoE layers contributing expert chunks")
    ap.add_argument("--title", default="Qwen3-30B-A3B")
    ap.add_argument("--baselines", choices=["qwen3", "deepseek", "none"],
                    default="qwen3")
    ap.add_argument("--baseline-csv", type=pathlib.Path, default=None,
                    help="CSV with columns: name,gb,mean_kld")
    ap.add_argument("--exclude-kbar", action="append", type=float, default=[],
                    help="Omit points with this K_bar from the plot; repeatable")
    args = ap.parse_args()
    # Make sibling scripts (bytes_calc.py) importable.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    # Read GGUF once — effective_gb reuses the cached bytes dict.
    effective_gb = make_effective_gb(
        args.gguf, n_experts=args.n_experts, n_layers=args.n_layers)

    def _load_summary(path: pathlib.Path) -> list[dict[str, str | float]]:
        out = []
        with open(path) as fh:
            for r in csv.DictReader(fh):
                try:
                    mode = r.get("mode", "")
                    if mode not in ("dynamic", "uniform"):
                        continue
                    klp = float(r["mean_kld"])
                    kbar = float(r["K_bar"])
                except (ValueError, KeyError):
                    continue
                if any(abs(kbar - x) < 1e-6 for x in args.exclude_kbar):
                    continue
                if mode == "dynamic":
                    if kbar >= 7.5:
                        continue
                    tau_str = r.get("tau", "")
                    eps_str = r.get("eps", "")
                    if eps_str:
                        series = "dynamic_eps"
                        lbl = f"ε={eps_str} → K̄={kbar:.2f}"
                    elif tau_str:
                        series = "dynamic_tau"
                        lbl = f"τ={tau_str}"
                    else:
                        series = "dynamic_kbar"
                        lbl = f"K̄={kbar:g}"
                else:
                    series = "uniform"
                    lbl = f"K={int(round(kbar))}"
                gb = effective_gb(kbar)
                row = dict(r)
                row.update({
                    "series": series,
                    "effective_gb": gb,
                    "label": lbl,
                })
                out.append(row)
        return out

    pts: list[dict[str, str | float]] = _load_summary(args.summary)
    pts_prev: list[dict[str, str | float]] = (
        _load_summary(args.summary_prev) if args.summary_prev else []
    )
    size_only: list[tuple[str, float]] = []

    inferred_baseline_csv = None
    if args.baseline_csv is None:
        for candidate in (
            args.summary.parent / "baseline_quants.csv",
            args.summary.parent.parent / "q2_q5_q8_baselines.csv",
            args.summary.parent.parent / "baseline_quants.csv",
        ):
            if candidate.exists():
                inferred_baseline_csv = candidate
                break

    if args.baseline_csv is not None or inferred_baseline_csv is not None:
        baselines = _read_baselines(args.baseline_csv or inferred_baseline_csv)
    elif args.baselines == "deepseek":
        baselines = DEEPSEEK_BASELINES
    elif args.baselines == "qwen3":
        baselines = QWEN3_BASELINES
    else:
        baselines = []

    # Add baselines.
    for name, gb, klp in baselines:
        if klp is None:
            size_only.append((name, gb))
        else:
            pts.append({
                "series": "baseline",
                "effective_gb": gb,
                "mean_kld": klp,
                "label": name,
                "mode": "baseline",
                "K_bar": "",
            })

    pts.sort(key=lambda r: float(r["effective_gb"]))

    out_csv = args.out_csv or (args.out.parent / "bytes_vs_kld.csv")
    metric_fields = [
        "mean_kld", "median_kld", "max_kld",
        "ppl_q", "ppl_base", "ppl_ratio", "same_top_p_pct",
    ]
    csv_fields = [
        "series", "label", "mode", "K_bar", "effective_gb",
        *metric_fields,
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for row in pts:
            w.writerow(row)
    print(f"wrote {out_csv}", file=sys.stderr)

    print(f"{'series':12s}  {'GB/tok':>7s}  {'KLD':>10s}  label")
    for row in pts:
        s = str(row["series"])
        gb = float(row["effective_gb"])
        klp = float(row["mean_kld"])
        lbl = str(row["label"])
        print(f"  {s:11s}  {gb:7.3f}  {klp:10.5f}  {lbl}")
    for lbl, gb in size_only:
        print(f"  {'baseline':11s}  {gb:7.3f}  {'':>10s}  {lbl} (size only)")

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        colors = {
            "uniform":      ("#444444", "o"),
            "dynamic_kbar": ("#d62728", "v"),   # memory-budget DP
            "dynamic_eps":  ("#2ca02c", "^"),   # error-budget DP
            "dynamic_tau":  ("#ff7f0e", "P"),   # Lagrangian dual
            "baseline":     ("#1f77b4", "*"),
        }
        # Draw previous-sweep series first (lighter, dashed, no labels)
        if pts_prev:
            prev_by_series: dict = {}
            for row in pts_prev:
                s = str(row["series"])
                if s == "baseline":
                    continue  # baselines unchanged — don't double-draw
                prev_by_series.setdefault(s, []).append((
                    float(row["effective_gb"]),
                    float(row["mean_kld"]),
                    str(row["label"]),
                ))
            for s, vals in prev_by_series.items():
                vals.sort()
                xs = [v[0] for v in vals]
                ys = [v[1] for v in vals]
                c, mk = colors.get(s, ("#888", "."))
                ax.plot(xs, ys, color=c, marker=mk, linestyle="--",
                        linewidth=0.9, markersize=5, alpha=0.4,
                        label=f"{s} (prev)")

        # Draw current series
        by_series: dict = {}
        for row in pts:
            s = str(row["series"])
            by_series.setdefault(s, []).append((
                float(row["effective_gb"]),
                float(row["mean_kld"]),
                str(row["label"]),
            ))
        for s, vals in by_series.items():
            vals.sort()
            xs = [v[0] for v in vals]
            ys = [v[1] for v in vals]
            c, m = colors.get(s, ("#888", "."))
            ms = 11 if s == "baseline" else 7
            ax.plot(xs, ys, color=c, marker=m, linestyle="-",
                     linewidth=1.4, markersize=ms, label=s)
            # Inline labels per point.
            label_color = {"baseline":     "#1f77b4",
                            "uniform":      "#444",
                            "dynamic_kbar": "#d62728",
                            "dynamic_eps":  "#2ca02c",
                            "dynamic_tau":  "#ff7f0e"}.get(s)
            if label_color is not None:
                for x, y, lbl in vals:
                    ax.annotate(lbl, (x, y), textcoords="offset points",
                                 xytext=(5, 4), fontsize=7, color=label_color)

        for lbl, x in size_only:
            ax.axvline(x, color="#777777", linestyle="--", linewidth=1.0, alpha=0.65)
            ax.annotate(lbl, xy=(x, 0.98), xycoords=("data", "axes fraction"),
                        xytext=(4, -2), textcoords="offset points",
                        rotation=90, va="top", ha="left", fontsize=7,
                        color="#666666")

        ax.set_xlabel("Effective model weight (GB, log)")
        ax.set_ylabel("Mean KLD vs BF16 ref (log)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(loc="lower left", fontsize=8)
        title_suffix = " + prev (dashed)" if pts_prev else ""
        ax.set_title(f"{args.title} — effective model weight vs KLD\n"
                     f"uniform · DP-kbar (mem budget) · baseline quants{title_suffix}")
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=160)
        print(f"\nwrote {args.out}", file=sys.stderr)
    except ImportError:
        print("matplotlib not available", file=sys.stderr)


if __name__ == "__main__":
    main()
