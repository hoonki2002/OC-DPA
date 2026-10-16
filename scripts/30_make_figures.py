"""Figures 1, 4 and 5 of the paper, drawn from the JSON files the pipeline writes.

  Fig. 1  decode latency (experts on flash vs on AIMC) and what sharing an ADC lane buys: TTFT, memory and logic die
          <- scripts/10_hardware.py
  Fig. 4  KL divergence against TTFT for the prefill budget k-bar = 2..8, OC-DPA vs uniform
          <- scripts/10_hardware.py (TTFT) and scripts/20_kl_sweep.py --noise "mixed,0.01,0.02,cal" --budgets 2,3,4,5,6,7
  Fig. 5  KL divergence against cell variation, experts only / with OC-DPA / with attention, router, head on AIMC
          <- scripts/20_kl_sweep.py --noise "mixed,s,2s,cal;..." (--dense attn,router,head for the third curve)

  python scripts/30_make_figures.py --hardware cache/hardware.json \
      --kl-task olmoe=cache/kl_olmoe_task.json,deepseek=cache/kl_deepseek_task.json \
      --kl-sigma olmoe=cache/sigma_olmoe.json,deepseek=cache/sigma_deepseek.json \
      --kl-sigma-dense olmoe=cache/sigma_olmoe_dense.json,deepseek=cache/sigma_deepseek_dense.json --out figs
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

_have = {f.name for f in fm.fontManager.ttflist}
FONT = next(n for n in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if n in _have)
plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "custom", "mathtext.rm": FONT,
                     "mathtext.it": f"{FONT}:italic", "mathtext.cal": FONT, "mathtext.sf": FONT, "mathtext.tt": FONT,
                     "mathtext.bf": f"{FONT}:bold", "mathtext.default": "it",
                     "pdf.fonttype": 42, "ps.fonttype": 42})              # TrueType in PDF (no Type 3 fonts)

NAMES = {"olmoe": "OLMoE-1B-7B", "deepseek": "DeepSeek-MoE-16B"}
OURS, UNI, REF, PRIOR = "#1F5FBF", "#E08A1E", "#4D4D4D", "#C62828"


def pairs(s):
    return {kv.split("=")[0]: kv.split("=", 1)[1] for kv in s.split(",")}


def kl_records(path):
    """{scenario: {prefill name: KL of the continuation}} from a 20_kl_sweep.py output."""
    out = {}
    for r in json.load(open(path)):
        out.setdefault(r["scenario"], {})[r["prefill"]] = r["kld_cont"]
    return out


def log_ticks(ax):
    """Readable 1-2-5 ticks on both log axes of a panel."""
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        cand = [m * 10 ** e for e in range(-1, 5) for m in (1, 2, 5)]
        ticks = [v for v in cand if lim[0] <= v <= lim[1]]
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1
def fig1(hw, out):
    olmoe = hw[NAMES["olmoe"]]
    npu_c, flash_c, aimc_c = "#D3E4F4", "#A8A8A8", "#2A9D8F"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.2, 2.15), dpi=200, gridspec_kw=dict(width_ratios=[0.8, 1.2]))
    bars = [("NPU+flash", olmoe["decode"]["flash"], flash_c), ("AIMC", olmoe["decode"]["aimc"], aimc_c)]
    for i, (lab, d, col) in enumerate(bars):
        npu, exp = d["npu_ms"], d["experts_ms"]
        ax.bar(i, npu, 0.56, color=npu_c, ec="black", lw=0.7, zorder=3)
        ax.bar(i, exp, 0.56, bottom=npu, color=col, ec="black", lw=0.7, zorder=3)
        ax.text(i, npu + exp + 7, f"{npu + exp:.0f} ms", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(i, npu / 2, f"{npu:.0f}", ha="center", va="center", fontsize=7, color="0.2", zorder=4)
        if exp > 50:
            ax.text(i, npu + exp / 2, f"{exp:.0f}", ha="center", va="center", fontsize=8, zorder=4)
        else:
            ax.text(i + 0.33, npu + exp / 2 + 3, f"{exp:.1f}", ha="left", va="center", fontsize=7, color=aimc_c)
    tot = [b[1]["npu_ms"] + b[1]["experts_ms"] for b in bars]
    ax.text(1, 72, f"{tot[0] / tot[1]:.1f}$\\times$ faster", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=aimc_c)
    ax.set_xticks([0, 1]); ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
    ax.set_xlim(-0.55, 1.6); ax.set_ylim(0, 300); ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("Decode latency per token (ms)", fontsize=8); ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(True, axis="y", color="0.88", lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title("Decode (OLMoE-1B-7B)", fontsize=9)
    ax.legend([Patch(fc=flash_c, ec="black", lw=0.6), Patch(fc=aimc_c, ec="black", lw=0.6), Patch(fc=npu_c, ec="black", lw=0.6)],
              ["experts on flash", "experts on AIMC", "NPU"], loc="upper right", frameon=False, fontsize=6.5,
              handlelength=1.1, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    # right: what sharing buys, OLMoE. Top: prefill TTFT. Bottom: the memory and the logic die; the larger one is the footprint.
    sv = [1, 2, 4, 8, 16]
    sh = olmoe["sharing"]
    ttft = [sh[f"s{v}_grouped"]["ttft_ms"] for v in sv]
    mem = [sh[f"s{v}_grouped"]["mem_mm2"] for v in sv]
    logic = [sh[f"s{v}_grouped"]["logic_mm2"] for v in sv]
    bal = min(range(len(sv)), key=lambda i: abs(mem[i] - logic[i]) / mem[i])          # dies about equal
    sub = bx.get_subplotspec().subgridspec(2, 1, hspace=0.12)
    fig.delaxes(bx)
    tx = fig.add_subplot(sub[0]); ax2 = fig.add_subplot(sub[1], sharex=tx)
    c_mem, c_log = "#34495E", "#E08A1E"
    tx.plot(sv, ttft, "-s", color="black", lw=1.6, ms=3.8, zorder=4)
    tx.set_yscale("log"); tx.set_ylim(150, 1700); tx.set_yticks([200, 500, 1000])
    tx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); tx.yaxis.set_minor_formatter(NullFormatter())
    tx.set_ylabel("Prefill TTFT (ms)", fontsize=7.5)
    tx.set_title("Effect of tier sharing (OLMoE-1B-7B)", fontsize=9)
    for i in (0, len(sv) - 1):
        tx.annotate(f"{ttft[i]:.0f}", (sv[i], ttft[i]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6.5)
    ax2.plot(sv, mem, "-o", color=c_mem, lw=1.5, ms=3.6, zorder=4)
    ax2.plot(sv, logic, "--^", color=c_log, lw=1.5, ms=3.8, mfc="white", mew=1.0, zorder=4)
    ax2.set_yscale("log"); ax2.set_ylim(22, 600); ax2.set_yticks([30, 100, 300])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_ylabel("Die area (mm$^2$)", fontsize=7.5)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_locator(FixedLocator(sv)); ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax2.set_xlim(0.85, 18.5); ax2.set_xlabel(r"Tiers sharing one ADC lane, $s$", fontsize=8)
    for a_ in (tx, ax2):
        a_.tick_params(labelsize=7); a_.grid(True, which="major", color="0.85", lw=0.6)
        a_.axvline(sv[bal], color="0.55", ls=":", lw=1.0, zorder=1)
    plt.setp(tx.get_xticklabels(), visible=False)
    ax2.text(sv[bal] * 1.1, 33, "dies about\nequal", fontsize=6.5, color="0.3", va="center")
    ax2.legend([Line2D([], [], color=c_mem, marker="o", ms=3.6, lw=1.5), Line2D([], [], color=c_log, ls="--", marker="^", ms=3.8, mfc="white", lw=1.5)],
               ["memory", "logic"], loc="lower left", frameon=False, fontsize=6.5, handlelength=2.0, handletextpad=0.4,
               labelspacing=0.25, borderaxespad=0.3)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.17, wspace=0.42)
    save(fig, out, "fig1_motivation")


# ---------------------------------------------------------------- Fig. 4
def fig4(hw, kl, out):
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.1), dpi=200)
    for ax, key in zip(axes, ("olmoe", "deepseek")):
        scen = kl[key]["mixed,0.01,0.02,cal"]
        b = hw[NAMES[key]]["budget"]
        pts = {"uniform": [], "DPA": []}
        for kb in range(2, 9):
            for name in pts:
                label = "all 8" if kb == 8 else f"{'uni' if name == 'uniform' else 'DPA'} {kb}"
                pts[name].append((b[f"{name}_{kb}"]["ttft_ms"], 1e3 * scen[label]))
        ax.plot(*zip(*pts["uniform"]), "-s", color=UNI, lw=1.5, ms=4, zorder=3)
        ax.plot(*zip(*pts["DPA"]), ":o", color=OURS, lw=1.6, ms=4.2, mfc="white", mew=1.0, zorder=5)
        ax.plot(*pts["DPA"][2], marker="*", ms=10, color=OURS, mec="white", mew=0.6, ls="none", zorder=6)
        ax.annotate(r"$\bar{k}$=2", pts["DPA"][0], textcoords="offset points", xytext=(6, -2), fontsize=7, color="0.25")
        ax.annotate(r"$\bar{k}$=8", pts["DPA"][-1], textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color="0.25")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.margins(x=0.08, y=0.12)
        log_ticks(ax)
        ax.set_title(NAMES[key], fontsize=9); ax.set_xlabel("TTFT (ms)", fontsize=8)
        ax.tick_params(labelsize=7.5); ax.grid(True, which="major", color="0.85", lw=0.6)
    axes[0].set_ylabel(r"KL divergence ($\mathregular{\times10^{-3}}$)", fontsize=8)
    h = [Line2D([], [], color=UNI, ls="-", marker="s", ms=4, lw=1.5),
         Line2D([], [], color=OURS, ls=":", marker="o", ms=4.2, mfc="white", lw=1.6),
         Line2D([], [], color=OURS, marker="*", ms=9, mec="white", mew=0.6, ls="none")]
    fig.legend(h, [r"AIMC + uniform $k$", "OC-DPA", r"operating point ($\bar{k}$=4)"], loc="upper center", ncol=3,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02), columnspacing=1.4, handlelength=2.2)
    fig.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.945))
    save(fig, out, "fig4_budget_sweep")


# ---------------------------------------------------------------- Fig. 5
def fig5(sig, dense, out):
    sigma = [0, 1, 2, 5, 10]

    def series(path, label):
        r = {}
        for scen, d in kl_records(path).items():
            r[round(float(scen.split(",")[1]) * 100)] = 1e3 * d[label]
        return [r[s] for s in sigma]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.1), dpi=200)
    for ax, key in zip(axes, ("olmoe", "deepseek")):
        exp, dpa = series(sig[key], "all 8"), series(sig[key], "DPA 4")
        den = series(dense[key], "all 8")
        ax.axvline(1, color="0.45", ls=(0, (3, 2)), lw=0.9, zorder=1)
        ax.text(1.12, 0.97, "task setting", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.5, color="0.35")
        ax.plot(sigma, den, "--^", color=PRIOR, lw=1.4, ms=4, zorder=3)
        ax.plot(sigma, exp, "-s", color=REF, lw=1.5, ms=4, zorder=4)
        ax.plot(sigma, dpa, ":o", color=OURS, lw=1.6, ms=4.2, mfc="white", mew=1.0, zorder=5)
        ax.set_yscale("log"); ax.set_xlim(-0.6, 10.6); ax.set_xticks(sigma); ax.margins(y=0.12)
        for axis, lim in ((ax.yaxis, ax.get_ylim()),):
            ticks = [m * 10 ** e for e in range(0, 4) for m in (1, 2, 5) if lim[0] <= m * 10 ** e <= lim[1]]
            axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(NAMES[key], fontsize=9); ax.set_xlabel(r"Cell variation of 1T1R, $\sigma$ (%)", fontsize=8)
        ax.tick_params(labelsize=7.5); ax.grid(True, which="major", color="0.85", lw=0.6)
    axes[0].set_ylabel(r"KL divergence ($\mathregular{\times10^{-3}}$)", fontsize=8)
    h = [Line2D([], [], color=REF, ls="-", marker="s", ms=4, lw=1.5),
         Line2D([], [], color=OURS, ls=":", marker="o", ms=4.2, mfc="white", lw=1.6),
         Line2D([], [], color=PRIOR, ls="--", marker="^", ms=4, lw=1.4)]
    fig.legend(h, ["experts on AIMC", r"+ OC-DPA $\bar{k}$=4", "experts + attention, router, head"], loc="upper center",
               ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02), columnspacing=1.6, handlelength=2.4)
    fig.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.945))
    save(fig, out, "fig5_cell_variation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default="cache/hardware.json")
    ap.add_argument("--kl-task", help="olmoe=PATH,deepseek=PATH  (task-setting sweep over the prefill budget)")
    ap.add_argument("--kl-sigma", help="olmoe=PATH,deepseek=PATH  (sigma sweep, experts on AIMC)")
    ap.add_argument("--kl-sigma-dense", help="olmoe=PATH,deepseek=PATH  (sigma sweep, + attention/router/head on AIMC)")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--only", default="1,4,5")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hw = json.load(open(a.hardware))
    todo = a.only.split(",")
    if "1" in todo:
        fig1(hw, out)
    if "4" in todo:
        fig4(hw, {k: kl_records(v) for k, v in pairs(a.kl_task).items()}, out)
    if "5" in todo:
        fig5(pairs(a.kl_sigma), pairs(a.kl_sigma_dense), out)
