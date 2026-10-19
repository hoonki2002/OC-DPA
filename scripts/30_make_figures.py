"""Figures 1, 4 and 5 of the paper, drawn from the JSON files the pipeline writes.

  Fig. 1  decode latency (experts on flash vs on AIMC) and the area-TTFT design space of the converter sharing
          (tiers per lane x columns per ADC), all planes read vs OC-DPA   <- scripts/10_hardware.py
  Fig. 4  KL divergence against TTFT for the prefill budget k-bar = 2..8, OC-DPA vs uniform
          <- scripts/10_hardware.py (TTFT) and scripts/21_kl_sweep.py --noise "mixed,0.01,0.02,cal" --budgets 2,3,4,5,6,7
  Fig. 5  KL divergence against cell variation, experts only / with OC-DPA / with attention, router, head on AIMC
          <- scripts/21_kl_sweep.py --noise "mixed,s,2s,cal;..." (--dense attn,router,head for the third curve)

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
                     "mathtext.bf": f"{FONT}:bold", "mathtext.bfit": f"{FONT}:italic:bold", "mathtext.default": "it",
                     "pdf.fonttype": 42, "ps.fonttype": 42})              # TrueType in PDF (no Type 3 fonts)

NAMES = {"olmoe": "OLMoE-1B-7B", "deepseek": "DeepSeek-MoE-16B"}
OURS, UNI, REF, PRIOR = "#1F5FBF", "#E08A1E", "#4D4D4D", "#C62828"


def pairs(s):
    return {kv.split("=")[0]: kv.split("=", 1)[1] for kv in s.split(",")}


def kl_records(path):
    """{scenario: {prefill name: KL of the continuation}} from a 21_kl_sweep.py output."""
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
def fig1(hw, out, right="sharing"):
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
    if right == "stacks":
        # right: the same 16 memory tiers with the ADC lanes shared by s = 1, 4 and 16 tiers (side view; die widths are
        # proportional to the edge of the die, sqrt(area)). Each vertical line is one set of ADC lanes, rising through
        # the tiers below it to the tier it serves.
        sh = olmoe["sharing"]
        fig.delaxes(bx)
        gx = fig.add_subplot(bx.get_subplotspec())
        gx.set_xlim(0, 6.5); gx.set_ylim(-0.78, 1.22); gx.axis("off")
        K = 21.0; T = 16; th, gap = 0.046, 0.010
        c_tier, c_logic, c_line = ["#9DB0C4", "#7B93AB"], "#E08A1E", "#1B2A3A"
        for col, sv in enumerate((1, 4, 16)):
            r = sh[f"s{sv}_grouped"]
            cx = 1.35 + 2.05 * col
            wm, wl = (r["mem_mm2"] ** 0.5) / K, (r["logic_mm2"] ** 0.5) / K
            gx.add_patch(plt.Rectangle((cx - wl / 2, 0.0), wl, 0.10, fc=c_logic, ec="black", lw=0.6, zorder=2))
            n = T // sv
            y_bot = 0.16
            for k in range(T):                       # tier k = 0 at the bottom
                gx.add_patch(plt.Rectangle((cx - wm / 2, y_bot + k * (th + gap)), wm, th, fc=c_tier[(k // sv) % 2],
                                           ec="black", lw=0.4, zorder=2))
            for g in range(n):
                xg = cx + (g - (n - 1) / 2) * (wm * 0.8 / (n - 1)) if n > 1 else cx
                y_top = y_bot + ((g + 1) * sv - 1) * (th + gap) + th / 2
                gx.plot([xg, xg], [0.10, y_top], color=c_line, lw=0.8 if n > 4 else 1.7, zorder=3, solid_capstyle="butt")
            top = y_bot + T * (th + gap)
            gx.text(cx, top + 0.06, f"$s$ = {sv}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            gx.text(cx, -0.08, f"{r['footprint_mm2']:,.0f} mm$^2$", ha="center", va="top", fontsize=8.5, fontweight="bold")
            gx.text(cx, -0.27, f"TTFT {r['ttft_ms']:.0f} ms", ha="center", va="top", fontsize=8, color="0.25")
        # bracket: the 16 tiers are the same in every stack
        yb0, yb1 = 0.16, 0.16 + T * (th + gap) - gap
        gx.annotate("", xy=(0.28, yb0), xytext=(0.28, yb1), arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.8))
        gx.text(0.13, (yb0 + yb1) / 2, "16 tiers", rotation=90, ha="center", va="center", fontsize=7.5, color="0.25")
        h = [Patch(fc=c_tier[0], ec="black", lw=0.5), Patch(fc=c_logic, ec="black", lw=0.5), Line2D([], [], color=c_line, lw=1.6)]
        gx.legend(h, ["memory tier", "logic die (ADC lanes)", "one set of ADC lanes"], loc="lower center", ncol=3, frameon=False,
                  fontsize=6.3, handlelength=1.3, handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.52, -0.04))
        gx.set_title("Sharing ADC lanes across tiers", fontsize=9)
    elif right == "sharing":
        # right: why the tiers share ADC lanes, and what it costs (OLMoE, 16 tiers, columns per ADC of the best baseline, all planes read).
        # Bottom: the memory die (cells, fixed by the 16 tiers) and the logic die (ADC lanes + NPU); the larger one is
        # the footprint. Top: the prefill TTFT of a 1,024-token prompt.
        sv = [1, 2, 4, 8, 16]
        sh = olmoe["sharing"]
        ttft = [sh[f"s{v}_grouped"]["ttft_ms"] for v in sv]
        mem = [sh[f"s{v}_grouped"]["mem_mm2"] for v in sv]
        logic = [sh[f"s{v}_grouped"]["logic_mm2"] for v in sv]
        foot = [sh[f"s{v}_grouped"]["footprint_mm2"] for v in sv]
        sub = bx.get_subplotspec().subgridspec(2, 1, hspace=0.12)
        fig.delaxes(bx)
        tx = fig.add_subplot(sub[0]); ax2 = fig.add_subplot(sub[1], sharex=tx)
        c_mem, c_log = "#34495E", "#E08A1E"
        tx.plot(sv, ttft, "-s", color="black", lw=1.6, ms=3.8, zorder=4)
        tx.set_yscale("log"); tx.set_ylim(150, 1500); tx.set_yticks([200, 500, 1000])
        tx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); tx.yaxis.set_minor_formatter(NullFormatter())
        tx.set_ylabel("Prefill TTFT (ms)", fontsize=7.5)
        tx.set_title("Sharing ADC lanes across tiers (OLMoE-1B-7B)", fontsize=9)
        for i, dy in ((0, 5), (len(sv) - 1, -11)):
            tx.annotate(f"{ttft[i]:.0f}", (sv[i], ttft[i]), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=6.5)
        tx.annotate(f"{ttft[-1] / ttft[0]:.1f}$\\times$ longer", (4, 420), fontsize=7, color="0.25", ha="center")
        ax2.fill_between(sv, foot, [min(mem[i], logic[i]) for i in range(len(sv))], color="0.88", lw=0, zorder=1)
        ax2.plot(sv, mem, "-o", color=c_mem, lw=1.5, ms=3.6, zorder=4)
        ax2.plot(sv, logic, "--^", color=c_log, lw=1.5, ms=3.8, mfc="white", mew=1.0, zorder=4)
        ax2.set_yscale("log"); ax2.set_ylim(40, 2600); ax2.set_yticks([100, 300, 1000])
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax2.yaxis.set_minor_formatter(NullFormatter())
        ax2.set_ylabel("Die area (mm$^2$)", fontsize=7.5)
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_locator(FixedLocator(sv)); ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax2.set_xlim(0.85, 18.5); ax2.set_xlabel(r"Tiers sharing one ADC lane, $s$ (16 tiers)", fontsize=8)
        ax2.annotate(f"footprint {foot[0]:,.0f} $\\rightarrow$ {foot[-1]:.0f} mm$^2$ ({foot[0] / foot[-1]:.1f}$\\times$)", (9.5, 1350), fontsize=6.8,
                     color="0.25", ha="center")
        for a_ in (tx, ax2):
            a_.tick_params(labelsize=7); a_.grid(True, which="major", color="0.85", lw=0.6)
        plt.setp(tx.get_xticklabels(), visible=False)
        ax2.legend([Line2D([], [], color=c_mem, marker="o", ms=3.6, lw=1.5), Line2D([], [], color=c_log, ls="--", marker="^", ms=3.8, mfc="white", lw=1.5)],
                   ["memory die", "logic die (ADC + NPU)"], loc="lower left", frameon=False, fontsize=6.3, handlelength=2.0,
                   handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3)
    elif right == "prompt":
        # right: what bounds the prefill at the best sharing as the prompt grows (OLMoE, all planes read unless noted).
        # The converter lanes grow with the prompt and overtake the NPU and DRAM at a few dozen tokens; OC-DPA halves
        # them by reading half the planes.
        pr = olmoe["prompt"]
        xs = [r["tokens"] for r in pr]
        npu_t = [r["npu_ms"] for r in pr]; lane = [r["lanes_ms"] for r in pr]; lane_o = [r["lanes_ocdpa_ms"] for r in pr]
        cross = next(i for i in range(len(xs)) if lane[i] > npu_t[i])
        x0 = (xs[cross - 1] * (npu_t[cross - 1] / lane[cross - 1]) ** 0 ) if cross else xs[0]
        # geometric interpolation of the crossing point of the two lines
        import math
        f = lambda i: math.log(lane[i] / npu_t[i])
        t = -f(cross - 1) / (f(cross) - f(cross - 1))
        xc = math.exp(math.log(xs[cross - 1]) + t * (math.log(xs[cross]) - math.log(xs[cross - 1])))
        bx.axvspan(xc, 5200, color="#F6E7D4", alpha=0.55, lw=0, zorder=0)
        bx.plot(xs, npu_t, "-o", color="#2A9D8F", lw=1.5, ms=3.2, zorder=4)
        bx.plot(xs, lane, "-s", color=REF, lw=1.6, ms=3.4, zorder=5)
        bx.plot(xs, lane_o, ":o", color=OURS, lw=1.7, ms=3.6, mfc="white", mew=1.0, zorder=6)
        bx.set_xscale("log", base=2); bx.set_yscale("log")
        bx.set_xticks([16, 64, 256, 1024, 4096]); bx.set_yticks([10, 100, 1000])
        for ax_ in (bx.xaxis, bx.yaxis):
            ax_.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax_.set_minor_formatter(NullFormatter())
        bx.set_xlim(13, 5200); bx.set_ylim(5, 3500)
        bx.axvline(xc, color="0.45", ls=":", lw=1.0, zorder=3)
        bx.text(xc * 0.9, 3000, "NPU\nbound", fontsize=6.5, color="0.3", ha="right", va="top")
        bx.text(xc * 1.12, 3000, "converter-lane bound", fontsize=6.5, color="0.3", ha="left", va="top")
        i = xs.index(1024)
        bx.annotate(f"lanes: {100 * lane[i] / (lane[i] + npu_t[i]):.0f}% of TTFT\nat 1,024 tokens", (1024, lane[i]), xytext=(95, 700),
                    textcoords="data", ha="left", va="center", fontsize=6.3, color="0.25",
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7, shrinkA=1, shrinkB=2))
        bx.annotate(f"{lane[i] / lane_o[i]:.1f}$\\times$", (1024, lane_o[i]), textcoords="offset points", xytext=(5, -10),
                    ha="left", fontsize=6.8, color=OURS, fontweight="bold")
        bx.set_xlabel("Prompt length (tokens)", fontsize=8); bx.set_ylabel("Prefill time (ms)", fontsize=8)
        bx.tick_params(labelsize=7); bx.grid(True, which="major", color="0.85", lw=0.6)
        bx.set_title("Prefill bound (OLMoE-1B-7B)", fontsize=9)
        bx.legend([Line2D([], [], color="#2A9D8F", marker="o", ms=3.2, lw=1.5), Line2D([], [], color=REF, marker="s", ms=3.4, lw=1.6),
                   Line2D([], [], color=OURS, ls=":", marker="o", ms=3.6, mfc="white", lw=1.7)],
                  ["NPU + DRAM", "converter lanes", "lanes, OC-DPA"], loc="lower right", frameon=False, fontsize=6.5,
                  handlelength=2.0, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3)
    else:
        # right: the design space of the converter sharing, OLMoE. Every point is a design with its tiers per ADC lane,
        # columns per ADC (and nothing else changed); grey: all planes read, blue: OC-DPA. Lines are the Pareto
        # fronts (least TTFT for a footprint); stars mark the best throughput per area of each; the red cross is 3D-AIMC.
        def front(pts):
            keep, best_t = [], float("inf")
            for f, t in sorted(pts):
                if t < best_t:
                    keep.append((f, t)); best_t = t
            return keep
        grid = olmoe["grid"]
        base = [(r["footprint_mm2"], r["ttft_ms"]) for r in grid if r["design"] == "all planes"]
        ours = [(r["footprint_mm2"], r["ttft_ms"]) for r in grid if r["design"] == "OC-DPA"]
        bx.scatter(*zip(*base), s=7, color=REF, alpha=0.45, lw=0, zorder=2)
        bx.scatter(*zip(*ours), s=7, color=OURS, alpha=0.45, lw=0, zorder=3)
        bx.plot(*zip(*front(base)), "-", color=REF, lw=1.4, zorder=4)
        bx.plot(*zip(*front(ours)), "-", color=OURS, lw=1.6, zorder=5)
        d = olmoe["designs"]
        for key, col, lab, dx, dy in (("Best baseline", REF, "best\nbaseline", 6, 3), ("OC-DPA", OURS, "OC-DPA", -8, 6)):
            x, y = d[key]["footprint_mm2"], d[key]["prefill"]["ttft_ms"]
            bx.plot(x, y, marker="*", ms=9, color=col, mec="white", mew=0.5, ls="none", zorder=7)
            bx.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=6.5, color=col,
                        ha="left" if dx > 0 else "right", va="bottom")
        x, y = d["3D-AIMC"]["footprint_mm2"], d["3D-AIMC"]["prefill"]["ttft_ms"]
        bx.plot(x, y, marker="X", ms=5.5, color=PRIOR, ls="none", zorder=7)
        bx.annotate("3D-AIMC", (x, y), textcoords="offset points", xytext=(5, 2), fontsize=6.5, color=PRIOR)
        bx.set_xscale("log"); bx.set_yscale("log")
        bx.set_xticks([100, 200, 500, 1000, 2000]); bx.set_yticks([50, 100, 200, 500, 1000])
        for ax_ in (bx.xaxis, bx.yaxis):
            ax_.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax_.set_minor_formatter(NullFormatter())
        bx.set_xlim(95, 2500); bx.set_ylim(45, 1400)
        bx.set_xlabel("Footprint (mm$^2$)", fontsize=8); bx.set_ylabel("Prefill TTFT (ms)", fontsize=8)
        bx.tick_params(labelsize=7); bx.grid(True, which="major", color="0.85", lw=0.6)
        bx.set_title("Sharing design space (OLMoE-1B-7B)", fontsize=9)
        bx.legend([Line2D([], [], color=REF, marker="o", ms=3, lw=1.4, alpha=0.8), Line2D([], [], color=OURS, marker="o", ms=3, lw=1.6, alpha=0.8)],
                  ["all planes", "OC-DPA"], loc="upper right", frameon=False, fontsize=6.5, handlelength=1.8, handletextpad=0.4,
                  labelspacing=0.25, borderaxespad=0.3)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.17, wspace=0.34)
    save(fig, out, "fig1_motivation" if right == "sharing" else f"fig1_{right}")


# ---------------------------------------------------------------- Fig. 1 (definition + sweep)
C_MEM, C_LOG = "#34495E", "#E08A1E"           # memory die / logic die, the same colours in the stack and in the plot


STAR = False                        # mark OC-DPA on the TTFT panel (--fig1-star)


def fig1_sweep(hw, out):
    """Left: one stack, T = 16 memory tiers on a logic die; s = 4 tiers share each ADC lane (alternating shades mark the
    groups of tiers that share a lane). Right: the sweep of s at the same 16 tiers: the prefill TTFT (top) and the memory
    die and logic die (bottom, same colours as the stack); the larger die is the footprint (grey band)."""
    sh = hw[NAMES["olmoe"]]["sharing"]
    sharing_mux = hw[NAMES["olmoe"]].get("sharing_mux", 8)
    sv = [1, 2, 4, 8, 16]
    ttft = [sh[f"s{v}_grouped"]["ttft_ms"] for v in sv]
    mem = [sh[f"s{v}_grouped"]["mem_mm2"] for v in sv]
    logic = [sh[f"s{v}_grouped"]["logic_mm2"] for v in sv]
    foot = [sh[f"s{v}_grouped"]["footprint_mm2"] for v in sv]
    fig = plt.figure(figsize=(5.2, 2.45), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.2], height_ratios=[1, 1], hspace=0.16, wspace=0.26,
                          left=0.02, right=0.985, top=0.92, bottom=0.14)
    gx = fig.add_subplot(gs[:, 0]); tx = fig.add_subplot(gs[1, 1]); ax2 = fig.add_subplot(gs[0, 1], sharex=tx)   # area on top, TTFT below
    # ---- left: the stack, from the top of the right panels down to the bottom of the figure
    pos = gx.get_position(); gx.set_position([pos.x0 - 0.005, 0.015, pos.width + 0.005, pos.y1 - 0.015])
    T, S0 = 16, 4
    shade = {0: "#8EA2B6", 1: C_MEM}
    gx.set_xlim(0.15, 4.55); gx.set_ylim(0.0, 0.99); gx.axis("off")
    y_log, th_logic = 0.085, 0.07                              # logic die; its label sits below it, at the bottom of the right panels
    y_bot = y_log + th_logic + 0.045
    top_lim = 0.9                                             # top of the stack; its label sits above
    th = (top_lim - y_bot) / T * 0.84; gap = (top_lim - y_bot) / T * 0.16
    cx, wm, wl = 2.35, 2.5, 3.4
    gx.add_patch(plt.Rectangle((cx - wl / 2, y_log), wl, th_logic, fc=C_LOG, ec="black", lw=0.6, zorder=2))
    for k in range(T):
        gx.add_patch(plt.Rectangle((cx - wm / 2, y_bot + k * (th + gap)), wm, th, fc=shade[(k // S0) % 2], ec="black", lw=0.4, zorder=2))
    n = T // S0
    for g in range(n):
        xg = cx + (g - (n - 1) / 2) * 0.58
        y_top = y_bot + ((g + 1) * S0 - 1) * (th + gap) + th / 2
        gx.plot([xg, xg], [y_log + th_logic, y_top], color="black", lw=1.7, zorder=3, solid_capstyle="butt")
    top = y_bot + T * (th + gap) - gap
    gx.annotate("", xy=(cx - wm / 2 - 0.3, y_bot), xytext=(cx - wm / 2 - 0.3, top), arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.9))
    gx.text(cx - wm / 2 - 0.45, (y_bot + top) / 2, "$T$ = 16 tiers", rotation=90, ha="right", va="center", fontsize=8.5, fontweight="bold")
    g0 = n - 1
    yg0, yg1 = y_bot + g0 * S0 * (th + gap), y_bot + (g0 + 1) * S0 * (th + gap) - gap
    gx.annotate("", xy=(cx + wm / 2 + 0.18, yg0), xytext=(cx + wm / 2 + 0.18, yg1), arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.1))
    gx.text(cx + wm / 2 + 0.3, (yg0 + yg1) / 2, "$s$ = 4", ha="left", va="center", fontsize=9, fontweight="bold", color="0.15")
    gx.text(cx, top + 0.03, "memory die", ha="center", va="bottom", fontsize=8.5, color=C_MEM, fontweight="bold")
    gx.text(cx, y_log - 0.02, "logic die", ha="center", va="top", fontsize=8.5, color="#B8680F", fontweight="bold")
    gx.set_title("Stack of $T$ memory tiers", fontsize=9, y=0.995)
    # ---- right: the sweep
    tx.plot(sv, ttft, "-s", color=PRIOR, lw=1.7, ms=4.0, zorder=4)
    oc = hw[NAMES["olmoe"]]["designs"]["OC-DPA"]; oc_s = hw[NAMES["olmoe"]]["configs"]["OC-DPA"]["share"]
    if STAR:
        tx.plot(oc_s, oc["prefill"]["ttft_ms"], marker="*", ms=11, color=OURS, mec="white", mew=0.6, ls="none", zorder=6)
        tx.annotate(f"OC-DPA\n{oc['prefill']['ttft_ms']:.0f}", (oc_s, oc["prefill"]["ttft_ms"]), textcoords="offset points", xytext=(0, -9),
                    ha="center", va="top", fontsize=6.8, color=OURS, fontweight="bold")
        tx.plot([oc_s, oc_s], [oc["prefill"]["ttft_ms"] * 1.06, ttft[sv.index(oc_s)] * 0.97], ":", color=OURS, lw=1.0, zorder=3)
    tx.set_yscale("log"); tx.set_ylim(min(ttft) * 0.72, max(ttft) * 1.55); tx.set_yticks([t for t in (100, 200, 500, 1000) if tx.get_ylim()[0] < t < tx.get_ylim()[1]])
    tx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); tx.yaxis.set_minor_formatter(NullFormatter())
    tx.set_ylabel("Prefill TTFT (ms)", fontsize=7.5)
    ax2.set_title("Sharing $s$ tiers per lane ($T$ = 16)", fontsize=9, y=0.995)
    for i, dy in ((0, 5), (len(sv) - 1, -11)):
        tx.annotate(f"{ttft[i]:.0f}", (sv[i], ttft[i]), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=6.5)
    tx.annotate(f"{ttft[-1] / ttft[0]:.1f}$\\times$ longer", (2.7, ttft[2] * 1.75), fontsize=7, color=PRIOR, ha="center", fontweight="bold")
    ax2.fill_between(sv, foot, [min(mem[i], logic[i]) for i in range(len(sv))], color="0.88", lw=0, zorder=1)
    ax2.plot(sv, mem, "-o", color=C_MEM, lw=1.5, ms=3.6, zorder=4)
    ax2.plot(sv, logic, "--^", color=C_LOG, lw=1.5, ms=3.8, mfc="white", mew=1.0, zorder=4)
    ax2.set_yscale("log"); ax2.set_ylim(min(mem) * 0.55, max(foot) * 1.45); ax2.set_yticks([t for t in (100, 300, 1000, 3000) if ax2.get_ylim()[0] < t < ax2.get_ylim()[1]])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_ylabel("Die area (mm$^2$)", fontsize=7.5)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_locator(FixedLocator(sv)); ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax2.set_xlim(0.85, 18.5); tx.set_xlabel(r"Tiers sharing one ADC lane, $s$", fontsize=8)
    for a_ in (tx, ax2):
        a_.tick_params(labelsize=7); a_.grid(True, which="major", color="0.85", lw=0.6)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.legend([Line2D([], [], color=C_MEM, marker="o", ms=3.6, lw=1.5), Line2D([], [], color=C_LOG, ls="--", marker="^", ms=3.8, mfc="white", lw=1.5)],
               ["memory die", "logic die"], loc="lower left", frameon=False, fontsize=6.3, handlelength=2.0,
               handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3)
    save(fig, out, "fig1_motivation")


# ---------------------------------------------------------------- Fig. 2 (architecture)
def fig2_arch(out):
    """Block diagram of the accelerator, in the colours of Fig. 1: memory die slate, logic die orange, OC-DPA blue."""
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
    fig = plt.figure(figsize=(7.0, 3.66), dpi=220)
    ax = fig.add_axes([0.004, 0.004, 0.992, 0.992]); ax.set_xlim(0, 106); ax.set_ylim(-3.0, 49.3); ax.axis("off")
    MEM_BG, LOG_BG, BLUE_BG, GREEN, TSV = "#E4E9EF", "#FCEBD3", "#CFE0F7", "#2E8B57", "#3A3A3A"
    TXT = dict(ha="center", va="center", color="black")

    def box(x0, y0, x1, y1, fc, ec, lw=1.4, r=1.0, z=2):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

    def arrow(xy0, xy1, color, lw=1.7, style="-|>", ms=10, z=5):
        ax.annotate("", xy=xy1, xytext=xy0, arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0), zorder=z)

    # geometry (y): logic tier 1.2-19.6, TSV gap, memory tiers 27.6-44.6
    LG0, LG1, MM0, MM1 = 1.2, 19.6, 27.6, 44.6

    # ---- left: the stack of Fig. 1 in two dimensions, all 16 tiers sharing one ADC lane (s = 16)
    T, y_logic, th_logic = 16, 16.2, 2.4
    y_bot = y_logic + th_logic + 1.1
    th, gap = 1.22, 0.24
    DXS = 2.4                                                  # the stack sits this far to the right
    X0, X1, XL0, XL1 = 5.0 + DXS, 19.0 + DXS, 3.2 + DXS, 20.8 + DXS
    cx = (X0 + X1) / 2
    ax.add_patch(Rectangle((XL0, y_logic), XL1 - XL0, th_logic, fc=C_LOG, ec="black", lw=0.8, zorder=3))
    for k in range(T):
        ax.add_patch(Rectangle((X0, y_bot + k * (th + gap)), X1 - X0, th, fc=C_MEM, ec="black", lw=0.5, zorder=3))
    top = y_bot + T * (th + gap) - gap
    ax.plot([cx, cx], [y_logic + th_logic, top - th / 2], color="black", lw=2.0, zorder=4, solid_capstyle="butt")   # the shared ADC lane
    ax.annotate("", xy=(2.0 + DXS, y_bot), xytext=(2.0 + DXS, top), arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.2, mutation_scale=9), zorder=4)
    ax.text(0.45 + DXS, (y_bot + top) / 2, "$\\mathbfit{T}$ = 16 tiers", rotation=90, fontsize=10.5, fontweight="bold", **TXT)
    ax.text(cx, top + 2.4, "$\\mathbfit{s}$ = 16", fontsize=10.5, fontweight="bold", **TXT)
    box(1.5, 1.4, 21.5, 13.4, "#EFEFEF", "#555555", lw=1.3)
    ax.text(11.5, 10.0, "LPDDR4", fontsize=11, fontweight="bold", **TXT)
    ax.text(11.5, 5.9, "[non-expert weights]\n[KV cache]", fontsize=7.6, linespacing=1.3, **TXT)

    # ---- right: memory tiers
    box(26.5, MM0, 99.0, MM1, MEM_BG, C_MEM, lw=1.7)
    ax.text(98.5, MM1 + 2.0, "Memory Tiers", fontsize=11.5, fontweight="bold", color=C_MEM, ha="right", va="center")
    by0, by1 = MM0 + 7.0, MM0 + 15.2
    blocks = ((28.5, 52.0, "Planes 1, 2 (base)", "1T1R [128 $\\times$ 128]"), (54.0, 82.0, "Planes 3--8", "Crossbar [256 $\\times$ 256]"),
              (84.0, 97.8, "$\\alpha,\\beta$ (8-bit)", "digital cells"))
    for x0, x1, t1, t2 in blocks:
        box(x0, by0, x1, by1, "white", C_MEM)
        ax.text((x0 + x1) / 2, by0 + 5.8, t1, fontsize=9.4 if x0 > 80 else 10.2, fontweight="bold", **TXT)
        ax.text((x0 + x1) / 2, by0 + 2.5, t2, fontsize=8.8 if x0 > 80 else 9.6, **TXT)
    for xc in (40.25, 68.0):
        ax.add_patch(Polygon([(xc - 8.8, MM0 + 6.1), (xc + 8.8, MM0 + 6.1), (xc + 6.2, MM0 + 2.0), (xc - 6.2, MM0 + 2.0)], fc="white", ec=C_MEM, lw=1.4, zorder=3))
        ax.text(xc, MM0 + 4.1, "MUX ($m$:1)", fontsize=9.6, **TXT)
        ax.plot([xc, xc], [MM0 + 6.1, by0], color=C_MEM, lw=1.4, zorder=3)

    # ---- TSVs: bidirectional pillars between the two tiers
    for xc in (33.0, 55.0, 70.0):
        arrow((xc, LG1 + 1.4), (xc, MM0 - 0.8), TSV, lw=3.0, style="<|-|>", ms=16, z=6)
    ax.text(38.0, (LG1 + MM0) / 2 + 0.2, "TSVs", fontsize=11.5, fontweight="bold", color=TSV, ha="left", va="center")

    # ---- logic tier (larger)
    box(26.5, LG0, 99.0, LG1, LOG_BG, C_LOG, lw=1.7)
    ax.text(98.5, LG1 + 1.8, "Logic Tier", fontsize=11.5, fontweight="bold", color="#B8680F", ha="right", va="center")
    ib0, ib1 = 5.4, 16.0                                             # inner blocks (original size, centred in the logic tier)
    box(28.5, ib0, 36.5, ib1, "#EFEFEF", "#555555", lw=1.3); ax.text(32.5, (ib0 + ib1) / 2, "NPU", fontsize=11, fontweight="bold", **TXT)
    for x0, x1 in ((39.5, 53.5), (56.0, 70.0)):
        box(x0, 7.4, x1, ib1, "white", C_LOG)
        ax.text((x0 + x1) / 2, 13.7, "SAR ADC", fontsize=9.8, fontweight="bold", **TXT)
        ax.text((x0 + x1) / 2, 11.5, "(8 bit)", fontsize=9.2, **TXT)
        ax.text((x0 + x1) / 2, 9.3, "per TSV", fontsize=9.2, **TXT)
    box(72.0, ib0, 82.0, ib1, "white", C_LOG)
    ax.text(77.0, 12.5, "Recom", fontsize=10, fontweight="bold", **TXT); ax.text(77.0, 9.2, "($\\alpha,\\beta$)", fontsize=9.8, **TXT)
    box(84.5, ib0, 97.8, ib1, BLUE_BG, OURS, lw=1.8)
    ax.text(91.15, 12.6, "Precision", fontsize=10, fontweight="bold", **TXT); ax.text(91.15, 9.1, "Allocator", fontsize=10, fontweight="bold", **TXT)

    # ---- data paths (green): ADC outputs -> Recom -> NPU
    for xc in (46.5, 63.0):
        ax.plot([xc, xc], [7.4, 2.9], color=GREEN, lw=1.9, zorder=4)
    ax.plot([46.5, 77.0], [2.9, 2.9], color=GREEN, lw=1.9, zorder=4)
    arrow((77.0, 2.9), (77.0, ib0), GREEN, lw=1.9, ms=17)
    ax.plot([77.0, 77.0], [ib1, 18.8], color=GREEN, lw=1.9, zorder=4); ax.plot([77.0, 32.5], [18.8, 18.8], color=GREEN, lw=1.9, zorder=4)
    arrow((32.5, 18.8), (32.5, ib1), GREEN, lw=1.9, ms=17)
    arrow((21.5, 9.0), (28.5, 9.0), "black", lw=1.8, ms=15)                              # DRAM -> NPU
    # ---- control (blue): router logits NPU -> allocator; plane enable allocator -> memory tiers
    ax.plot([36.5, 38.0], [7.2, 7.2], color=OURS, lw=1.7, zorder=4); ax.plot([38.0, 38.0], [7.2, -0.4], color=OURS, lw=1.7, zorder=4)
    ax.plot([38.0, 91.15], [-0.4, -0.4], color=OURS, lw=1.7, zorder=4)
    arrow((91.15, -0.4), (91.15, ib0), OURS, lw=1.7, ms=15)
    ax.text(64.0, -1.0, "router logits", fontsize=10, fontweight="bold", style="italic", color=OURS, ha="center", va="top")
    ax.plot([97.8, 101.8], [10.7, 10.7], color=OURS, lw=1.7, zorder=4); ax.plot([101.8, 101.8], [10.7, MM0 + 3.8], color=OURS, lw=1.7, zorder=4)
    arrow((101.8, MM0 + 3.8), (99.0, MM0 + 3.8), OURS, lw=1.7, ms=15)
    ax.text(104.5, 21.0, "plane enable", rotation=90, fontsize=10, fontweight="bold", style="italic", color=OURS, ha="center", va="center")
    for t in ax.texts:                                         # larger type: the figure is printed at about half this width
        t.set_fontsize(t.get_fontsize() * 1.18)
    save(fig, out, "fig2_arch")


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


# ---------------------------------------------------------------- Fig. 4 (merged): budget sweep and cell variation, OLMoE
def fig4_merged(hw, kl, sig, dense, out):
    """(a) KL divergence against TTFT for the prefill budget k-bar = 2..8, OC-DPA vs uniform; (b) KL divergence against
    cell variation: experts on AIMC, with OC-DPA at k-bar = 4, and with attention, router and head on AIMC too."""
    sigma = [1, 2, 5]               # the 0 % point is too close to 1 % to read, the 10 % point (up to 581) compresses the rest

    def series(path, label):
        r = {}
        for scen, d in kl_records(path).items():
            r[round(float(scen.split(",")[1]) * 100)] = 1e3 * d[label]
        return [r[s] for s in sigma]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.2, 2.4), dpi=200)
    scen = kl["mixed,0.01,0.02,cal"]
    b = hw[NAMES["olmoe"]]["budget"]
    pts = {"uniform": [], "DPA": []}
    for kb in range(2, 9):
        for name in pts:
            label = "all 8" if kb == 8 else f"{'uni' if name == 'uniform' else 'DPA'} {kb}"
            pts[name].append((b[f"{name}_{kb}"]["ttft_ms"], 1e3 * scen[label]))
    ax.plot(*zip(*pts["uniform"]), "-s", color=UNI, lw=1.5, ms=4, zorder=3)
    ax.plot(*zip(*pts["DPA"]), ":o", color=OURS, lw=1.6, ms=4.2, mfc="white", mew=1.0, zorder=5)
    ax.plot(*pts["DPA"][2], "o", color=OURS, ms=4.2, mfc=OURS, mew=1.0, zorder=6)      # k-bar = 4, the budget of the right panel
    ax.annotate(r"$\bar{k}$=2", pts["DPA"][0], textcoords="offset points", xytext=(6, -2), fontsize=7, color="0.25")
    ax.annotate(r"$\bar{k}$=8", pts["DPA"][-1], textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color="0.25")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.margins(x=0.08, y=0.12)
    log_ticks(ax)
    ax.set_xlabel("TTFT (ms)", fontsize=8); ax.set_ylabel(r"KL divergence ($\mathregular{\times10^{-3}}$)", fontsize=8)
    exp, dpa = series(sig, "all 8"), series(sig, "DPA 4")
    den = series(dense, "all 8") if dense and Path(dense).exists() else None      # attention, router, head on AIMC (optional)
    pos = list(range(len(sigma)))
    t0 = sigma.index(1)
    bx.axvline(t0, color="0.45", ls=(0, (3, 2)), lw=0.9, zorder=1)
    bx.text(t0 + 0.06, 0.97, "task setting", transform=bx.get_xaxis_transform(), ha="left", va="top", fontsize=6.5, color="0.35")
    if den:
        bx.plot(pos, den, "--^", color=PRIOR, lw=1.4, ms=4, zorder=3)
    bx.plot(pos, exp, "-s", color=REF, lw=1.5, ms=4, zorder=4)
    bx.plot(pos, dpa, ":o", color=OURS, lw=1.6, ms=4.2, mfc=OURS, mew=1.0, zorder=5)
    def fmt(v):
        return f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"
    for i, sg in enumerate(sigma):             # the values of the largest variation only: highest of the curves above it, lowest below, the middle one beside it
        if i != len(sigma) - 1:
            continue
        pts_ = [(dpa[i], OURS), (exp[i], REF)] + ([(den[i], PRIOR)] if den else [])
        pts_.sort(key=lambda t: t[0], reverse=True)
        for rank, (v, col) in enumerate(pts_):
            if rank == 0:
                off, ha_ = (0, 5), "center"
            elif rank == len(pts_) - 1:
                off, ha_ = (0, -14), "center"
            else:
                off, ha_ = (6, 1), "left"
            bx.annotate(fmt(v), (pos[i], v), textcoords="offset points", xytext=off, ha=ha_, fontsize=7, color=col,
                        bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85), zorder=7)
    bx.set_yscale("log"); bx.set_xlim(-0.5, len(sigma) - 0.5); bx.set_xticks(pos); bx.set_xticklabels([str(v) for v in sigma]); bx.margins(y=0.2)
    lim = bx.get_ylim()
    ticks = [m * 10 ** e for e in range(0, 4) for m in (1, 2, 5) if lim[0] <= m * 10 ** e <= lim[1]]
    bx.yaxis.set_major_locator(FixedLocator(ticks)); bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bx.yaxis.set_minor_formatter(NullFormatter())
    bx.set_xlabel(r"Cell variation of 1T1R, $\sigma$ (%)", fontsize=8)
    lim = ax.get_xlim()
    xt = [v for v in (150, 200, 300, 500) if lim[0] <= v <= lim[1]]
    ax.xaxis.set_major_locator(FixedLocator(xt)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    for a_ in (ax, bx):
        a_.tick_params(labelsize=7.5); a_.grid(True, which="major", color="0.85", lw=0.6)
    ha = [Line2D([], [], color=UNI, ls="-", marker="s", ms=4, lw=1.5),
          Line2D([], [], color=OURS, ls=":", marker="o", ms=4.2, mfc="white", lw=1.6)]
    ax.legend(ha, [r"AIMC + uniform $k$", "OC-DPA"], loc="lower center", bbox_to_anchor=(0.5, 1.035), ncol=2, frameon=False,
              fontsize=6.2, columnspacing=0.8, handlelength=1.5, handletextpad=0.4, borderpad=0.1, borderaxespad=0)
    hb = [Line2D([], [], color=REF, ls="-", marker="s", ms=4, lw=1.5),
          Line2D([], [], color=OURS, ls=":", marker="o", ms=4.2, mfc=OURS, lw=1.6)]
    lb = ["experts on AIMC", r"+ OC-DPA $\bar{k}$=4"]
    if den:
        hb.append(Line2D([], [], color=PRIOR, ls="--", marker="^", ms=4, lw=1.4)); lb.append("all on AIMC")
    bx.legend(hb, lb, loc="lower right", bbox_to_anchor=(1.0, 1.035), ncol=3, frameon=False,
              fontsize=6.2, columnspacing=0.8, handlelength=1.5, handletextpad=0.4, borderpad=0.1, borderaxespad=0)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "quality_olmoe")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default="cache/hardware.json")
    ap.add_argument("--kl-task", help="olmoe=PATH,deepseek=PATH  (task-setting sweep over the prefill budget)")
    ap.add_argument("--kl-sigma", help="olmoe=PATH,deepseek=PATH  (sigma sweep, experts on AIMC)")
    ap.add_argument("--kl-sigma-dense", help="olmoe=PATH,deepseek=PATH  (sigma sweep, + attention/router/head on AIMC)")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--only", default="1,2,4,5")
    ap.add_argument("--fig1-star", action="store_true", help="mark OC-DPA with a star on the TTFT panel of Fig. 1")
    ap.add_argument("--fig1-right", default="sweep", choices=["sweep", "stacks", "sharing", "prompt", "pareto"], help="right panel of Fig. 1")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    STAR = a.fig1_star
    hw = json.load(open(a.hardware))
    todo = a.only.split(",")
    if "1" in todo:
        fig1_sweep(hw, out) if a.fig1_right == "sweep" else fig1(hw, out, a.fig1_right)
    if "2" in todo:
        fig2_arch(out)
    if "4" in todo:
        fig4(hw, {k: kl_records(v) for k, v in pairs(a.kl_task).items()}, out)
    if "5" in todo:
        fig5(pairs(a.kl_sigma), pairs(a.kl_sigma_dense), out)
    if "q" in todo:        # merged Fig. 4 (OLMoE): budget sweep (a) and cell variation (b)
        fig4_merged(hw, kl_records(pairs(a.kl_task)["olmoe"]), pairs(a.kl_sigma)["olmoe"], pairs(a.kl_sigma_dense)["olmoe"], out)
