"""CPU check of the paper's numbers at one point per step: no GPU, no model, no NeuroSim build, a few minutes.

  python scripts/00_verify.py

  1  encoder       a random matrix: deterministic, error falls with k, scales fall by ~1.8x per plane      (Sec. III-B)
  2  LUT           the residual LUT of the allocator is 6,912 B = 6.75 KB                                   (Sec. III-C)
  3  allocator     the greedy rule returns the dynamic program's plane counts on 100 % of the dispatches    (Sec. III-C)
  4  analog path   the bit-serial array computes what the hardware model prices                             (Sec. IV-A)
  5  hardware      Table I / II numbers of OLMoE (TTFT, area, throughput per area) against results/        (Sec. IV-B)

Every step prints PASS or FAIL; the exit code is the number of failures.
"""
import json, re, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np, torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
PY = sys.executable
fails = 0


def step(name):
    def deco(fn):
        global fails
        t0 = time.time()
        try:
            msg = fn()
            print(f"PASS  {name}: {msg}  [{time.time() - t0:.0f} s]", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL  {name}: {e}  [{time.time() - t0:.0f} s]", flush=True)
        return fn
    return deco


def run(*cmd, **kw):
    r = subprocess.run([PY, *map(str, cmd)], cwd=REPO, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr)[-400:])
    return r.stdout


@step("1 encoder")
def _():
    from ocdpa.encoder import compute_codebook
    torch.manual_seed(0)
    W = torch.randn(32, 256) * 0.02
    B, a, b, n, m, J = compute_codebook(W, refine_rounds=5, expansion_rounds=5)
    B2, a2, b2, *_ = compute_codebook(W, refine_rounds=5, expansion_rounds=5)
    assert torch.equal(B, B2) and torch.equal(a, a2) and torch.equal(b, b2), "not deterministic"
    w = W.reshape(-1, 128)
    err = [float(((b[:, None] + (a[:, None, :k] * B[..., :k]).sum(-1) - w) ** 2).sum() / (w ** 2).sum()) for k in range(2, 9)]
    assert all(x > y for x, y in zip(err, err[1:])), f"error does not fall with k: {err}"
    ratio = float((a[:, 1:] / a[:, :-1]).mean())
    assert 0.5 < ratio < 0.65, f"alpha ratio {ratio:.2f}"
    return f"deterministic; relative error {err[0]:.2e} (k=2) -> {err[-1]:.1e} (k=8); alpha falls by {1 / ratio:.2f}x per plane"


@step("2 LUT")
def _():
    out = Path(tempfile.mkdtemp()) / "lut.hex"
    run("scripts/04_export_lut.py", "--out", out)
    s = json.load(open(str(out) + ".json"))
    assert s["bytes"] == 6912, s
    return f"{s['rows']} rows x {s['entries_per_row']} entries x {s['bits']} bits = {s['bytes']:,} B (6.75 KB)"


@step("3 allocator")
def _():
    o = run("scripts/05_check_allocator.py", "--device", "cpu", "--dispatches", "512", "--tokens", "128")
    g = [float(x) for x in re.findall(r"greedy\s+same plane counts as the DP on\s+([\d.]+)%", o)]
    h = [float(x) for x in re.findall(r"hardware\s+same plane counts as the DP on\s+([\d.]+)%", o)]
    assert g and all(x == 100.0 for x in g), f"greedy {g}"
    return f"greedy = DP on {g[0]:.0f}% of the dispatches, the fixed-point datapath on {h[0]:.0f}%"


@step("4 analog path")
def _():
    o = run("scripts/08_check_analog_path.py")
    worst = max(float(x) for x in re.findall(r"= ([\d.]+e-\d+)", o))
    assert worst < 1e-3, o[-300:]
    return f"all deviations <= {worst:.1e}"


@step("5 IR-drop")
def _():
    import os
    tmp = tempfile.mkdtemp()
    r = subprocess.run([PY, str(REPO / "scripts" / "07_calibrate_irdrop.py")], cwd=tmp, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(REPO)))
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr)[-300:])
    one, xb = re.findall(r"varies with position by ([\d.]+)% \(rms\)", r.stdout)
    assert abs(float(one) - 0.7) < 0.05 and abs(float(xb) - 2.2) < 0.05, (one, xb)
    return f"column gain varies by {one}% (1T1R) and {xb}% (crossbar)"


@step("6 hardware")
def _():
    tmp = Path(tempfile.mkdtemp())
    run("scripts/13_hardware_report.py", "--out", tmp)
    new, ref = json.load(open(tmp / "hardware_claims.json")), json.load(open(REPO / "results" / "hardware_claims.json"))
    for k in ("ttft_ms", "mm2", "tok_s_mm2"):
        for d in ("Best baseline", "OC-DPA"):
            a, r = new[f"OLMoE-1B-7B.{d}"][k], ref[f"OLMoE-1B-7B.{d}"][k]
            assert abs(a / r - 1) < 1e-3, (d, k, a, r)
    o = new["OLMoE-1B-7B.OC-DPA"]
    return f"OLMoE OC-DPA {o['ttft_ms']:.0f} ms, {o['mm2']:.0f} mm2, {o['tok_s_mm2']:.1f} tok/s/mm2 as in results/"


print(f"\n{'all checks passed' if not fails else str(fails) + ' check(s) failed'}")
sys.exit(fails)
