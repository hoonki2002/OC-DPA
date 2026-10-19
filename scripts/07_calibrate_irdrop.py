"""Column-position attenuation of a binary-plane read, from the mesh IR-drop solver.

The solver (ocdpa/irdrop/mesh.py) resolves the wordline/bitline network of one
array. Per-token solves are out of reach, so the read is reduced to a static
gain per physical column: the mean of (solved column current / ideal column
current) over random binary patterns at 50% density and random read voltages.
Cells use Ron = 300 kOhm (the value of the NeuroSim runs), Roff = 150 x Ron;
the 1T1R cell adds its access resistance, Ron x 0.25.

  1T1R      128 x 128, all 128 rows driven
  crossbar  256 x 256, 128 rows driven per read (top or bottom half)

Run first as a check: the relative error norm must reproduce the 6.1% / 7.3%
quoted for the two cell types.
"""
import os, sys, time
import numpy as np
from ocdpa.irdrop import mesh

RON = 300e3
mesh.R_ON, mesh.R_OFF = RON, 150 * RON
mesh.CELLS["1T1R"] = (4, 12, RON * mesh.IR_DROP_TOLERANCE)


def run(cell, n, m, active, trials, rng):
    r_pitch, c_pitch, r_series = mesh.CELLS[cell]
    r_row, r_col = mesh.UNIT_R * r_pitch * mesh.FEATURE, mesh.UNIT_R * c_pitch * mesh.FEATURE
    gains, errs = [], []
    for _ in range(trials):
        on = rng.random((n, m)) < 0.5
        g = 1.0 / (np.where(on, mesh.R_ON, mesh.R_OFF) + r_series)
        v = np.zeros(n)
        v[active] = rng.random(len(v[active])) * 0.5
        got, want = mesh.solve(g, v, r_row, r_col), mesh.ideal(g, v)
        gains.append(got / want)
        errs.append(np.linalg.norm(got - want) / np.linalg.norm(want))
    return np.mean(gains, 0), float(np.mean(errs))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = {}
    t0 = time.time()
    g, e = run("1T1R", 128, 128, slice(0, 128), trials, rng)
    out["1t1r"] = g[None]
    print(f"1T1R 128x128: relative error {100*e:.2f}%  gain {g.min():.3f}..{g.max():.3f} [{time.time()-t0:.0f}s]", flush=True)
    halves = []
    for lab, rows in (("top", slice(0, 128)), ("bottom", slice(128, 256))):
        g, e = run("crossbar", 256, 256, rows, trials, rng)
        halves.append(g)
        print(f"crossbar 256x256 ({lab} half driven): relative error {100*e:.2f}%  gain {g.min():.3f}..{g.max():.3f} [{time.time()-t0:.0f}s]", flush=True)
    out["xbar"] = np.stack(halves)
    os.makedirs("cache", exist_ok=True)
    np.savez("cache/irdrop_gain.npz", **out)
    for key, label in (("1t1r", "1T1R"), ("xbar", "crossbar")):
        g = out[key] / out[key].mean()
        print(f"{label}: column gain varies with position by {100 * np.sqrt(((g - 1) ** 2).mean()):.1f}% (rms) after the global trim")
