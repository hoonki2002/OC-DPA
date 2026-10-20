"""PPA of the precision allocator: wrapper around the NeuroSim `allocator` driver.

`allocator` builds the block out of NeuroSim's own primitive library at the
same technology node as the analog subarray, so the two numbers are
commensurable. What matters for the paper is not the allocator's absolute cost
but two ratios:

  energy   allocator energy per dispatch / ADC energy of that dispatch
  latency  allocator latency / the window it has to hide in

The second window is set by the algorithm, not by scheduling luck. Every
expert is granted at least KMIN = 2 planes, so planes 0 and 1 are read
unconditionally and their conversions cannot depend on the allocation. The
allocator therefore runs in the shadow of two plane reads, a window that is one
subarray read per plane however widely the dispatch is parallelised.
"""
from __future__ import annotations

import os, subprocess
from dataclasses import dataclass


from ocdpa.paths import ALLOCATOR as BIN


@dataclass(frozen=True)
class Allocator:
    mode: int = 0
    dp_par: int = 1
    n: int = 8
    kbar: int = 4
    kmin: int = 2
    kmax: int = 8
    q_lut: int = 9
    q_s: int = 12
    lut_rows: int = 1024
    banks: int = 1
    technode: int = 22


def run(a: Allocator) -> dict:
    env = dict(os.environ, PA_MODE=str(a.mode), PA_DPPAR=str(a.dp_par), PA_N=str(a.n), PA_KBAR=str(a.kbar), PA_KMIN=str(a.kmin),
               PA_KMAX=str(a.kmax), PA_QLUT=str(a.q_lut), PA_QS=str(a.q_s),
               PA_LUTROWS=str(a.lut_rows), PA_BANKS=str(a.banks),
               PA_TECHNODE=str(a.technode))
    out = subprocess.run([str(BIN)], env=env, capture_output=True, text=True).stdout
    rec = {}
    for line in out.splitlines():
        if not line or line.startswith(("cfg,", "[", "dbg,")):
            continue
        f = line.split(",")
        for i in range(0, len(f) - 1, 2):
            try:
                rec[f[i]] = float(f[i + 1])
            except ValueError:
                pass
    if "area_um2" not in rec:
        raise RuntimeError(f"allocator gave no area:\n{out}")
    return rec
