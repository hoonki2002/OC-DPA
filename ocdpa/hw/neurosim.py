"""Per-conversion cost of one analog column read, measured with NeuroSim.

`probe` builds a single NeuroSim SubArray with our mapping -- 128 rows, binary
cells, an 8:1 column mux -- and reports the area, latency and energy of one
read, broken into ADC, accumulation, array and periphery. One read converts
`cols` columns, so dividing gives the cost of the quantity our simulator
counts: one column conversion for one input bit of one plane.

The ADC resolution is not free to choose. 128 rows of +/-1 cells driven by a
1-bit input produce 257 distinct sums, so the converter must resolve 8 bits for
the column read to be lossless; `adc_bits` below is the levelOutput that
follows from the array, not a tuning knob.
"""
from __future__ import annotations

import json, os, subprocess
from dataclasses import dataclass

from ocdpa.paths import CACHE as _CACHE_DIR, PROBE
CACHE = _CACHE_DIR / "neurosim_cost.json"       # memoised NeuroSim runs
CELLS = {"1T1R": (2, 1), "crossbar": (2, 4), "SRAM": (1, 1)}


@dataclass(frozen=True)
class Subarray:
    cell: str = "1T1R"
    rows: int = 128
    cols: int = 128
    adc_bits: int = 8
    col_muxed: int = 8
    sar: bool = True
    cell_bit: int = 1
    # NeuroSim ships Ron = 6 kOhm. A 128-row column of such cells drops most of
    # the read voltage in the wire, so the IR-drop solver in ocdpa/irdrop puts the
    # usable floor three orders higher; 300 kOhm is the value used throughout.
    r_on: float = 300e3

    @property
    def key(self):
        return (f"{self.cell}/{self.rows}x{self.cols}/adc{self.adc_bits}/mux{self.col_muxed}"
                f"/sar{int(self.sar)}/cb{self.cell_bit}/ron{self.r_on:.0f}")


def _probe(s: Subarray) -> dict:
    memcell, access = CELLS[s.cell]
    env = dict(os.environ,
               NS_MEMCELL=str(memcell), NS_ACCESS=str(access),
               NS_CELLBIT=str(s.cell_bit), NS_LEVELOUTPUT=str(2 ** s.adc_bits),
               NS_COLMUXED=str(s.col_muxed), NS_SARADC=str(int(s.sar)),
               NS_ROWS=str(s.rows), NS_COLS=str(s.cols), NS_RON=f"{s.r_on:.0f}")
    out = subprocess.run([str(PROBE)], env=env, capture_output=True, text=True).stdout
    rec = {}
    for line in out.strip().splitlines():
        if not line or line.startswith(("cfg,", "[")):
            continue
        f = line.split(",")
        for i in range(0, len(f) - 1, 2):
            try:
                rec[f[i]] = float(f[i + 1])
            except ValueError:
                pass
    if "readEnergy_pJ" not in rec:
        raise RuntimeError(f"probe gave no energy for {s.key}:\n{out}")
    return rec


def measure(s: Subarray = Subarray()) -> dict:
    """Per-column-conversion cost, cached so the C++ model runs once per config."""
    store = json.load(open(CACHE)) if CACHE.exists() else {}
    if s.key not in store:
        r = _probe(s)
        n = s.cols                      # conversions produced by one subarray read
        store[s.key] = {
            "pJ_per_conv": r["readEnergy_pJ"] / n,
            "pJ_adc": r["eADC_pJ"] / n,
            "pJ_array": r["eArray_pJ"] / n,
            "pJ_accum": r["eAccum_pJ"] / n,
            "pJ_other": r["eOther_pJ"] / n,
            "ns_per_read": r["readLatency_ns"],
            "conv_per_read": n,
            "area_um2": r["area_um2"],
            "area_adc_um2": r["areaADC_um2"],
            "area_array_um2": r["areaArray_um2"],
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(store, open(CACHE, "w"), indent=1)
    return store[s.key]


def energy_J(conversions: int, s: Subarray = Subarray()) -> dict:
    c = measure(s)
    terms = {k[3:]: conversions * c[k] * 1e-12 for k in
             ("pJ_adc", "pJ_array", "pJ_accum", "pJ_other")}
    terms["total"] = sum(terms.values())
    terms["adc_share"] = terms["adc"] / terms["total"]
    # Columns are converted `col_muxed` at a time, so a read's latency covers
    # cols/col_muxed conversions per cycle; serial array time is read latency
    # scaled by how many subarray reads the conversions amount to.
    terms["serial_ns"] = conversions / c["conv_per_read"] * c["ns_per_read"]
    return terms


if __name__ == "__main__":
    for cell in CELLS:
        c = measure(Subarray(cell=cell))
        print(f"{cell:9s} {c['pJ_per_conv']:7.3f} pJ/conv   ADC {100*c['pJ_adc']/c['pJ_per_conv']:5.1f}%"
              f"   {c['ns_per_read']:7.1f} ns/read   {c['area_um2']:8.0f} um2"
              f"  (ADC {100*c['area_adc_um2']/c['area_um2']:.0f}%)")
