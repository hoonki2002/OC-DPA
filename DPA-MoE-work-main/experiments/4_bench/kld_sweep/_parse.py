"""Shared KLD / PPL output parsing for the layout sweep."""
from __future__ import annotations

import pathlib
import re
import statistics

# Canonical regex patterns — single source of truth.
# Applied with re.MULTILINE against llama-perplexity stdout.
KLD_PATTERNS: dict[str, str] = {
    "mean_kld":       r"^Mean\s+KLD:\s+([\d.eE+-]+)",
    "median_kld":     r"^Median\s+KLD:\s+([\d.eE+-]+)",
    "max_kld":        r"^Maximum\s+KLD:\s+([\d.eE+-]+)",
    "ppl_q":          r"^q-ppl:\s+([\d.eE+-]+)",
    "ppl_base":       r"^base-ppl:\s+([\d.eE+-]+)",
    "ppl_ratio":      r"^ppl\(q\)/ppl\(base\):\s+([\d.eE+-]+)",
    "same_top_p_pct": r"^Same top p:\s+([\d.]+)",
}


def parse_kld_output(text: str) -> dict[str, float] | None:
    """Parse llama-perplexity stdout; return metrics dict or None if no mean_kld."""
    out: dict[str, float] = {}
    for key, pat in KLD_PATTERNS.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    if "mean_kld" not in out:
        return None
    return out


def kbar_from_err(err_path: pathlib.Path) -> float | None:
    """Extract observed mean K̄ from .err log (DPA_MOE_LOG_KBAR output)."""
    if not err_path.exists():
        return None
    latest: dict[int, float] = {}
    for line in err_path.read_text(errors="replace").splitlines():
        m = re.search(r"kbar L=(\d+).*K_mean=([\d.]+)", line)
        if m:
            latest[int(m.group(1))] = float(m.group(2))
    return statistics.mean(latest.values()) if latest else None
