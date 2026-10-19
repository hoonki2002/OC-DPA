"""Compute per-token byte budget for the dpa_moe chunked GGUF.

Reads the actual GGUF tensor sizes (no hard-coded numbers — they
change when the encoder's dtype policy changes).  The MoE term
depends on K[L, e]; the rest is a constant per token.

Breakdown emitted:
  attn      — Σ blk.X.attn_{q,k,v,output}.weight bytes (read every token)
  norms     — small (RMSNorms), included in attn for simplicity
  lm_head   — output.weight bytes (read every output token)
  token_embd— token_embd.weight bytes (read every input token)
  router    — Σ blk.X.ffn_gate_inp.weight bytes (cheap, F32 today)
  moe       — top_k × n_layers × K̄ × chunk_bytes_per_expert
              (one (gate+up+down)-set of K chunks; runtime loads
              exactly K chunks per dispatched expert)

Per-token MoE bytes scale linearly with K̄.  The other terms are
K-independent.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[3]  # analysis → 4_bench → experiments → DPA-MoE
sys.path.insert(0, str(_REPO / "src" / "llama.cpp" / "gguf-py"))

# bytes/element for each GGML dtype — used in both gguf_bytes() and full_model_bytes()
_GGML_TYPE_BYTES: dict[str, float] = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0,
    "Q4_0": 0.5625, "Q4_1": 0.625, "Q5_0": 0.6875, "Q5_1": 0.75,
    "Q8_0": 1.0625,
    "Q2_K": 0.328125, "Q3_K": 0.4375, "Q4_K": 0.5625,
    "Q5_K": 0.6875,  "Q6_K": 0.8203125,
}


def gguf_bytes(gguf_path: pathlib.Path):
    """Return a dict of byte sizes per logical component."""
    import gguf
    import re
    r = gguf.GGUFReader(str(gguf_path))

    def t_bytes(t):
        # tensor.shape from gguf is reverse-of-numpy.  Total element count
        # is product.
        n = 1
        for d in t.shape: n *= int(d)
        bpe = _GGML_TYPE_BYTES.get(t.tensor_type.name)
        if bpe is None:
            return 0  # unknown dtype — skip
        return int(n * bpe)

    components = {
        "attn": 0, "lm_head": 0, "token_embd": 0,
        "router": 0, "moe_meta": 0,  # moe_meta = canonical placeholders
        "norms": 0, "other": 0,
    }
    moe_chunk_bytes_per_expert = 0
    moe_n_chunks_per_expert = 0
    for t in r.tensors:
        name = t.name
        sz = t_bytes(t)
        if name.startswith("blk.") and "attn_" in name and ".weight" in name and ".scale" not in name:
            components["attn"] += sz
        elif name == "output.weight":
            components["lm_head"] += sz
        elif name == "token_embd.weight":
            components["token_embd"] += sz
        elif name.endswith(".ffn_gate_inp.weight"):
            components["router"] += sz
        elif name.endswith(".ffn_gate_exps.weight") or name.endswith(".ffn_up_exps.weight") or name.endswith(".ffn_down_exps.weight"):
            components["moe_meta"] += sz   # 1-element f16 stubs in chunked GGUF
        elif "_norm" in name:
            components["norms"] += sz
        else:
            components["other"] += sz

    # dpa_moe packs per-expert chunk bytes as managed-tensor blobs + KV.
    # Some architectures (DeepSeek-MoE) have dense early layers, so find the
    # first layer that has all three managed expert tensors instead of assuming
    # blk.0 is MoE.
    by_layer: dict[int, dict[str, str]] = {}
    managed_layers: set[int] = set()
    managed_experts: set[int] = set()
    pat = re.compile(
        r"^dpa_moe\.tensor\.blk\.(\d+)\."
        r"(ffn_(?:gate|up|down)_exps)\.weight:e(\d+)\.chunk_bytes$"
    )
    for kv in r.fields:
        m = pat.match(kv)
        if not m:
            continue
        layer = int(m.group(1))
        expert = int(m.group(3))
        managed_layers.add(layer)
        managed_experts.add(expert)
        if expert == 0:
            by_layer.setdefault(layer, {})[m.group(2)] = kv
    chunk_b_kv = 0
    chunk_count = 0
    for _layer, keys in sorted(by_layer.items()):
        if {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"} <= set(keys):
            for kv in keys.values():
                f = r.fields[kv]
                chunk_count = len(f.data)
                # The first chunk byte size (all chunks same for ss_anybcq).
                first = int(f.parts[f.data[0]][0])
                chunk_b_kv += first
            break
    components["moe_chunk_per_expert"] = chunk_b_kv  # gate + up + down chunk bytes
    components["moe_n_chunks_per_expert"] = chunk_count
    components["moe_managed_layers"] = len(managed_layers)
    components["moe_managed_experts"] = len(managed_experts)
    return components


def per_token_bytes(gguf_path: pathlib.Path, kbar: float,
                     top_k: int = 8, n_layers: int = 48,
                     match_q8: bool = False) -> dict:
    """Return per-token bytes dict.

    ``kbar`` is the average chunks pinned per (L, e); top_k is how
    many experts the router selects per token; n_layers is the
    number of MoE layers.

    Default (``match_q8=False``) reads ACTUAL on-disk tensor sizes
    from the GGUF.  After the lm_head repack to Q8_0, this gives the
    correct per-token byte budget directly comparable to standard
    Q-quant baselines (which also keep token_embd F16, router F32).

    ``match_q8=True`` (legacy) projects all non-managed weights to a
    hypothetical Q8 size — useful only when the GGUF being audited
    has lm_head F32 and you want a "what if it were Q8" estimate.
    The token_embd and router projections are LIES (real Q-baselines
    keep those at F16 and F32 respectively); don't use this for
    plot axes against Q-quant anchors.
    """
    b = gguf_bytes(gguf_path)
    Q8_BPE = 1.0625   # bytes/element for Q8_0
    if match_q8:
        # Replace F32 lm_head with its Q8_0 equivalent (≈ 1/3.76 of F32 bytes).
        # We assume lm_head is F32 (4 BPE) on disk and we want to project
        # it to Q8_0.  Same for any other oversized fixed components.
        attn        = int(b["attn"]       * Q8_BPE / 1.0625)   # attn already Q8_0
        lm_head     = int(b["lm_head"]    * Q8_BPE / 4.0)      # F32 → Q8_0
        token_embd  = int(b["token_embd"] * Q8_BPE / 2.0)      # F16/BF16 → Q8_0
        router      = int(b["router"]     * Q8_BPE / 4.0)      # F32 → Q8_0
        norms       = b["norms"]                                # F32 norms stay
    else:
        attn        = b["attn"]
        lm_head     = b["lm_head"]
        token_embd  = b["token_embd"]
        router      = b["router"]
        norms       = b["norms"]
    # ``other`` includes fixed architecture weights outside attention/router,
    # for example DeepSeek's dense layer-0 FFN and shared experts.
    other = b["other"] + b["moe_meta"]
    moe = int(top_k * n_layers * kbar * b["moe_chunk_per_expert"])
    return {
        "attn":       attn,
        "lm_head":    lm_head,
        "token_embd": token_embd,
        "router":     router,
        "norms":      norms,
        "other":      other,
        "moe":        moe,
        "total":      attn + lm_head + token_embd + router + norms + other + moe,
    }


def full_model_bytes(gguf_path: pathlib.Path) -> int:
    """Sum of ALL tensor bytes in the GGUF — the true memory footprint."""
    import gguf
    r = gguf.GGUFReader(str(gguf_path))
    total = 0
    for t in r.tensors:
        n = 1
        for d in t.shape: n *= int(d)
        bpe = _GGML_TYPE_BYTES.get(t.tensor_type.name, 0)
        total += int(n * bpe)
    return total


def effective_model_bytes(gguf_path: pathlib.Path, kbar: float,
                           n_experts: int | None = None, top_k: int = 8,
                           n_layers: int | None = None) -> int:
    """Effective model weight = non-expert bytes + ALL expert bytes at kbar.

    For MoE with streaming: only top_k experts activate per token, but you
    need to store all n_experts.  The "effective weight" is:
        non_expert_bytes + n_experts * n_layers * kbar * chunk_bytes_per_expert

    This is equivalent to: non_expert + per_token_moe * (n_experts / top_k).
    """
    b = gguf_bytes(gguf_path)
    if n_experts is None:
        n_experts = int(b.get("moe_managed_experts", 0))
    if n_layers is None:
        n_layers = int(b.get("moe_managed_layers", 0))
    non_expert = (
        b["attn"] + b["lm_head"] + b["token_embd"] + b["router"]
        + b["norms"] + b["other"] + b["moe_meta"]
    )
    all_expert = int(n_experts * n_layers * kbar * b["moe_chunk_per_expert"])
    return non_expert + all_expert


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=pathlib.Path,
                     default=pathlib.Path("cache/models/Qwen3-30B-A3B/chunked/ss_anybcq.gguf"))
    ap.add_argument("--kbars", default="2,3,4,5,6,7,8",
                     help="comma-separated K̄ values to evaluate")
    args = ap.parse_args()

    b = gguf_bytes(args.gguf)
    print(f"--- {args.gguf.name} fixed components (bytes; GB) ---")
    for k, v in sorted(b.items()):
        if isinstance(v, int):
            print(f"  {k:12s} {v:>14d}  ({v/1e9:.3f} GB)")
    print()
    print(f"  n_chunks_per_expert (max physical) = {b['moe_n_chunks_per_expert']}")
    print(f"  chunk bytes per expert (gate+up+down) = {b['moe_chunk_per_expert']/1e3:.1f} KB")
    print()
    print(f"{'K̄':>4s}  {'attn':>8s}  {'lm_head':>8s}  {'embd':>8s}  "
           f"{'router':>8s}  {'other':>8s}  {'moe':>8s}  {'TOTAL':>8s}")
    for k in args.kbars.split(","):
        try: kbar = float(k.strip())
        except: continue
        d = per_token_bytes(args.gguf, kbar)
        print(f"  {kbar:4.2f} "
               f"  {d['attn']/1e9:6.3f} G  {d['lm_head']/1e9:6.3f} G  "
               f"{d['token_embd']/1e9:6.3f} G  {d['router']/1e9:6.3f} G  "
               f"{d['other']/1e9:6.3f} G  {d['moe']/1e9:6.3f} G  "
               f"{d['total']/1e9:6.3f} G")


if __name__ == "__main__":
    main()
