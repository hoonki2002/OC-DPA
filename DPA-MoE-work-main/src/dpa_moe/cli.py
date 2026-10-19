"""CLI entry point (``dpa-moe-encode``) — phase dispatch + variant registry.

Phases:
    calibrate → assemble → finalize    (``encode`` = calibrate + assemble,
                                         ``all`` = every phase)

Variants:
    ss_anybcq           : FLAG_ANY_PREC=0, one shared α set (SharedScale
                          AnyBCQ). Default; joint incremental
                          block-reconstruction calibration.
    anybcq              : FLAG_ANY_PREC=1, per-tier α/β (standard AnyBCQ).

Each experiment config (``experiments/config/<model>/base.yaml``)
describes exactly one model; see ``_model_entry`` for the keys read here.
"""
from __future__ import annotations

import argparse
import pathlib

import yaml

# Triggering the upstream-packaging shim must happen before any
# transitive `anybcq` import chain fires.
from . import upstream_shims  # noqa: F401  side-effect import
from .adapters import get_adapter
from .assemble import do_assemble
from .calibrate import do_calibrate_anybcq, do_calibrate_ss_anybcq
from .hp import HP
from .reader import _packed_chunks_anybcq, _packed_chunks_ss_anybcq
from .runtime import set_required_runtime

# Variant registry — single dispatch table for calibrate + reader.
# Keep ss_anybcq and anybcq paths as separate top-level functions so the
# two encoding paths stay unambiguous (no flag-driven branching mid-stream).
#
# A variant selects the on-disk packing + calibration policy.  The MoE
# arch (tensor-name map, HF unfuse logic, GGUF metadata) is orthogonal
# and is selected by the config's ``model.adapter``.
_VARIANT_REGISTRY = {
    "ss_anybcq": {
        "calibrate": do_calibrate_ss_anybcq,
        "read":      _packed_chunks_ss_anybcq,
        "flag_any_prec": False,
        "unmanaged_qtype": "Q8_0",
    },
    "anybcq": {
        "calibrate": do_calibrate_anybcq,
        "read":      _packed_chunks_anybcq,
        "flag_any_prec": True,
    },
}

# Per-adapter GGUF arch keys — passed to GGUFWriter so the file declares
# the right architecture for upstream llama.cpp's model dispatch.
_GGUF_ARCH = {
    "qwen3_moe":    "qwen3moe",
    "deepseek_moe": "deepseek",
}


def _model_entry(manifest: dict) -> dict:
    """Model-level settings from a config's ``model:`` block."""
    model = manifest["model"]
    return {
        "key": model["key"],
        "adapter": model["adapter"],
        "hf_dir": pathlib.Path(model["hf"]["local_dir"]),
        "codebooks_dirs": model["codebooks"],
        "final_ggufs": model["ggufs"],
    }


def _load_config(path: pathlib.Path) -> tuple[dict, HP, list[pathlib.Path]]:
    manifest = yaml.safe_load(open(path))
    encode = manifest["encode"]
    hp = HP.from_manifest(dict(encode["hyperparams"]))
    calib_files = [pathlib.Path(p) for p in encode["calibration"]["files"]]
    return manifest, hp, calib_files


def _resolve_variant_paths(entry: dict, variant: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Per-variant codebooks_dir + final_gguf from the ``model:`` block."""
    return (pathlib.Path(entry["codebooks_dirs"][variant]),
            pathlib.Path(entry["final_ggufs"][variant]))


def main():
    p = argparse.ArgumentParser(prog="dpa-moe-encode")
    p.add_argument("phase", choices=["calibrate", "assemble", "encode",
                                      "finalize", "all"])
    p.add_argument("--config", required=True, type=pathlib.Path,
                   help="experiments/config/<model>/base.yaml")
    p.add_argument("--variant", default=None,
                   choices=list(_VARIANT_REGISTRY.keys()),
                   help=("Encoder variant (default: the config's encode.variant, "
                         "else ss_anybcq). ss_anybcq = FLAG_ANY_PREC=0, shared "
                         "α/β; anybcq = FLAG_ANY_PREC=1, per-tier α/β."))
    p.add_argument("--model-key", default=None,
                   help="Optional check that the config's model.key matches.")
    p.add_argument("--limit-layers", type=int, default=None,
                   help="DEBUG: encode only the first N MoE layers (fast iteration).")
    p.add_argument("--force", action="store_true",
                   help="Re-run a phase even if its output already exists.")
    p.add_argument("--calibration-mode", default="sequential",
                   choices=["sequential", "layer_fp_parallel"],
                   help=("Calibration execution mode. sequential is the original "
                         "layer-sequential AnyBCQ contract; layer_fp_parallel "
                         "calibrates the layer range in DPA_MOE_LAYER_RANGE "
                         "against full-precision inputs (one process per GPU, "
                         "see experiments/2_encode/calibrate_parallel.sh)."))
    p.add_argument("--residuals-npz", type=pathlib.Path, default=None,
                   help=("Optional expert_residuals.npz from "
                         "experiments/3_dynamic_precision/profiling/profile_residuals.py. "
                         "When given, the assemble phase packs R + N + "
                         "K_min/K_max as ``dpa_moe.expert_residuals.*`` "
                         "KV arrays so the runtime can compute a static "
                         "layout for any budget without an external npz."))
    p.add_argument("--codebooks-dir", type=pathlib.Path, default=None,
                   help=("Override the config codebooks dir for calibration "
                         "or assembly. Useful for ablation runs that must not "
                         "overwrite the baseline codebooks."))
    p.add_argument("--final-gguf", type=pathlib.Path, default=None,
                   help="Override the config final GGUF path for assembly.")
    args = p.parse_args()

    manifest, hp, calib_files = _load_config(args.config)
    entry = _model_entry(manifest)
    if args.model_key is not None and args.model_key != entry["key"]:
        raise SystemExit(f"--model-key {args.model_key!r} does not match "
                         f"config model.key {entry['key']!r} ({args.config})")
    variant = args.variant or manifest["encode"].get("variant", "ss_anybcq")
    codebooks_dir, final = _resolve_variant_paths(entry, variant)
    if args.codebooks_dir is not None:
        codebooks_dir = args.codebooks_dir
    if args.final_gguf is not None:
        final = args.final_gguf

    variant_entry = _VARIANT_REGISTRY[variant]
    adapter_name = entry["adapter"]
    adapter = get_adapter(adapter_name)
    gguf_arch = _GGUF_ARCH.get(adapter_name)
    if gguf_arch is None:
        raise SystemExit(
            f"adapter {adapter_name!r} has no GGUF arch mapping in cli._GGUF_ARCH "
            "— add an entry there before encoding.")

    if args.phase in ("calibrate", "encode", "all"):
        sentinel = codebooks_dir / "layer_00.safetensors"
        if sentinel.exists() and not args.force:
            print(f"[calibrate] SKIP — {sentinel} exists (--force to redo)",
                  flush=True)
        else:
            variant_entry["calibrate"](
                entry["hf_dir"], calib_files, hp, codebooks_dir,
                limit_layers=args.limit_layers, adapter=adapter,
                calibration_mode=args.calibration_mode)

    if args.phase in ("assemble", "encode", "all"):
        # Assemble writes the final GGUF (Q8_0 unmanaged tensors inline);
        # llama-quantize is not used because its validator rejects the
        # 1-byte canonical placeholders for managed tensors.
        if final.exists() and not args.force:
            print(f"[assemble] SKIP — {final} exists (--force to redo)",
                  flush=True)
        else:
            do_assemble(entry["hf_dir"], codebooks_dir, final, hp,
                        variant=variant,
                        limit_layers=args.limit_layers,
                        required_runtime=False,
                        reader=variant_entry["read"],
                        unmanaged_qtype=variant_entry.get("unmanaged_qtype", "Q8_0"),
                        adapter=adapter, gguf_arch=gguf_arch,
                        residuals_npz=args.residuals_npz)

    if args.phase in ("finalize", "all"):
        set_required_runtime(final, True)


if __name__ == "__main__":
    main()
