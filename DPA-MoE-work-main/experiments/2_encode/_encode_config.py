"""Print step-2 settings from an experiment config as KEY=VALUE lines.

Used by run.sh and calibrate_parallel.sh:

    eval "$(.venv/bin/python experiments/2_encode/_encode_config.py CONFIG [VARIANT])"

Emits MODEL_KEY, VARIANT, N_LAYERS, START_LAYER, CFG_GPUS, CFG_INNER_BS,
CODEBOOKS_DIR and FINAL_GGUF. START_LAYER skips leading dense decoder
layers (``first_k_dense_replace`` in the HF config, e.g. DeepSeek-MoE),
which have no experts to calibrate.
"""
from __future__ import annotations

import json
import pathlib
import shlex
import sys

import yaml


def main() -> None:
    config_path = pathlib.Path(sys.argv[1])
    m = yaml.safe_load(config_path.read_text())
    model, encode = m["model"], m["encode"]
    variant = (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2]
               else encode.get("variant", "ss_anybcq"))

    hf_config = pathlib.Path(model["hf"]["local_dir"]) / "config.json"
    start_layer = 0
    if hf_config.exists():
        start_layer = int(json.loads(hf_config.read_text()).get("first_k_dense_replace") or 0)

    fields = {
        "MODEL_KEY": model["key"],
        "VARIANT": variant,
        "N_LAYERS": model["n_layers"],
        "START_LAYER": start_layer,
        "CFG_GPUS": ",".join(str(g) for g in encode.get("gpus", [])),
        "CFG_INNER_BS": encode.get("inner_bs", 2),
        "CODEBOOKS_DIR": model["codebooks"][variant],
        "FINAL_GGUF": model["ggufs"][variant],
    }
    for key, value in fields.items():
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
