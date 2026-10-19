"""Experiment configs resolve to a known adapter, GGUF arch and output paths."""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from dpa_moe import HP

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "experiments/config").glob("*/base.yaml"))


@pytest.fixture(scope="module")
def cli():
    from dpa_moe import cli
    return cli


def test_configs_present():
    assert CONFIGS, "no experiments/config/*/base.yaml found"


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.parent.name)
def test_cli_resolves_config(cli, config):
    from dpa_moe.adapters import get_adapter

    manifest, hp, calib_files = cli._load_config(config)
    entry = cli._model_entry(manifest)
    assert isinstance(hp, HP)
    assert calib_files, "encode.calibration.files is empty"
    assert get_adapter(entry["adapter"]).name == entry["adapter"]
    assert entry["adapter"] in cli._GGUF_ARCH
    variant = manifest["encode"].get("variant", "ss_anybcq")
    assert variant in cli._VARIANT_REGISTRY
    codebooks, gguf = cli._resolve_variant_paths(entry, variant)
    assert gguf.suffix == ".gguf" and codebooks.name


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.parent.name)
def test_encode_harness_config(config):
    out = subprocess.run(
        [sys.executable, str(REPO / "experiments/2_encode/_encode_config.py"), str(config)],
        check=True, capture_output=True, text=True).stdout
    fields = dict(line.split("=", 1) for line in out.splitlines())
    assert set(fields) == {"MODEL_KEY", "VARIANT", "N_LAYERS", "START_LAYER", "CFG_GPUS",
                           "CFG_INNER_BS", "CODEBOOKS_DIR", "FINAL_GGUF"}
    assert int(fields["N_LAYERS"]) > int(fields["START_LAYER"]) >= 0
