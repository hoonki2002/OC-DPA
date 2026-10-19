"""The encoder and the llama.cpp runtime agree on the stream format and GGUF keys.

Reads the runtime sources from the third_party/llama.cpp submodule; skipped
when the submodule is not checked out.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from dpa_moe import format as fmt

EXT = pathlib.Path(__file__).resolve().parents[1] / "third_party/llama.cpp/dpa-moe-ext"
pytestmark = pytest.mark.skipif(not EXT.is_dir(), reason="llama.cpp submodule not checked out")


def _runtime_sources() -> str:
    return "\n".join(p.read_text(errors="ignore")
                     for p in EXT.rglob("*") if p.suffix in {".h", ".cpp", ".cu", ".cuh"})


def test_stream_header_constants():
    src = (EXT / "decoder/anybcq/kernels/anybcq_format.h").read_text()

    def const(name: str) -> int:
        m = re.search(rf"\b{name}\s*=\s*(0x[0-9A-Fa-f]+|\d+)", src)
        assert m, f"{name} not found in anybcq_format.h"
        return int(m.group(1), 0)

    assert const("kStreamMagic") == fmt.MAGIC
    assert const("kStreamVersion") == fmt.STREAM_VERSION
    assert const("kHeaderSize") == fmt.HEADER_SIZE
    assert re.search(r'struct\.Struct\("<IBHBBB2xIIII4x"\)', src)
    assert fmt.HEADER_STRUCT.format == "<IBHBBB2xIIII4x"


@pytest.mark.parametrize("key", [
    "version", "encoder", "group_size", "base_precision", "target_precision",
    "managed_tensors", "required_runtime", "executor", "bundle_tensors",
])
def test_scalar_gguf_keys_known_to_runtime(key):
    assert f'"{fmt.KV_PREFIX}{key}"' in _runtime_sources()


@pytest.mark.parametrize("prefix", ["tensor.", "bytes.", "bundle."])
def test_per_tensor_gguf_key_prefixes_known_to_runtime(prefix):
    assert f'"{fmt.KV_PREFIX}{prefix}' in _runtime_sources()
