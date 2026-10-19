"""Idempotent byte-patch of `dpa_moe.required_runtime` in a GGUF file."""
from __future__ import annotations

import pathlib

from .format import KV_PREFIX


def set_required_runtime(gguf_path: pathlib.Path, value: bool) -> None:
    """Idempotent + readback-verified byte-patch of the
    `dpa_moe.required_runtime` bool KV.

    1. Read the file via gguf-py to locate the field's byte offset.
    2. If the stored value already equals `value`, do nothing.
    3. Otherwise: open the file `r+b`, seek to the bool byte, write
       `value`.
    4. Re-read via gguf-py and assert the stored value matches.
    """
    import gguf
    r = gguf.GGUFReader(str(gguf_path))
    field = r.get_field(f"{KV_PREFIX}required_runtime")
    if field is None:
        raise RuntimeError(f"required_runtime field missing from {gguf_path}")
    current = bool(field.parts[field.data[0]][0])
    if current == value:
        print(f"[finalize] SKIP — required_runtime already {value}", flush=True)
        return
    # Field offset → value byte: 8 bytes (key_len u64) + key_len bytes
    # of key + 4 bytes (type tag u32) = `8 + len(key) + 4 = 12 + len(key)`.
    key = f"{KV_PREFIX}required_runtime"
    bool_offset = field.offset + 8 + len(key) + 4
    with open(gguf_path, "r+b") as f:
        f.seek(bool_offset)
        old = f.read(1)
        if old not in (b"\x00", b"\x01"):
            raise RuntimeError(f"unexpected byte at required_runtime offset: {old!r}")
        f.seek(bool_offset)
        f.write(b"\x01" if value else b"\x00")
        f.flush()
    # Readback.
    r2 = gguf.GGUFReader(str(gguf_path))
    field2 = r2.get_field(f"{KV_PREFIX}required_runtime")
    new = bool(field2.parts[field2.data[0]][0])
    if new != value:
        raise RuntimeError(f"required_runtime readback failed: wrote {value}, read {new}")
    print(f"[finalize] required_runtime: {current} → {new}", flush=True)
