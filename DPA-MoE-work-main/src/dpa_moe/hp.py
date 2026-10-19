"""Hyperparameters dataclass for the chunked AnyBCQ encoder."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class HP:
    base_precision: int = 2
    target_precision: int = 8
    group_size: int = 64
    iters: int = 10000
    num_samples: int = 128
    batch_size: int = 16
    seq_len: int = 2048
    w_lr1: float = 4e-5
    w_lr2: float = 4e-5
    w_lr3: float = 4e-5
    asymmetric: bool = True
    train_beta: bool = True               # anybcq variant only; ss_anybcq keeps β fixed
    any_precision_alpha: bool = False     # False → ss_anybcq layout (one shared α set)
    recon_dtype: str = "float32"          # upstream: "float32" or "bfloat16" only

    @property
    def add_bits(self) -> int:
        return self.target_precision - self.base_precision

    @classmethod
    def from_manifest(cls, d: dict) -> "HP":
        d = dict(d)
        ab_user = d.pop("add_bits", None)
        # PyYAML quirk: a plain `4e-5` literal parses as a STRING (the spec
        # requires `4.0e-5` for scientific floats). Coerce defensively so a
        # silent type bug doesn't surface 3 minutes later inside upstream
        # AnyBCQ's `lr * 1.0`.
        for k in ("w_lr1", "w_lr2", "w_lr3"):
            if k in d and not isinstance(d[k], (int, float)):
                try:
                    d[k] = float(d[k])
                except Exception as e:
                    raise TypeError(f"config encoder.{k} = {d[k]!r} is not numeric") from e
        hp = cls(**d)
        if ab_user is not None and ab_user != hp.add_bits:
            raise ValueError(
                f"config add_bits={ab_user} contradicts "
                f"target_precision - base_precision = {hp.add_bits}")
        if hp.recon_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"recon_dtype must be float32 or bfloat16, got {hp.recon_dtype}")
        return hp
