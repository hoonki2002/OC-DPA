"""How an expert matrix is stored on the array: binary planes and 8-bit scales.

An expert matrix is encoded (scripts/01_encode_experts.py) as

    W^(k) ~ beta + sum_{p=1..k} alpha_p * S_p ,   S_p in {-1, +1},

per output row and group of 128 inputs. The signs S_p are the bit-planes
(one bit per cell); alpha_p and beta are stored beside them as 8-bit
logarithmic codes (`log_quant`), the format the recombination unit reads. The
first k planes always form a valid k-bit model, so the number of planes read
is a run-time knob.
"""
from __future__ import annotations

import numpy as np
import torch

GS, KMAX = 128, 8          # inputs per scale group, planes per weight
SCALE_BITS = 8             # storage width of alpha and beta (logarithmic code)


def log_quant(x, bits, signed):
    """Store x in `bits` bits as a logarithmic code: a sign bit (if `signed`)
    and a level index at a fixed step in log2 covering 16 octaves below the
    matrix maximum (one shared exponent per matrix). bits = None keeps x as is
    (fp16 reference); bits = 0 drops the term."""
    if bits is None:
        return x
    if bits == 0:
        return torch.zeros_like(x)
    levels = 2 ** (bits - 1 if signed else bits)
    step = 16.0 / levels
    mag = x.abs().clamp_min(1e-30)
    emax = torch.ceil(torch.log2(mag.max()))
    q = torch.round((emax - torch.log2(mag)) / step).clamp(0, levels - 1)
    return torch.sign(x) * torch.exp2(emax - q * step)


_MASKS = None


def _unpack(packed, count, device):
    """Bits to +/-1 on the GPU. MSB first, matching numpy.packbits."""
    global _MASKS
    if _MASKS is None or _MASKS.device != packed.device:
        _MASKS = (1 << torch.arange(7, -1, -1, dtype=torch.uint8, device=packed.device))
    bits = (packed.unsqueeze(-1) & _MASKS).ne(0).reshape(-1)[:count]
    return bits.to(torch.int8) * 2 - 1


class PlaneBank:
    """One layer's planes, held packed on the device and unpacked on demand.

    A layer is 64 experts x 3 matrices x 16.8 M signs. Unpacked to fp32 that is
    12.9 GB, so the bits stay packed (403 MB) and each expert is expanded only
    while it is being read.
    """

    def __init__(self, path, device, home=None, scale_bits=SCALE_BITS):
        """`home` is where the packed bits live; `device` is where they are
        unpacked (for a 16 B model the planes are 16 GB, so they stay in host
        memory and only the expert being read crosses the bus). `scale_bits`
        is the storage width of alpha and beta; None keeps them at fp16."""
        store = np.load(path)
        self.device = device
        home = home or device
        self.packed, self.scales, self.shapes = {}, {}, {}
        for key in store.files:
            expert, mat, kind = key.split(".")
            ident = (int(expert), mat)
            if kind == "B":
                self.packed[ident] = torch.from_numpy(store[key]).to(home)
            elif kind == "s":
                self.shapes[ident] = tuple(int(v) for v in store[key])
            else:                                   # "a" = alpha, "b" = beta
                scale = torch.from_numpy(store[key]).to(device).float()
                self.scales.setdefault(ident, {})[kind] = log_quant(scale, scale_bits, kind == "b")
        store.close()

    def get(self, expert, mat):
        ident = (expert, mat)
        n, m, J = self.shapes[ident]
        B = _unpack(self.packed[ident].to(self.device, non_blocking=True),
                    n * J * GS * KMAX, self.device)
        B = B.view(n * J, GS, KMAX)
        s = self.scales[ident]
        return B, s["a"], s["b"], n, m, J
