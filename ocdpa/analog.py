"""Simulated analog array: converters, non-idealities and the expert module.

`AnalogExperts` is a drop-in for a MoE block's `experts`, so the router, the
top-k selection and the residual path stay exactly as the released model
computes them; only the expert matrix-vector products move onto the simulated
array. Precision is chosen per (token, expert) by an allocator
(ocdpa/allocators.py) and truncates the plane loop, which is what the shared
ADC skips cycles for.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ocdpa.planes import GS, KMAX, SCALE_BITS, PlaneBank, log_quant   # noqa: F401  (re-exported)

INPUT_BITS = 8          # bit-serial input: one array read per input bit
TOKEN_CHUNK = 256       # tokens per call of the bit-serial path (memory bound)


def adc(x, bits, full_scale):
    """Column converter. `full_scale` is the reference of the converter; analog_matmul sets it from the largest
    input of the call (an oracle range, not a range calibrated offline), and the inputs are not quantised."""
    if bits is None:
        return x
    step = 2.0 * full_scale / (2 ** bits)
    return torch.clamp(torch.round(x / step), -2 ** (bits - 1), 2 ** (bits - 1) - 1) * step


class NoiseModel:
    """Analog non-idealities of the cells that hold each plane.

    cells  which cell holds plane p: 'mixed' = 1T1R for the two base planes and
           crossbar for planes 3-8 (the proposal), 'xbar' / '1t1r' = one kind
           for all eight planes.
    sigma  Gaussian cell-to-cell variation, relative to the on-state
           conductance, frozen per cell (programming variation: the same cell
           has the same error in every read). One sigma per cell kind.
    ir     'none'; 'raw' = the static column-position gain solved by
           irdrop/mesh.py (calib_irdrop.py), applied before the converter;
           'cal' = the same gain divided by its mean, i.e. a global gain
           trim leaves only the position-dependent part.
    """

    def __init__(self, cells="mixed", sigma_1t1r=0.0, sigma_xbar=0.0, ir="none", seed=0,
                 gains="cache/irdrop_gain.npz"):
        assert cells in ("mixed", "xbar", "1t1r") and ir in ("none", "raw", "cal")
        self.cells, self.ir, self.seed = cells, ir, seed
        self.sig = {"1t1r": sigma_1t1r, "xbar": sigma_xbar}
        self.table = None
        if ir != "none":
            d = np.load(gains)
            self.table = {"1t1r": torch.from_numpy(d["1t1r"]).float(),
                          "xbar": torch.from_numpy(d["xbar"]).float()}
        self._gain = {}

    def kind(self, p):
        return {"xbar": "xbar", "1t1r": "1t1r"}.get(self.cells, "1t1r" if p < 2 else "xbar")

    def sigma(self, p):
        return self.sig[self.kind(p)]

    def gain(self, p, rows, J, device):
        """(rows, J) multiplier on the column sums of plane p, or None."""
        if self.table is None:
            return None
        kind = self.kind(p)
        key = (kind, rows, J, str(device))
        if key not in self._gain:
            t = self.table[kind].to(device)
            r = torch.arange(rows, device=device)
            if kind == "1t1r":
                g = t[0, r % t.shape[1]][:, None].expand(rows, J)
            else:                                   # group j sits in half j % 2 of the 256-row array
                half = torch.arange(J, device=device) % 2
                g = t[half[None, :], (r % t.shape[1])[:, None]]
            if self.ir == "cal":
                g = g / t.mean()
            self._gain[key] = g.contiguous()
        return self._gain[key]


def plane_bits(base, k, schedule=None):
    """Converter resolution for each of the k planes.

    A SAR converter resolves one bit per cycle, so truncating the cycle count
    is a runtime knob on the same hardware, not a second converter. Plane p
    enters the output multiplied by alpha_p, and the encoder sorts alpha
    descending, so the quantisation noise a plane injects is weighted by
    alpha_p^2: the later planes can be resolved more coarsely for the same
    output error. `schedule` gives the bits to shave off plane p.
    """
    if base is None:
        return [None] * k
    if schedule is None:
        return [base] * k
    return [max(2, base - int(schedule[min(p, len(schedule) - 1)])) for p in range(k)]

def quantize_input(x, J, bits=INPUT_BITS, per_group=True):
    """Symmetric integer quantisation of the input: x ~ scale * q, q in [-2^(b-1), 2^(b-1) - 1].

    x: (tokens, J * GS). One scale per group of GS inputs (per_group, the granularity at which the array
    output is rescaled anyway) or one per token. Returns q (tokens, J, GS) and scale (tokens, J, 1)."""
    qmax = 2 ** (bits - 1) - 1
    xg = x.reshape(len(x), J, GS)
    peak = xg.abs().amax(dim=(1, 2), keepdim=True).expand(-1, J, 1) if not per_group else xg.abs().amax(dim=2, keepdim=True)
    scale = (peak / qmax).clamp_min(1e-30)
    q = torch.clamp(torch.round(xg / scale), -qmax - 1, qmax)
    return q, scale


def analog_matmul(x, B, alpha, beta, rows, J, k, *, adc_bits=8, variation=0.0,
                  onoffratio=150.0, counter=None, input_bits=INPUT_BITS,
                  adc_schedule=None, tally=None, noise=None, key=0, serial=True, per_group_input=True):
    """Differential-pair read of the first k planes. x: (tokens, in_features).

    serial=True (default): the path the hardware takes. The input is quantised to `input_bits` bits (symmetric, one scale
    per group of 128 inputs, or per token), applied bit-serially (two's complement), and every input bit is one read of the array: the
    column sum of the 0/1 input against the +/-1 cells of a group, with cell variation and IR-drop gain, goes
    through the ADC, whose range is fixed by the array (128 rows of full-swing cells); the digital shift-and-add
    recombines the bits, then the scales alpha_p and the offset beta are applied.

    serial=False is the earlier idealisation, kept for comparison: the inputs enter at full precision and each
    group's column sum is converted once, with a converter range set by the largest input of the call (a
    data-dependent range; no input quantisation)."""
    if serial and len(x) > TOKEN_CHUNK:        # the input-bit passes are stacked: bound the memory by splitting the tokens
        return torch.cat([analog_matmul(x[i:i + TOKEN_CHUNK], B, alpha, beta, rows, J, k, adc_bits=adc_bits,
                                        variation=variation, onoffratio=onoffratio, counter=counter,
                                        input_bits=input_bits, adc_schedule=adc_schedule, tally=tally,
                                        noise=noise, key=key, serial=True, per_group_input=per_group_input)
                          for i in range(0, len(x), TOKEN_CHUNK)])
    upper, lower = 1.0, 1.0 / onoffratio
    span = upper - lower
    # The pre-ADC column sum is a continuous physical quantity, so it is
    # accumulated in fp32 whatever precision the surrounding model runs at;
    # the only discretisation is the explicit converter below.
    x = x.float()
    padded = J * GS
    if x.shape[1] < padded:
        x = torch.cat([x, x.new_zeros(len(x), padded - x.shape[1])], 1)
    xg = x.reshape(len(x), J, GS)
    a = alpha.reshape(rows, J, KMAX)
    bt = beta.reshape(rows, J)
    Bv = B.reshape(rows, J, GS, KMAX)
    if serial:
        qg, in_scale = quantize_input(x, J, input_bits, per_group_input)
        xg = qg * in_scale                                  # the value the quantised input represents
        full_scale = GS * span                              # fixed by the array: 128 rows of full-swing cells
        qu = (qg.long() + 2 ** input_bits) % (2 ** input_bits)          # two's-complement bit pattern
        bit_weight = [float(2 ** i) for i in range(input_bits - 1)] + [-float(2 ** (input_bits - 1))]
        in_bits = torch.stack([((qu >> i) & 1).float() for i in range(input_bits)], 0).reshape(input_bits * len(x), J, GS)
        bit_weight = torch.tensor(bit_weight, device=x.device).view(input_bits, 1, 1, 1)
    else:
        full_scale = xg.abs().sum(-1).max() * span
    out = torch.einsum("bj,rj->br", xg.sum(-1), bt)
    bits = plane_bits(adc_bits, k, adc_schedule)
    for p in range(k):
        plane = Bv[:, :, :, p].float()
        if variation > 0:
            g = plane * span
            g = g + g.abs() * torch.randn_like(g) * variation
        else:
            g = plane * span
        if noise is not None and noise.sigma(p) > 0:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(noise.seed * 1_000_003 + key * 8 + p)
            g = g + noise.sigma(p) * upper * torch.randn(plane.shape, generator=gen, device=x.device)
        gain = noise.gain(p, rows, J, x.device) if noise is not None else None
        if serial:
            # one array read and one conversion per input bit; the bits are stacked along the token axis
            pass_sum = torch.einsum("bjc,rjc->brj", in_bits, g).view(input_bits, len(x), rows, J)
            if gain is not None:
                pass_sum = pass_sum * gain[None, None]
            partial = (bit_weight * (adc(pass_sum, bits[p], full_scale) / span)).sum(0)
            partial = partial * in_scale.squeeze(-1).unsqueeze(1)          # (tokens, 1, groups)
        else:
            partial = torch.einsum("bjc,rjc->brj", xg, g)
            if gain is not None:
                partial = partial * gain[None]
            partial = adc(partial, bits[p], full_scale) / span
        out = out + torch.einsum("brj,rj->br", partial, a[:, :, p])
        if tally is not None:
            tally[bits[p]] = tally.get(bits[p], 0) + rows * J * len(x) * input_bits
    if counter is not None:
        # The input is applied bit-serially, so a group is converted once per
        # input bit: NeuroSim's numReadPulse = numBitInput.
        counter[0] += rows * J * k * len(x) * input_bits
    return out


NATIVE = 100   # a plan value k + NATIVE reads k planes of the native-4-bit copy


class AnalogExperts(nn.Module):
    def __init__(self, original, bank, *, layer=0, adc_bits=8, variation=0.0,
                 onoffratio=150.0, native_bank=None):
        super().__init__()
        self.num_experts = original.num_experts
        self.act_fn = original.act_fn
        self.bank = bank
        self.native_bank = native_bank   # separately quantised 4-bit copy (policy k >= NATIVE)
        self.layer = layer
        self.adc_bits, self.variation, self.onoffratio = adc_bits, variation, onoffratio
        self.adc_schedule = None      # bits to shave off plane p; None = flat
        self.tally = {}               # conversions keyed by converter resolution
        self.noise = None             # NoiseModel or None
        self.serial = True            # bit-serial, 8-bit input path of the hardware (False: full-precision inputs)
        self.per_group_input = True   # input scale per group of 128 inputs (False: per token)
        self.policy = None        # allocator: (layer, expert, tokens, pos) -> k tensor
        self.counter = [0]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        if hasattr(self.policy, "prepare"):
            self.policy.prepare(self.layer, top_k_index, top_k_weights)
        opts = dict(adc_bits=self.adc_bits, variation=self.variation,
                    onoffratio=self.onoffratio, counter=self.counter,
                    adc_schedule=self.adc_schedule, tally=self.tally, noise=self.noise, serial=self.serial, per_group_input=self.per_group_input)
        for e in hit:
            e = int(e[0])
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            ks = self.policy(self.layer, e, tok, pos, top_k_weights, None)
            for k in torch.unique(ks).tolist():
                sel = ks == k
                t, p = tok[sel], pos[sel]
                state = hidden_states[t]
                bank = self.native_bank if k >= NATIVE else self.bank
                k = k - NATIVE if k >= NATIVE else k
                Bg, ag, bg, rg, mg, Jg = bank.get(e, "gate_proj")
                Bu, au, bu, ru, mu, Ju = bank.get(e, "up_proj")
                Bd, ad, bd, rd, md, Jd = bank.get(e, "down_proj")
                key = (self.layer * 4096 + e) * 3
                gate = analog_matmul(state, Bg, ag, bg, rg, Jg, k, key=key, **opts)
                up = analog_matmul(state, Bu, au, bu, ru, Ju, k, key=key + 1, **opts)
                mid = self.act_fn(gate) * up
                outp = analog_matmul(mid, Bd, ad, bd, rd, Jd, k, key=key + 2, **opts)
                outp = outp[:, :hidden_states.shape[1]] * top_k_weights[t, p, None]
                final.index_add_(0, t, outp.to(final.dtype))
        return final
