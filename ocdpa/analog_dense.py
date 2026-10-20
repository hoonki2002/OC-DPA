"""All-analog variant of a model's non-expert linear layers (attention q/k/v/o,
router, lm_head, and the always-on shared experts of DeepSeek-MoE) with the conventional AIMC mapping: uniform 8-bit weights,
one binary plane per bit (weight 2^-p), one scale per (row, group of 128).
No learned per-plane scales and no offset: only experts need the any-precision
encoding, because only they are read at variable precision.

The reads go through the same analog_matmul as the experts (8-bit ADC, IR-drop
gain, cell variation), all eight planes."""
import torch, torch.nn as nn
from ocdpa.analog import analog_matmul
from ocdpa.planes import _unpack, GS, KMAX, log_quant


class Holder:
    noise = None
    serial = True
    per_group_input = True


def _pack(bits):
    n = bits.numel()
    pad = (-n) % 8
    if pad:
        bits = torch.cat([bits, bits.new_zeros(pad)])
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=bits.device)
    return (bits.view(-1, 8) * w).sum(1).to(torch.uint8)


class IntBank:
    def __init__(self, items, device="cuda", scale_bits=None):
        """items: {ident: weight (n x m)}. Offline step, run once. `scale_bits`: storage width of the per-group scales as a
        logarithmic code, like the scales of the experts (None keeps them exact)."""
        self.device, self.packed, self.scales, self.shapes, self._planes = device, {}, {}, {}, {}
        for ident, W in items.items():
            n, m = W.shape
            J = (m + GS - 1) // GS
            packs, scs = [], []
            for r0 in range(0, n, 2048):
                Wc = W[r0:r0 + 2048].float().to(device)
                Wg = torch.zeros(Wc.shape[0], J * GS, device=device)
                Wg[:, :m] = Wc
                Wg = Wg.view(-1, J, GS)
                A = (Wg.abs().amax(-1) / (1 - 2.0 ** -KMAX)).clamp_min(1e-12)
                idx = torch.clamp(torch.round(128.0 * Wg / A[..., None] + 127.5), 0, 255).to(torch.uint8)
                bits = torch.stack([(idx >> (KMAX - p)) & 1 for p in range(1, KMAX + 1)], -1)
                packs.append(_pack(bits.reshape(-1)))
                p_ = torch.arange(1, KMAX + 1, device=device, dtype=torch.float32)
                scs.append((A[..., None] * 2.0 ** (-p_)).contiguous())
                del Wc, Wg, idx, bits
            self.packed[ident] = torch.cat(packs)
            self.scales[ident] = log_quant(torch.cat(scs), scale_bits, False)
            self.shapes[ident] = (n, m, J)

    def get(self, key, mat):
        ident = (key, mat)
        n, m, J = self.shapes[ident]
        if ident not in self._planes:
            self._planes[ident] = _unpack(self.packed[ident], n * J * GS * KMAX, self.device).view(n * J, GS, KMAX)
        return self._planes[ident], self.scales[ident], torch.zeros(n, J, device=self.device), n, m, J


class AnalogLinear(nn.Module):
    def __init__(self, bank, ident, holder, key):
        super().__init__()
        self.bank, self.ident, self.holder, self.key = bank, ident, holder, key

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        B, a, b, n, m, J = self.bank.get(*self.ident)
        step = 512
        outs = []
        Bv = B.view(n, J, GS, KMAX)
        for ci, r0 in enumerate(range(0, n, step)):
            r1 = min(n, r0 + step)
            outs.append(analog_matmul(x2, Bv[r0:r1].reshape((r1 - r0) * J, GS, KMAX), a[r0:r1], b[r0:r1],
                                      r1 - r0, J, KMAX, adc_bits=8, noise=self.holder.noise,
                                      key=self.key + 7919 * ci, serial=self.holder.serial,
                                      per_group_input=self.holder.per_group_input))
        out = torch.cat(outs, 1)
        return out[:, :n].reshape(*shp[:-1], n).to(x.dtype)


class AnalogRouter(nn.Module):
    """OlmoeTopKRouter with its logits computed on the arrays."""
    def __init__(self, orig, lin):
        super().__init__()
        self.top_k, self.norm, self.hidden_dim, self.lin = orig.top_k, orig.norm_topk_prob, orig.hidden_dim, lin

    def forward(self, hidden_states):
        logits = self.lin(hidden_states.reshape(-1, self.hidden_dim))
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        top, idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm:
            top = top / top.sum(dim=-1, keepdim=True)
        return logits, top.to(logits.dtype), idx


def build(model, which, holder, scale_bits=None):
    """-> list of (parent, attribute, original, analog); `which` is a subset of
    {'attn', 'router', 'head', 'shared'}."""
    items, plan = {}, []
    for li, layer in enumerate(model.model.layers):
        if "attn" in which:
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                items[(li, nm)] = getattr(layer.self_attn, nm).weight.data
                plan.append((layer.self_attn, nm, (li, nm)))
        if "router" in which and hasattr(layer.mlp, "gate"):
            items[(li, "gate")] = layer.mlp.gate.weight.data
            plan.append((layer.mlp, "gate", (li, "gate")))
        if "shared" in which and hasattr(layer.mlp, "shared_expert"):
            for nm in ("gate_proj", "up_proj", "down_proj"):
                items[(li, "shared_" + nm)] = getattr(layer.mlp.shared_expert, nm).weight.data
                plan.append((layer.mlp.shared_expert, nm, (li, "shared_" + nm)))
    if "head" in which:
        items[(99, "lm_head")] = model.lm_head.weight.data
        plan.append((model, "lm_head", (99, "lm_head")))
    bank = IntBank(items, scale_bits=scale_bits)
    key = 1_000_000
    out = []
    for parent, nm, ident in plan:
        lin = AnalogLinear(bank, ident, holder, key)
        orig = getattr(parent, nm)
        out.append((parent, nm, orig, AnalogRouter(orig, lin) if nm == "gate" else lin))
        key += 1
    return out
