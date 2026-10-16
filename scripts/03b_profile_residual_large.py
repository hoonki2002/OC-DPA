"""Residual table R[layer, expert, k] for a model too large to hold resident.

Same definition as profile_residual.py -- the mean squared gap between the
k-plane analog expert and the released full-precision expert, over the tokens
the router actually sent to that expert -- but the skeleton carries no expert
weights, the planes live in host memory, and the reference weights are read
from the checkpoint one expert at a time.

Because the planes are prefix-decodable the gate and up projections are
accumulated plane by plane and read out at every k, so only the down
projection is recomputed per k.
"""
import argparse, time
from pathlib import Path

import numpy as np, torch

from ocdpa.models import load, tokenizer
from ocdpa.analog import PlaneBank, analog_matmul, adc, GS, KMAX
from ocdpa.fp_experts import FPExperts

KMIN = 2
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
ap.add_argument("--planes", default="cache/planes_qwen3")
ap.add_argument("--tokens", type=int, default=2048)
ap.add_argument("--scale-bits", type=int, default=8, help="storage width of alpha/beta (log code); 0 keeps fp16")
ap.add_argument("--adc-bits", type=int, default=8)
ap.add_argument("--variation", type=float, default=0.0)
ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (0 = all)")
ap.add_argument("--out", default="cache/residual_qwen3.npy")
args = ap.parse_args()
bits = None if args.adc_bits == 0 else args.adc_bits


def prefix_matmul(x, B, alpha, beta, rows, J, *, adc_bits, variation, onoffratio=150.0):
    """Outputs for every k in 1..KMAX from a single pass over the planes."""
    span = 1.0 - 1.0 / onoffratio
    x = x.float()
    padded = J * GS
    if x.shape[1] < padded:
        x = torch.cat([x, x.new_zeros(len(x), padded - x.shape[1])], 1)
    xg = x.reshape(len(x), J, GS)
    a = alpha.reshape(rows, J, KMAX)
    Bv = B.reshape(rows, J, GS, KMAX)
    full_scale = xg.abs().sum(-1).max() * span
    out = torch.einsum("bj,rj->br", xg.sum(-1), beta.reshape(rows, J))
    outs = []
    for p in range(KMAX):
        g = Bv[:, :, :, p].float() * span
        if variation > 0:
            g = g + g.abs() * torch.randn_like(g) * variation
        partial = torch.einsum("bjc,rjc->brj", xg, g)
        partial = adc(partial, adc_bits, full_scale) / span
        out = out + torch.einsum("brj,rj->br", partial, a[:, :, p])
        outs.append(out.clone())
    return outs


t0 = time.time()
model, cfg = load(args.model, patch=lambda li, o: FPExperts(
    args.model, li, o.num_experts, o.act_fn))
L, E, H = cfg.num_hidden_layers, cfg.num_experts, cfg.hidden_size

from datasets import load_dataset
tok = tokenizer(args.model)
ds = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")
ids = tok("\n\n".join(ds["text"][:3000]), return_tensors="pt").input_ids[:, :args.tokens].cuda()

cap = {}
moe = [li for li, layer in enumerate(model.model.layers) if hasattr(layer.mlp, "experts")]
hooks = [model.model.layers[li].mlp.experts.register_forward_pre_hook(
    (lambda li: (lambda m, inp: cap.__setitem__(
        li, (inp[0].detach().reshape(-1, H).float().cpu(), inp[1].detach().cpu()))))(li))
    for li in moe]
with torch.no_grad():
    model(ids)
for h in hooks:
    h.remove()
print(f"captured {ids.shape[1]} tokens for {L} layers [{time.time()-t0:.0f}s]", flush=True)

R = np.zeros((L, E, KMAX + 1), dtype=np.float64)
act = model.model.layers[moe[0]].mlp.experts.act_fn
done = args.layers or L
for li in [l for l in moe if l < done]:
    X, idx = cap[li]
    X = X.cuda()
    bank = PlaneBank(Path(args.planes) / f"L{li}.npz", "cuda", home="cpu", scale_bits=(args.scale_bits or None))
    fp = FPExperts(args.model, li, E, act)
    for e in range(E):
        sel = (idx == e).any(-1).nonzero().squeeze(-1).cuda()
        if len(sel) == 0:
            continue
        x = X[sel]
        w = fp._weights(e)
        exact = (act(x @ w["gate_proj"].T) * (x @ w["up_proj"].T)) @ w["down_proj"].T
        del w
        o = dict(adc_bits=bits, variation=args.variation)
        Bg, ag, bg, rg, mg, Jg = bank.get(e, "gate_proj")
        gs = prefix_matmul(x, Bg, ag, bg, rg, Jg, **o)
        del Bg
        Bu, au, bu, ru, mu, Ju = bank.get(e, "up_proj")
        us = prefix_matmul(x, Bu, au, bu, ru, Ju, **o)
        del Bu
        Bd, ad, bd, rd, md, Jd = bank.get(e, "down_proj")
        for k in range(KMIN, KMAX + 1):
            mid = act(gs[k - 1][:, :mg]) * us[k - 1][:, :mu]
            y = analog_matmul(mid, Bd, ad, bd, rd, Jd, k, **o)[:, :H]
            R[li, e, k] = float(((y - exact) ** 2).mean())
        del Bd, gs, us, exact
    del bank, fp, X
    cap[li] = None
    torch.cuda.empty_cache()
    print(f"  L{li} done [{time.time()-t0:.0f}s]", flush=True)

# Experts the router never reached carry no measurement; give them the layer's
# median curve so the allocator cannot treat them as free.
seen = R[:, :, KMAX] > 0
for li in [l for l in moe if l < done]:
    if seen[li].any() and not seen[li].all():
        R[li, ~seen[li]] = np.median(R[li, seen[li]], axis=0)

for k in range(KMAX - 1, KMIN - 1, -1):
    R[:, :, k] = np.maximum(R[:, :, k], R[:, :, k + 1])
R[:, :, :KMIN] = R[:, :, KMIN][:, :, None]
np.save(args.out, R)
print(f"saved {args.out}  shape {R.shape}")
