"""Offline residual table R[layer, expert, k], the gate-free half of DPA.

Captured against the released full-precision expert weights, read back from
the checkpoint one layer at a time so the model and the planes never have to
be resident together.

R[l, e, k] = E_x || E_e^(k)(x) - E_e(x) ||^2 over the tokens the router
actually sent to expert e, matching the paper's definition.
"""
import argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from ocdpa.models import load_layer
from ocdpa.analog import PlaneBank, analog_matmul

KMIN, KMAX = 2, 8
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0125")
ap.add_argument("--planes", default="cache/planes_olmoe")
ap.add_argument("--tokens", type=int, default=2048)
ap.add_argument("--scale-bits", type=int, default=8, help="storage width of alpha/beta (log code); 0 keeps fp16")
ap.add_argument("--adc-bits", type=int, default=0)
ap.add_argument("--variation", type=float, default=0.0)
ap.add_argument("--out", default="cache/residual_olmoe.npy")
args = ap.parse_args()
bits = None if args.adc_bits == 0 else args.adc_bits

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
cfg = model.config
from datasets import load_dataset
ds = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")
ids = tok("\n\n".join(ds["text"][:3000]), return_tensors="pt").input_ids[:, :args.tokens].cuda()

cap = {}
hooks = []
for li, layer in enumerate(model.model.layers):
    def mk(li):
        def f(mod, inp, out):
            cap[li] = inp[0].detach().reshape(-1, cfg.hidden_size).float().cpu()
        return f
    hooks.append(layer.mlp.experts.register_forward_pre_hook(
        lambda m, i, li=li: cap.__setitem__(li, i[0].detach().reshape(-1, cfg.hidden_size).float().cpu())))
    hooks.append(layer.mlp.gate.register_forward_hook(
        lambda m, i, o, li=li: cap.__setitem__(("r", li), o[2].detach().cpu())))
with torch.no_grad():
    model(ids)
for h in hooks:
    h.remove()
del model
torch.cuda.empty_cache()
print(f"captured {len(ids[0])} tokens for {cfg.num_hidden_layers} layers", flush=True)

R = np.zeros((cfg.num_hidden_layers, cfg.num_experts, KMAX + 1), dtype=np.float64)
for li in range(cfg.num_hidden_layers):
    X = cap[li].cuda()
    routed = cap[("r", li)].cuda()
    bank = PlaneBank(Path(args.planes) / f"L{li}.npz", "cuda", scale_bits=(args.scale_bits or None))
    weights = load_layer(args.model, li)
    for e in range(cfg.num_experts):
        sel = (routed == e).any(-1).nonzero().squeeze(-1)
        if len(sel) == 0:
            continue
        x = X[sel]
        wg = weights[f"model.layers.{li}.mlp.experts.{e}.gate_proj.weight"].float().cuda()
        wu = weights[f"model.layers.{li}.mlp.experts.{e}.up_proj.weight"].float().cuda()
        wd = weights[f"model.layers.{li}.mlp.experts.{e}.down_proj.weight"].float().cuda()
        exact = (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T
        Bg, ag, bg, rg, mg, Jg = bank.get(e, "gate_proj")
        Bu, au, bu, ru, mu, Ju = bank.get(e, "up_proj")
        Bd, ad, bd, rd, md, Jd = bank.get(e, "down_proj")
        o = dict(adc_bits=bits, variation=args.variation, serial=False)
        for k in range(KMIN, KMAX + 1):
            g = analog_matmul(x, Bg, ag, bg, rg, Jg, k, **o)
            u = analog_matmul(x, Bu, au, bu, ru, Ju, k, **o)
            y = analog_matmul(F.silu(g) * u, Bd, ad, bd, rd, Jd, k, **o)[:, :cfg.hidden_size]
            R[li, e, k] = float(((y - exact) ** 2).mean())
        del wg, wu, wd, exact
    del bank, X
    torch.cuda.empty_cache()
    print(f"  L{li} done", flush=True)

for k in range(KMAX - 1, KMIN - 1, -1):
    R[:, :, k] = np.maximum(R[:, :, k], R[:, :, k + 1])
R[:, :, :KMIN] = R[:, :, KMIN][:, :, None]
np.save(args.out, R)
print(f"saved {args.out}  shape {R.shape}")
