"""Encode every routed-expert matrix of a model once and cache the planes as packed bits.

Each matrix is encoded with DPA-MoE's SharedScale AnyBCQ optimizer (ocdpa/encoder.py;
set DPAMOE_DIR to a checkout of DPA-MoE) into 8 sign planes B (+/-1, one bit per
entry: 6.4 GB packed for OLMoE instead of 51 GB as int8) and the scales alpha
(8 per group of 128 inputs) and beta (1 per group), written as fp16. The scales
are rounded to their 8-bit logarithmic code when the planes are loaded
(ocdpa/planes.py), so this file does not fix the storage width.

  python scripts/01_encode_experts.py --model allenai/OLMoE-1B-7B-0125 --out cache/planes_olmoe
  python scripts/01_encode_experts.py --model deepseek-ai/deepseek-moe-16b-base --out cache/planes_deepseek
"""
import argparse, time, json
from pathlib import Path
import numpy as np, torch

from ocdpa.encoder import compute_codebook
from ocdpa.models import load_layer, moe_layers

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0125", help="Hugging Face id of an already-downloaded MoE checkpoint")
ap.add_argument("--out", default="cache/planes_olmoe")
args = ap.parse_args()
MODEL = args.model
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
GS, KMAX = 128, 8
MATS = ("gate_proj", "up_proj", "down_proj")

layers = moe_layers(MODEL)
print(f"{len(layers)} MoE layers", flush=True)
t0 = time.time()
meta = {}
for li in layers:
    dest = OUT / f"L{li}.npz"
    if dest.exists():
        print(f"  L{li} cached", flush=True); continue
    tensors = load_layer(MODEL, li)
    store = {}
    # `.mlp.experts.<id>.` only: DeepSeek also carries `.mlp.shared_experts.`,
    # which has no expert index and must not inflate the count.
    marker = f"model.layers.{li}.mlp.experts."
    n_exp = len({k[len(marker):].split(".")[0] for k in tensors if k.startswith(marker)})
    for e in range(n_exp):
        for mat in MATS:
            key = f"model.layers.{li}.mlp.experts.{e}.{mat}.weight"
            if key not in tensors:
                continue
            W = tensors[key].float().cuda()
            B, alpha, beta, n, m, J = compute_codebook(W, base_precision=2,
                                                       target_precision=KMAX, group_size=GS)
            bits = ((B > 0).to(torch.uint8).cpu().numpy().reshape(-1))
            store[f"{e}.{mat}.B"] = np.packbits(bits)
            store[f"{e}.{mat}.a"] = alpha.half().cpu().numpy()
            store[f"{e}.{mat}.b"] = beta.half().cpu().numpy()
            store[f"{e}.{mat}.s"] = np.array([n, m, J], dtype=np.int32)
            del W, B, alpha, beta
        torch.cuda.empty_cache()
    np.savez(dest, **store)
    meta[li] = n_exp
    print(f"  L{li}: {n_exp} experts x {len(MATS)} matrices  "
          f"{dest.stat().st_size/1e6:.0f} MB  [{time.time()-t0:.0f}s]", flush=True)
json.dump(meta, open(OUT / "meta.json", "w"))
print(f"done in {time.time()-t0:.0f}s")
