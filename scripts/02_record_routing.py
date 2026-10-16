"""Record real routing: which experts each token reaches, with what gate.

The system model needs two things a uniform-random router cannot give: how
often a recently used expert is used again (the DRAM expert cache of the NPU
baseline lives on that), and how routed experts collide inside one ADC
sharing group of the 3D stack. Both depend only on the router's choices, so
one forward pass is recorded and every allocator's plane counts are rebuilt
offline from the stored gates.

Saves cache/trace_<tag>.npz with idx (L, T, topk) int16 and gate (L, T, topk).

Two traces are recorded from WikiText-2 (test): the 8192-token measurement trace on which every
cost is scored, and the calibration trace that fixes the sharing groups, which is its first 2048 tokens.

  python scripts/02_record_routing.py --model M --planes P --tag olmoe       --tokens 2048 --start 0
  python scripts/02_record_routing.py --model M --planes P --tag olmoe_long  --tokens 8192 --start 0
"""
import argparse, time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from ocdpa.models import load, tokenizer
from ocdpa.analog import AnalogExperts, PlaneBank
from ocdpa.allocators import Uniform

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--planes", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--scale-bits", type=int, default=8, help="storage width of alpha/beta (log code); 0 keeps fp16")
ap.add_argument("--tokens", type=int, default=2048)
ap.add_argument("--start", type=int, default=0, help="first token of the trace in the tokenised corpus")
ap.add_argument("--home", default="cuda", help="where packed planes live")
ap.add_argument("--window", type=int, default=1024, help="context length per forward")
args = ap.parse_args()


class Recorder(Uniform):
    """Uniform k=4 on the array, keeping each layer's routing as it passes."""
    def __init__(self):
        super().__init__(4)
        self.idx, self.gate = {}, {}

    def prepare(self, layer, top_k_index, top_k_weights):
        self.idx.setdefault(layer, []).append(top_k_index.cpu().numpy().astype(np.int16))
        self.gate.setdefault(layer, []).append(top_k_weights.float().cpu().numpy())


rec = Recorder()
def patch(li, original):
    bank = PlaneBank(Path(args.planes) / f"L{li}.npz", "cuda", home=args.home, scale_bits=(args.scale_bits or None))
    m = AnalogExperts(original, bank, layer=li, adc_bits=8)
    m.policy = rec
    return m

t0 = time.time()
model, cfg = load(args.model, patch=patch)
tok = tokenizer(args.model)
ds = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[:, args.start:args.start + args.tokens]
with torch.no_grad():
    for i in range(0, ids.shape[1], args.window):
        model(ids[:, i:i+args.window].cuda())
        print(f"  window {i//args.window} [{time.time()-t0:.0f}s]", flush=True)
L = sorted(rec.idx)
idx = np.stack([np.concatenate(rec.idx[l]) for l in L])
gate = np.stack([np.concatenate(rec.gate[l]) for l in L])
np.savez_compressed(f"cache/trace_{args.tag}.npz", idx=idx, gate=gate)
print(f"saved cache/trace_{args.tag}.npz idx {idx.shape} [{time.time()-t0:.0f}s]")
