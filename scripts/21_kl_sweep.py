"""Phase-aware precision: a cheap prefill, a full-precision decode.

On the proposal the decode step's experts cost ~0.3 % of the token, so decode
should simply run every plane; prefill is where experts set TTFT and the plane
budget pays. The open question is what a low-budget prefill does to the
answer: the prompt's KV cache is then built from approximated experts, and
every decoded token attends to it.

Measured with teacher forcing on windows of P prompt + D continuation tokens.
One causal forward with a per-position plane plan is exactly prefill-then-
decode: positions < P take the prefill plan (budget kbar_p, uniform or DPA),
positions >= P take k = 8, and each continuation position sees only the
prompt's KV and the continuation before it. KL divergence against the
released full-precision experts is reported on the continuation positions
alone -- the tokens the user reads -- and on the prompt positions separately.
"""
import argparse, json, time
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset

from ocdpa.models import load, tokenizer
from ocdpa.analog import AnalogExperts, PlaneBank, NoiseModel, NATIVE
from ocdpa.allocators import DPA
from ocdpa.fp_experts import FPExperts

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--planes", required=True)
ap.add_argument("--residual", required=True)
ap.add_argument("--home", default="cuda")
ap.add_argument("--prompt", type=int, default=1024)
ap.add_argument("--cont", type=int, default=256)
ap.add_argument("--windows", type=int, default=4)
ap.add_argument("--out", required=True)
ap.add_argument("--sweep", default="", help="comma list of prefill budgets; runs uniform and DPA at each, decode at 8")
ap.add_argument("--noise", default="", help="semicolon-separated scenarios 'cells,sigma_1t1r,sigma_xbar,ir' "
                "(cells mixed|xbar|1t1r, ir none|raw|cal); each is run at --budgets, decode at 8")
ap.add_argument("--budgets", default="4")
ap.add_argument("--decode-same", action="store_true",
                help="also run each budget in the decode phase (k-bar throughout, no switch to k = 8): a fixed-precision array")
ap.add_argument("--native-planes", default="", help="separately quantised 4-bit copy (scripts/13): adds the rows 'native 4' (4 bits throughout) and 'native 4 + 8' (4-bit prefill copy, 8-bit decode copy)")
ap.add_argument("--native-only", action="store_true", help="with --native-planes: only the fixed-precision rows; --planes is then the native 8-bit copy")
ap.add_argument("--ideal-inputs", action="store_true", help="earlier idealisation: full-precision inputs, one conversion per group (default: 8-bit inputs applied bit-serially, one conversion per input bit)")
ap.add_argument("--input-per-token", action="store_true", help="one input scale per token instead of per group of 128")
ap.add_argument("--adc-bits", type=int, default=8, help="ADC resolution (0: ideal converter)")
ap.add_argument("--skip-all8", action="store_true", help="skip the all-planes reference row (it is measured by another run of the same planes)")
ap.add_argument("--no-dpa", action="store_true", help="skip the OC-DPA rows (e.g. the top-k planes of a native integer copy, which has no residual table)")
ap.add_argument("--no-uniform", action="store_true", help="skip the uniform-allocation rows")
ap.add_argument("--all8-only", action="store_true", help="only the all-plane reference row (diagnostics)")
ap.add_argument("--scale-bits", type=int, default=8, help="storage width of alpha/beta (log code); 0 keeps fp16")
ap.add_argument("--dense-scale-bits", type=int, default=0, help="with --dense: storage width of the scales of the non-expert matrices (log code; 0 keeps them exact)")
ap.add_argument("--dense", default="", help="non-expert layers run on AIMC (conventional 8-bit slicing): attn,router,head,shared (shared = DeepSeek's always-on experts, which sit in the stack)")
args = ap.parse_args()


class Phase:
    """Prefill plan for positions < P, a fixed k for the rest."""

    def __init__(self, kbar_p, dpa, P, k_decode, residual, topk):
        self.kbar, self.P, self.kd = kbar_p, P, k_decode
        self.inner = DPA(kbar_p, residual, topk) if dpa else None
        self.plan = {}

    def prepare(self, layer, idx, w):
        if self.inner is not None:
            self.inner.prepare(layer, idx, w)
            plan = self.inner.plan[layer].clone()
        else:
            plan = torch.full(idx.shape, self.kbar, dtype=torch.long, device=idx.device)
        if self.kd is not None:
            plan[self.P:] = self.kd
        self.plan[layer] = plan

    def __call__(self, layer, expert, tokens, pos, gates, residual):
        return self.plan[layer][tokens, pos]


t0 = time.time()
banks = {}
def patch(li, original):
    banks[li] = PlaneBank(Path(args.planes) / f"L{li}.npz", "cuda", home=args.home, scale_bits=(args.scale_bits or None))
    nb = (PlaneBank(Path(args.native_planes) / f"L{li}.npz", "cuda", home=args.home, scale_bits=(args.scale_bits or None))
          if args.native_planes else None)
    return AnalogExperts(original, banks[li], layer=li, adc_bits=(args.adc_bits or None), native_bank=nb)

model, cfg = load(args.model, patch=patch)
analog = {li: layer.mlp.experts for li, layer in enumerate(model.model.layers)
          if hasattr(layer.mlp, "experts")}
topk = cfg.num_experts_per_tok
R = np.load(args.residual)

tok = tokenizer(args.model)
ds = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
W = args.prompt + args.cont
# Windows start well past the trace's tokens so the two experiments do not
# share text.
start = 16384
windows = [ids[:, start + i * W: start + (i + 1) * W].cuda() for i in range(args.windows)]

for li in analog:
    model.model.layers[li].mlp.experts = FPExperts(args.model, li, cfg.num_experts, analog[li].act_fn)
refs = []
with torch.no_grad():
    for w in windows:
        refs.append(F.log_softmax(model(w).logits.float(), -1).cpu())
for li in analog:
    model.model.layers[li].mlp.experts = analog[li]
print(f"reference: {len(windows)} x {W} tokens [{time.time()-t0:.0f}s]", flush=True)

P = args.prompt
def run(policy):
    for a in analog.values():
        a.policy = policy
    kp = kc = 0.0
    per_window = []                                    # cont. KL of every window, for confidence intervals
    with torch.no_grad():
        for w, r in zip(windows, refs):
            out = F.log_softmax(model(w).logits.float(), -1)[0]
            kl = F.kl_div(out, r[0].cuda(), log_target=True, reduction="none").sum(-1)
            kp += float(kl[:P].sum()); kc += float(kl[P:].sum())
            per_window.append(float(kl[P:].mean()))
    run.windows = per_window
    n = len(windows)
    return kp / (n * P), kc / (n * (W - P))

HOLD = None
if args.dense:
    from ocdpa.analog_dense import build as build_dense, Holder
    HOLD = Holder()
    for parent, nm_, orig_, ana_ in build_dense(model, set(args.dense.split(",")), HOLD, scale_bits=(args.dense_scale_bits or None)):
        setattr(parent, nm_, ana_)

if args.noise:
    scenarios = [x.strip() for x in args.noise.split(";") if x.strip()]
    configs = [] if args.skip_all8 else [(8, False, None)]
    for kb in ([] if args.all8_only else [int(x) for x in args.budgets.split(",")]):
        configs += ([] if args.no_uniform else [(kb, False, 8)]) + ([] if args.no_dpa else [(kb, True, 8)])
        if args.decode_same:
            configs += [(kb, False, None)]
    if args.native_planes:
        native = [(NATIVE + 4, False, NATIVE + 4), (NATIVE + 4, False, 8)]
        configs = ([] if args.skip_all8 else [(8, False, None)]) + native if args.native_only else configs + native
    res = []
    print(f"\n  {'scenario':>34} {'prefill':>10} | {'KLD prompt':>10} {'KLD cont.':>10}", flush=True)
    for sc in scenarios:
        parts = [v.strip() for v in sc.split(",")]
        c, s1, sx, ir = parts[:4]
        nm = None if sc.startswith("none") else NoiseModel(c, float(s1), float(sx), ir)
        for a_ in analog.values():
            a_.noise = nm
            a_.serial = not args.ideal_inputs
            a_.per_group_input = not args.input_per_token
        if HOLD is not None:
            HOLD.noise = nm
            HOLD.serial = not args.ideal_inputs
            HOLD.per_group_input = not args.input_per_token
        for kb, dpa, kd in configs:
            kp, kc = run(Phase(kb, dpa, P, kd, R, topk))
            name = ("native 4" if kd == kb else "native 4 + 8") if kb >= NATIVE else "all 8" if (kd is None and kb == 8) else f"{'DPA' if dpa else 'uni'} {kb}" + (" same" if kd is None else "")
            res.append(dict(scenario=sc, prefill=name, kbar_p=kb, dpa=dpa, kld_prompt=kp, kld_cont=kc, kld_cont_windows=run.windows))
            print(f"  {sc:>34} {name:>10} | {kp:10.5f} {kc:10.5f}   [{time.time()-t0:.0f}s]", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
    raise SystemExit

configs = [(8, False, None)]                                    # all planes throughout
if args.sweep:
    for kb in [int(x) for x in args.sweep.split(",")]:
        configs += [(kb, False, 8), (kb, True, 8)]
else:
    for kb in (3, 4, 5):
        configs += [(kb, False, 8), (kb, True, 8),               # cheap prefill, full decode
                    (kb, True, None)]                            # same budget throughout
res = []
print(f"\n  {'prefill':>12} {'decode':>7} | {'KLD prompt':>10} {'KLD cont.':>10}", flush=True)
for kb, dpa, kd in configs:
    kp, kc = run(Phase(kb, dpa, P, kd, R, topk))
    name = f"{'DPA' if dpa else 'uni'} k={kb}"
    dec = "8" if kd == 8 else ("same" if kd is None else str(kd))
    res.append(dict(prefill=name, kbar_p=kb, dpa=dpa, decode=dec, kld_prompt=kp, kld_cont=kc, kld_cont_windows=run.windows))
    print(f"  {name:>12} {dec:>7} | {kp:10.5f} {kc:10.5f}   [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
