"""Fixed-precision (native) copies of every routed expert, quantised with GPTQ (4 and 8 bit, group 128).

The 'native 4-bit / 8-bit AIMC' of the precision-alternative comparison: each matrix is quantised on
its own with GPTQ (Frantar et al., 2023), not read as the first planes of the any-precision code.
A b-bit integer q = sum_j 2^j b_j is exactly b sign planes,

    w ~ lo + s*q = (lo + (2^b - 1)/2 * s) + sum_{p=0..b-1} (s * 2^(b-2-p)) * S_p ,  S_p = 2 b_(b-1-p) - 1,

so it is written in the same plane format as the any-precision code (MSB first, beta, alpha) and
runs through the same analog read. Planes beyond b are padding with alpha = 0 and are never read.

GPTQ needs the second-order statistics H = X^T X of each matrix's input. They are gathered by running
the full-precision model on WikiText-2 *train* text (disjoint from the test text of the KL
experiments): for gate/up the expert's input, for down the expert's gated intermediate activation,
over the tokens routed to that expert. A few layers are processed per pass so that H fits on the GPU.

  python scripts/20_encode_native_gptq.py --model allenai/OLMoE-1B-7B-0125 --out cache/planes_olmoe_gptq
  -> cache/planes_olmoe_gptq4/ and cache/planes_olmoe_gptq8/
"""
import argparse, time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset

from ocdpa.fp_experts import FPExperts
from ocdpa.models import load, moe_layers, tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0125")
ap.add_argument("--out", required=True, help="prefix: writes <out>4 and <out>8")
ap.add_argument("--bits", default="4,8")
ap.add_argument("--chunk", type=int, default=4, help="layers whose Hessians are held per pass")
ap.add_argument("--tokens", type=int, default=32768, help="calibration tokens")
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--batch", type=int, default=4, help="windows per forward")
ap.add_argument("--min-tokens", type=int, default=64, help="fewer routed tokens than this: H = I (round-to-nearest)")
ap.add_argument("--damp", type=float, default=0.01)
args = ap.parse_args()
GS, KMAX = 128, 8
BITS = [int(b) for b in args.bits.split(",")]
OUT = {b: Path(f"{args.out}{b}") for b in BITS}
for d in OUT.values():
    d.mkdir(parents=True, exist_ok=True)


class Stop(Exception):
    pass


class Collector(FPExperts):
    """Full-precision expert stage that also accumulates H = X^T X per expert and matrix."""

    def __init__(self, *a, last=False, **kw):
        super().__init__(*a, **kw)
        self.last = last
        self.Hgu, self.Hdn, self.n = {}, {}, {}

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        mask = torch.nn.functional.one_hot(top_k_index, self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            w = self._weights(e)
            x = hidden_states[tok].to(self.dtype)
            mid = self.act_fn(x @ w["gate_proj"].T) * (x @ w["up_proj"].T)
            if e not in self.n:
                self.Hgu[e] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                self.Hdn[e] = torch.zeros(mid.shape[1], mid.shape[1], device=x.device)
                self.n[e] = 0
            self.Hgu[e] += x.T @ x
            self.Hdn[e] += mid.T @ mid
            self.n[e] += len(tok)
            y = (mid @ w["down_proj"].T) * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, y.to(final.dtype))
            del w
        if self.last:
            raise Stop
        return final


def inverse_factor(H, n):
    """Upper Cholesky factor of H^-1 (GPTQ), with dampening; round-to-nearest if H is not trusted."""
    H = H.clone()
    d = H.shape[0]
    if n < args.min_tokens:
        H = torch.eye(d, device=H.device)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += args.damp * torch.diag(H).mean() * torch.eye(d, device=H.device)
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)


def gptq(W, Hinv, bits):
    """GPTQ with asymmetric min/max scales per (row, group of GS inputs).
    W (rows, cols) -> integer codes (rows, cols), lo and s (rows, cols/GS)."""
    W = W.clone().float()
    rows, cols = W.shape
    levels = 2 ** bits - 1
    Q = torch.zeros(rows, cols, dtype=torch.long, device=W.device)
    LO = torch.zeros(rows, cols // GS, device=W.device)
    S = torch.zeros_like(LO)
    for g, i1 in enumerate(range(0, cols, GS)):
        i2 = i1 + GS
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hb = Hinv[i1:i2, i1:i2]
        lo, hi = W1.min(1).values, W1.max(1).values
        s = ((hi - lo) / levels).clamp_min(1e-12)
        LO[:, g], S[:, g] = lo, s
        for i in range(GS):
            w = W1[:, i]
            q = torch.round((w - lo) / s).clamp(0, levels)
            Q[:, i1 + i] = q.long()
            err = (w - (lo + s * q)) / Hb[i, i]
            W1[:, i:] -= err[:, None] * Hb[i, i:][None]
            Err[:, i] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q, LO, S


def to_planes(Q, LO, S, bits):
    """Codes -> (signs (n*J, GS, KMAX) bool, alpha (n*J, KMAX), beta (n*J))."""
    n, m = Q.shape
    J = m // GS
    q = Q.reshape(n * J, GS)
    lo, s = LO.reshape(n * J), S.reshape(n * J)
    planes = torch.ones(n * J, GS, KMAX, dtype=torch.bool, device=Q.device)
    alpha = torch.zeros(n * J, KMAX, device=Q.device)
    for p in range(bits):
        planes[:, :, p] = ((q >> (bits - 1 - p)) & 1).bool()
        alpha[:, p] = s * 2.0 ** (bits - 2 - p)
    return planes, alpha, lo + (2 ** bits - 1) / 2 * s


def store_matrix(store, e, mat, Q, LO, S, bits):
    planes, alpha, beta = to_planes(Q, LO, S, bits)
    n, m = Q.shape
    store[f"{e}.{mat}.B"] = np.packbits(planes.cpu().numpy().reshape(-1))
    store[f"{e}.{mat}.a"] = alpha.half().cpu().numpy()
    store[f"{e}.{mat}.b"] = beta.half().cpu().numpy()
    store[f"{e}.{mat}.s"] = np.array([n, m, m // GS], dtype=np.int32)


if __name__ == "__main__":
    t0 = time.time()
    tok = tokenizer(args.model)
    ds = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    W = args.window
    n_win = args.tokens // W
    wins = ids[: n_win * W].reshape(n_win, W).cuda()

    def patch(li, original):
        return FPExperts(args.model, li, original.num_experts, original.act_fn)
    model, cfg = load(args.model, patch=patch)
    layers = [li for li in moe_layers(args.model) if not all((OUT[b] / f"L{li}.npz").exists() for b in BITS)]
    print(f"{len(layers)} layers to do, {n_win} calibration windows of {W} tokens", flush=True)

    for c in range(0, len(layers), args.chunk):
        chunk = layers[c:c + args.chunk]
        keep = {li: model.model.layers[li].mlp.experts for li in chunk}
        col = {}
        for li in chunk:
            fp = keep[li]
            col[li] = Collector(args.model, li, fp.num_experts, fp.act_fn, last=(li == chunk[-1]))
            model.model.layers[li].mlp.experts = col[li]
        with torch.no_grad():
            for b in range(0, n_win, args.batch):
                try:
                    model(wins[b:b + args.batch])
                except Stop:
                    pass
        for li in chunk:
            model.model.layers[li].mlp.experts = keep[li]
        print(f"  layers {chunk}: Hessians gathered [{time.time()-t0:.0f}s]", flush=True)

        for li in chunk:
            cl = col[li]
            stores = {b: {} for b in BITS}
            for e in range(cl.num_experts):
                w = cl._weights(e)
                if e not in cl.n:
                    cl.n[e], cl.Hgu[e], cl.Hdn[e] = 0, torch.zeros(w["gate_proj"].shape[1], w["gate_proj"].shape[1], device="cuda"), torch.zeros(w["down_proj"].shape[1], w["down_proj"].shape[1], device="cuda")
                Hi_gu = inverse_factor(cl.Hgu[e], cl.n[e])
                Hi_dn = inverse_factor(cl.Hdn[e], cl.n[e])
                gu = torch.cat([w["gate_proj"], w["up_proj"]], 0)       # same input -> same H, rows independent
                I = w["gate_proj"].shape[0]
                for b in BITS:
                    Q, LO, S = gptq(gu, Hi_gu, b)
                    store_matrix(stores[b], e, "gate_proj", Q[:I], LO[:I], S[:I], b)
                    store_matrix(stores[b], e, "up_proj", Q[I:], LO[I:], S[I:], b)
                    Q, LO, S = gptq(w["down_proj"], Hi_dn, b)
                    store_matrix(stores[b], e, "down_proj", Q, LO, S, b)
                del w
            for b in BITS:
                np.savez(OUT[b] / f"L{li}.npz", **stores[b])
            print(f"  L{li}: quantised [{time.time()-t0:.0f}s]", flush=True)
            del col[li], cl
        torch.cuda.empty_cache()
