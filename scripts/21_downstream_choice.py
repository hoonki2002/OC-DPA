"""Downstream accuracy under phase-aware precision.

Every multiple-choice item is run the way the device would run it: the
question is a prefill and the answer is decoded. The context goes through the
analog experts once, at the prefill budget, and builds the KV cache; each
candidate answer then attends to that cache and is scored with the decode
budget. The first answer token is predicted by the prefill's last position,
exactly as the first generated token would be -- so single-token answers
(BoolQ) measure the prefill precision alone.

Tasks (lm-evaluation-harness prompt formats, 0-shot): HellaSwag (acc_norm, by answer length in bytes),
WinoGrande (partial scoring) and BoolQ (acc). Items are a fixed random subsample per task.
"""
import argparse, json, os, re, time
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset

from ocdpa.models import load, tokenizer
from ocdpa.analog import AnalogExperts, PlaneBank
from ocdpa.allocators import Uniform, DPA

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--planes", required=True)
ap.add_argument("--residual", required=True)
ap.add_argument("--home", default="cuda")
ap.add_argument("--items", type=int, default=500)
ap.add_argument("--tok-budget", type=int, default=4096, help="context tokens per forward")
ap.add_argument("--configs", default="8-8,DPA4-8,uni4-8,DPA4-DPA4,DPA3-8")
ap.add_argument("--out", required=True)
ap.add_argument("--scale-bits", type=int, default=8, help="storage width of alpha/beta (log code); 0 keeps fp16")
args = ap.parse_args()


# ---------------------------------------------------------------- tasks
def hs_clean(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def tasks(n, seed=0):
    """name -> list of (contexts, continuations, gold, metric).

    `contexts` has one string per group of continuations; most tasks share one
    context across all choices, WinoGrande has one context per option and a
    shared continuation."""
    rng = np.random.default_rng(seed)
    pick = lambda ds: [ds[int(i)] for i in sorted(rng.choice(len(ds), min(n, len(ds)), replace=False))]
    T = {}
    hs = []
    for r in pick(load_dataset("Rowan/hellaswag", split="validation")):
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        ends = r["endings"] if isinstance(r["endings"], list) else eval(r["endings"])
        hs.append(([hs_clean(r["activity_label"] + ": " + ctx)], [[hs_clean(e) for e in ends]],
                   int(r["label"]), "acc_norm"))
    T["hellaswag"] = hs
    wg = []
    for r in pick(load_dataset("allenai/winogrande", "winogrande_xl", split="validation")):
        i = r["sentence"].index("_")
        cont = r["sentence"][i + 1:].strip()
        ctxs = [r["sentence"][:i] + r["option1"], r["sentence"][:i] + r["option2"]]
        wg.append((ctxs, [[cont], [cont]], int(r["answer"]) - 1, "acc"))
    T["winogrande"] = wg
    T["boolq"] = [([f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"], [["no", "yes"]],
                   int(r["label"]), "acc")
                  for r in pick(load_dataset("aps/super_glue", "boolq", split="validation"))]
    return T


# ---------------------------------------------------------------- model
banks = {}
def patch(li, original):
    banks[li] = PlaneBank(Path(args.planes) / f"L{li}.npz", "cuda", home=args.home, scale_bits=(args.scale_bits or None))
    return AnalogExperts(original, banks[li], layer=li, adc_bits=8)

t0 = time.time()
model, cfg = load(args.model, patch=patch)
analog = [l.mlp.experts for l in model.model.layers if hasattr(l.mlp, "experts")]
topk = cfg.num_experts_per_tok
R = np.load(args.residual)
tok = tokenizer(args.model)
BOS = cfg.bos_token_id
PAD = tok.pad_token_id if tok.pad_token_id is not None else (cfg.eos_token_id or 0)
if isinstance(PAD, list):
    PAD = PAD[0]


def policy(spec):
    """'DPA4' / 'uni4' / '8' -> allocator."""
    if spec.startswith("DPA"):
        return DPA(int(spec[3:]), R, topk)
    return Uniform(int(spec.replace("uni", "")))


def set_policy(p):
    for a in analog:
        a.policy = p


def encode(T):
    """Flatten every task into context rows and continuation rows."""
    ctx_rows, cont_rows = [], []      # cont: (ctx_row, token ids, task, item, choice, nbytes)
    for name, items in T.items():
        for ii, (ctxs, conts, gold, metric) in enumerate(items):
            for gi, (c, group) in enumerate(zip(ctxs, conts)):
                ids = tok(c, add_special_tokens=False).input_ids
                if BOS is not None:
                    ids = [BOS] + ids
                ci = len(ctx_rows)
                ctx_rows.append(ids)
                for ch_i, ans in enumerate(group):
                    choice = gi if len(ctxs) > 1 else ch_i
                    a = tok(" " + ans, add_special_tokens=False).input_ids
                    cont_rows.append((ci, a, name, ii, choice, len(ans.encode())))
    return ctx_rows, cont_rows


def score(ctx_rows, cont_rows, pre, dec):
    """log P(answer | context) for every continuation row."""
    by_ctx = {}
    for r, c in enumerate(cont_rows):
        by_ctx.setdefault(c[0], []).append(r)
    order = sorted(range(len(ctx_rows)), key=lambda i: len(ctx_rows[i]))
    out = np.zeros(len(cont_rows))
    # both passes must fit: contexts x longest context, and answers x
    # longest answer
    n_ans = {i: len(v) for i, v in by_ctx.items()}
    a_len = {i: max(len(cont_rows[r][1]) for r in v) for i, v in by_ctx.items()}
    batches, cur, cur_len, cur_n, cur_a = [], [], 0, 0, 0
    for i in order:
        L = len(ctx_rows[i])
        if cur and ((len(cur) + 1) * max(cur_len, L) > args.tok_budget or
                    (cur_n + n_ans[i]) * max(cur_a, a_len[i]) > args.tok_budget):
            batches.append(cur); cur, cur_len, cur_n, cur_a = [], 0, 0, 0
        cur.append(i); cur_len = max(cur_len, L)
        cur_n += n_ans[i]; cur_a = max(cur_a, a_len[i])
    if cur:
        batches.append(cur)
    for b in batches:
        L = max(len(ctx_rows[i]) for i in b)
        ids = torch.full((len(b), L), PAD, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):                         # left padding
            ids[r, L - len(ctx_rows[i]):] = torch.tensor(ctx_rows[i]); mask[r, L - len(ctx_rows[i]):] = 1
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        ids, mask, pos = ids.cuda(), mask.cuda(), pos.cuda()
        set_policy(pre)
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                      logits_to_keep=1)                   # only the next-token logits
        first = F.log_softmax(o.logits[:, -1].float(), -1)
        cache = o.past_key_values
        rows = [(r, cr) for r, i in enumerate(b) for cr in by_ctx[i]]
        sel = torch.tensor([r for r, _ in rows], device="cuda")
        A = max(len(cont_rows[cr][1]) for _, cr in rows)
        a_ids = torch.full((len(rows), A), PAD, dtype=torch.long, device="cuda")
        a_mask = torch.zeros((len(rows), A), dtype=torch.long, device="cuda")
        for j, (_, cr) in enumerate(rows):
            a = cont_rows[cr][1]
            a_ids[j, :len(a)] = torch.tensor(a); a_mask[j, :len(a)] = 1
        lp = torch.stack([first[r, a_ids[j, 0]] for j, (r, _) in enumerate(rows)])
        if A > 1:
            cache.batch_select_indices(sel)
            full_mask = torch.cat([mask[sel], a_mask], 1)
            a_pos = mask[sel].sum(1, keepdim=True) + torch.arange(A, device="cuda")[None]
            set_policy(dec)
            with torch.no_grad():
                o2 = model(input_ids=a_ids, attention_mask=full_mask, position_ids=a_pos,
                           past_key_values=cache, use_cache=True)
            # a vocabulary of 1e5 over thousands of answer tokens does not fit
            # as one fp32 tensor; normalise a few rows at a time
            tail = torch.zeros(len(rows), device="cuda")
            step = max(1, 1024 // A)                     # ~1k answer tokens at a time
            for c0 in range(0, len(rows), step):
                lg = F.log_softmax(o2.logits[c0:c0 + step, :-1].float(), -1)
                tail[c0:c0 + step] = (lg.gather(2, a_ids[c0:c0 + step, 1:, None])[..., 0]
                                      * a_mask[c0:c0 + step, 1:]).sum(1)
                del lg
            lp = lp + tail
        for j, (_, cr) in enumerate(rows):
            out[cr] = float(lp[j])
        del o, cache
        if A > 1:
            del o2
    return out


def accuracy(T, cont_rows, lp):
    best = {}
    for (ci, a, name, ii, choice, nb), v in zip(cont_rows, lp):
        metric = T[name][ii][3]
        best.setdefault((name, ii), {})[choice] = v / nb if metric == "acc_norm" else v
    acc = {}
    for (name, ii), d in best.items():
        pred = max(d, key=d.get)
        acc.setdefault(name, []).append(int(pred == T[name][ii][2]))
    res = {k: 100 * float(np.mean(v)) for k, v in acc.items()}
    res["avg"] = float(np.mean([res[k] for k in T]))
    return res


T = tasks(args.items)
ctx_rows, cont_rows = encode(T)
print(f"{len(T)} tasks, {sum(len(v) for v in T.values())} items, {len(ctx_rows)} contexts "
      f"({sum(map(len, ctx_rows))} tokens), {len(cont_rows)} answers [{time.time()-t0:.0f}s]", flush=True)
results = json.load(open(args.out)) if Path(args.out).exists() else {}
names = list(T)
print("  " + f"{'prefill->decode':>16} " + " ".join(f"{n[:9]:>9}" for n in names) + f" {'avg':>6}", flush=True)
for spec in args.configs.split(","):
    if spec in results:
        r = results[spec]
    else:
        pre, dec = spec.split("-")
        r = accuracy(T, cont_rows, score(ctx_rows, cont_rows, policy(pre), policy(dec)))
        results[spec] = r
        json.dump(results, open(args.out, "w"), indent=1)
    print("  " + f"{spec:>16} " + " ".join(f"{r[n]:9.2f}" for n in names) + f" {r['avg']:6.2f}"
          f"   [{time.time()-t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB]", flush=True)
