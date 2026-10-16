"""GSM8K and HumanEval, generated the way the device would run them.

Prefill (the prompt) at the prefill budget, then greedy decoding at the decode
budget, attending to the prefill's KV cache. Configs:

  digital   the released model in bf16 on the GPU (no AIMC)   --mode digital
  DPA4-8    AIMC experts, OC-DPA at k-bar 4 in prefill, k = 8 in decode
  uni4-8    AIMC experts, uniform k = 4 in prefill, k = 8 in decode

AIMC runs include the default non-idealities (--noise). Progress is saved
after every batch, so an interrupted run resumes where it stopped.
HumanEval programs run in a subprocess with a time and memory limit.
"""
import argparse, json, os, re, resource, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np, torch
from datasets import load_dataset

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["analog", "digital"], required=True)
ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0125-Instruct")
ap.add_argument("--planes", default="cache/planes_olmoe_instruct")
ap.add_argument("--residual", default="cache/residual_olmoe_instruct.npy")
ap.add_argument("--noise", default="mixed,0.01,0.02,cal")
ap.add_argument("--configs", default="DPA4-8,uni4-8")
ap.add_argument("--tasks", default="gsm8k,humaneval")
ap.add_argument("--gsm-problems", type=int, default=300)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max-new-gsm", type=int, default=320)
ap.add_argument("--max-new-he", type=int, default=512)
ap.add_argument("--limit", type=int, default=0, help="debug: cap problems per task")
ap.add_argument("--out", required=True)
ap.add_argument("--scale-bits", type=int, default=8, help="storage width of alpha/beta (log code); 0 keeps fp16")
args = ap.parse_args()
t0 = time.time()

from ocdpa.models import tokenizer
tok = tokenizer(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

if args.mode == "analog":
    from ocdpa.models import load
    from ocdpa.analog import AnalogExperts, PlaneBank, NoiseModel
    from ocdpa.allocators import Uniform, DPA
    def patch(li, original):
        return AnalogExperts(original, PlaneBank(Path(args.planes) / f"L{li}.npz", "cuda", home="cuda", scale_bits=(args.scale_bits or None)),
                                  layer=li, adc_bits=8)
    model, cfg = load(args.model, patch=patch)
    analog = [l.mlp.experts for l in model.model.layers if hasattr(l.mlp, "experts")]
    c_, s1_, sx_, ir_ = [v.strip() for v in args.noise.split(",")]
    NOISE = None if args.noise == "none" else NoiseModel(c_, float(s1_), float(sx_), ir_)
    for a in analog:
        a.noise = NOISE
    R = np.load(args.residual)
    def policy(spec):
        return DPA(int(spec[3:]), R, cfg.num_experts_per_tok) if spec.startswith("DPA") \
            else Uniform(int(spec.replace("uni", "")))
    def set_policy(p):
        for a in analog:
            a.policy = p
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    cfg = model.config
    policy = lambda spec: None
    set_policy = lambda p: None
EOS = {tok.eos_token_id} | ({cfg.eos_token_id} if isinstance(cfg.eos_token_id, int) else set(cfg.eos_token_id or []))
print(f"model ready [{time.time()-t0:.0f}s]", flush=True)


# ---------------------------------------------------------------- tasks
def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

def build(task):
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rng = np.random.default_rng(0)
        n = args.limit or min(args.gsm_problems, len(ds))
        pick = sorted(rng.choice(len(ds), n, replace=False).tolist())
        rows = [ds[i] for i in pick]
        gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]
        ask = [r["question"] + "\nSolve it step by step, then finish with 'The answer is N.' "
               "where N is the final number." for r in rows]
        return [chat(q) for q in ask], gold, args.max_new_gsm
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = [ds[i] for i in range(args.limit or len(ds))]
    ask = ["Complete the following Python function. Write the complete function in a single "
           "```python code block.\n\n```python\n" + r["prompt"].rstrip() + "\n```" for r in rows]
    return [chat(q) for q in ask], rows, args.max_new_he


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def run_program(code, timeout=15):
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, capture_output=True,
                               timeout=timeout + 5, preexec_fn=limits)
            return p.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def he_correct(reply, row):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
    if not blocks:
        m = re.search(r"```(?:python|py)?\s*\n(.*)$", reply, re.S)     # unterminated block
        blocks = [m.group(1)] if m else [reply]
    ep = row["entry_point"]
    code = next((b for b in blocks if f"def {ep}" in b), blocks[0])
    header = "\n".join(l for l in row["prompt"].splitlines() if re.match(r"\s*(import|from)\s", l))
    return run_program(f"{header}\n{code}\n\n{row['test']}\n\ncheck({ep})\n")


def correct(task, reply, ref):
    if task == "gsm8k":
        n = last_number(reply)
        return n is not None and abs(float(n) - float(ref)) < 1e-6
    return he_correct(reply, ref)


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate(prompts, pre, dec, max_new):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids, mask = enc.input_ids.cuda(), enc.attention_mask.cuda()
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    set_policy(pre)
    o = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
    cache, nxt = o.past_key_values, o.logits[:, -1].argmax(-1)
    out = [[] for _ in prompts]
    done = torch.zeros(len(prompts), dtype=torch.bool, device="cuda")
    set_policy(dec)
    last = pos[:, -1]
    for step in range(max_new):
        for b in range(len(prompts)):
            if not done[b]:
                out[b].append(int(nxt[b]))
        done |= torch.tensor([int(t) in EOS for t in nxt.tolist()], device="cuda")
        if bool(done.all()):
            break
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        last = last + 1
        o = model(input_ids=nxt[:, None], attention_mask=mask, position_ids=last[:, None],
                  past_key_values=cache, use_cache=True)
        cache, nxt = o.past_key_values, o.logits[:, -1].argmax(-1)
    return [tok.decode(x, skip_special_tokens=True) for x in out], step + 1


results = json.load(open(args.out)) if Path(args.out).exists() else {}
def save():
    json.dump(results, open(args.out, "w"), indent=1)

for task in args.tasks.split(","):
    prompts, refs, max_new = build(task)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))      # similar lengths together
    print(f"{task}: {len(prompts)} problems, batch {args.batch}, max_new {max_new} [{time.time()-t0:.0f}s]", flush=True)
    specs = ["digital"] if args.mode == "digital" else args.configs.split(",")
    for spec in specs:
        key = f"{task}/{spec}"
        r = results.get(key, {})
        if "acc" in r:
            print(f"  {key}: {r['acc']:.1f}% (cached)", flush=True); continue
        pre, dec = (None, None) if spec == "digital" else (policy(s) for s in spec.split("-"))
        replies = r.get("partial", [])                  # replies in `order`
        for s in range(len(replies), len(order), args.batch):
            idx = order[s:s + args.batch]
            out, n = generate([prompts[i] for i in idx], pre, dec, max_new)
            replies += out
            results[key] = dict(partial=replies); save()
            print(f"    {key} batch {s // args.batch}: {n} steps [{time.time()-t0:.0f}s]", flush=True)
        by_idx = {i: rep for i, rep in zip(order, replies)}
        ok = [bool(correct(task, by_idx[i], refs[i])) for i in range(len(prompts))]
        results[key] = dict(acc=100 * float(np.mean(ok)), n=len(ok), correct=[int(c) for c in ok],
                            replies=[by_idx[i] for i in range(len(prompts))])
        save()
        print(f"  {key}: {results[key]['acc']:.1f}%  ({sum(ok)}/{len(ok)}) [{time.time()-t0:.0f}s]", flush=True)
