#!/usr/bin/env python3
"""Build train-only multiple-choice calibration text with a fixed seed.

The dynamic residual profile should see the same scoring shape as
llama-perplexity --multiple-choice: prompt + every candidate continuation.
This script writes one text block per candidate from train splits only.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
OUT_DEFAULT = REPO / "cache/datasets/mc_train_candidates_seed0.txt"


def sample_rows(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n <= 0 or n >= len(rows):
        rows = list(rows)
        rng.shuffle(rows)
        return rows
    return rng.sample(rows, n)


def add_block(out: list[str], task: str, item_id: str, prompt: str,
              choices: list[str]) -> None:
    for i, choice in enumerate(choices):
        out.append(
            f"[task={task} id={item_id} candidate={i}]\n"
            f"{prompt.rstrip()}\nAnswer: {choice.strip()}\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hellaswag", type=int, default=1200)
    ap.add_argument("--arc-challenge", type=int, default=800)
    ap.add_argument("--arc-easy", type=int, default=600)
    ap.add_argument("--winogrande", type=int, default=600)
    ap.add_argument("--piqa", type=int, default=400)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("missing dependency: pip install datasets") from exc

    rng = random.Random(args.seed)
    blocks: list[str] = []

    arc_specs = [
        ("arc_challenge", "ARC-Challenge", args.arc_challenge),
        ("arc_easy", "ARC-Easy", args.arc_easy),
    ]
    for task, config, n in arc_specs:
        rows = list(load_dataset("allenai/ai2_arc", config, split="train"))
        for row in sample_rows(rows, n, rng):
            choices = row["choices"]["text"]
            prompt = f"Question: {row['question']}"
            add_block(blocks, task, str(row.get("id", "")), prompt, choices)

    rows = list(load_dataset("hellaswag", split="train"))
    for row in sample_rows(rows, args.hellaswag, rng):
        prompt = f"{row['ctx_a']} {row['ctx_b']}".strip()
        add_block(blocks, "hellaswag", str(row.get("ind", "")), prompt, row["endings"])

    rows = list(load_dataset("winogrande", "winogrande_xl", split="train"))
    for idx, row in enumerate(sample_rows(rows, args.winogrande, rng)):
        prompt = f"Question: {row['sentence']}\nFill in the blank."
        add_block(blocks, "winogrande", str(row.get("qID", idx)),
                  prompt, [row["option1"], row["option2"]])

    try:
        rows = list(load_dataset("piqa", split="train"))
    except Exception as exc:
        print(f"[warn] skipping PIQA train calibration: {exc}")
    else:
        for idx, row in enumerate(sample_rows(rows, args.piqa, rng)):
            prompt = f"Question: {row['goal']}"
            add_block(blocks, "piqa", str(idx), prompt, [row["sol1"], row["sol2"]])

    rng.shuffle(blocks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(blocks) + "\n")
    print(f"wrote {args.out} ({len(blocks)} candidate blocks, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
