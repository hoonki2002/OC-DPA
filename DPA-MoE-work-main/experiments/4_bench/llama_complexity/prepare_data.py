#!/usr/bin/env python3
"""Prepare the multiple-choice JSONL datasets read by run_bench.py.

Splits match the paper runs: ARC-Challenge test (1172), ARC-Easy
validation (570), PIQA validation (1838), WinoGrande-xl validation (1267).
HellaSwag is built from a local ``hellaswag_val_full.txt``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "cache/datasets"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def arc(config: str, split: str, subject: str) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    rows = []
    for ex in ds:
        labels = [str(x) for x in ex["choices"]["label"]]
        texts = [str(x) for x in ex["choices"]["text"]]
        answer = str(ex["answerKey"])
        if answer not in labels and answer.isdigit():
            answer = chr(ord("A") + int(answer) - 1)
        choices = {chr(ord("A") + i): text for i, text in enumerate(texts)}
        correct_idx = labels.index(answer)
        rows.append({
            "id": ex["id"],
            "question": ex["question"],
            "choices": choices,
            "correct_letter": chr(ord("A") + correct_idx),
            "subject": subject,
            "kind": "mc",
        })
    return rows


def piqa() -> list[dict]:
    # The original ybisk/piqa loader script is unsupported by datasets>=4;
    # the hub's parquet conversion of the same repo has identical rows.
    ds = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
    return [{
        "id": f"piqa_{i:05d}",
        "question": ex["goal"],
        "choices": {"A": ex["sol1"], "B": ex["sol2"]},
        "correct_letter": "AB"[int(ex["label"])],
        "subject": "piqa",
        "kind": "mc",
    } for i, ex in enumerate(ds)]


def winogrande() -> list[dict]:
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    rows = []
    for i, ex in enumerate(ds):
        prefix, suffix = ex["sentence"].split("_", 1)
        choices = {
            "A": ex["option1"] + suffix,
            "B": ex["option2"] + suffix,
        }
        rows.append({
            "id": ex.get("qID", f"winogrande_{i:05d}"),
            "question": prefix,
            "choices": choices,
            "correct_letter": "A" if str(ex["answer"]) == "1" else "B",
            "subject": "winogrande",
            "kind": "cloze",
        })
    return rows


def hellaswag() -> list[dict]:
    path = DATA / "hellaswag_val_full.txt"
    lines = [line.rstrip("\n") for line in path.open()]
    if len(lines) % 6 != 0:
        raise ValueError(f"{path} should have 6 lines per example; got {len(lines)}")
    rows = []
    for i in range(0, len(lines), 6):
        prompt = lines[i]
        answer_idx = int(lines[i + 1])
        endings = lines[i + 2:i + 6]
        choices = {chr(ord("A") + j): ending for j, ending in enumerate(endings)}
        rows.append({
            "id": f"hellaswag_{i // 6:05d}",
            "question": prompt,
            "choices": choices,
            "correct_letter": chr(ord("A") + answer_idx),
            "subject": "hellaswag",
            "kind": "mc",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="arc_c,arc_e,piqa,winogrande",
                    help="comma-separated tasks: arc_c,arc_e,piqa,winogrande,hellaswag")
    args = ap.parse_args()

    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        if task == "arc_c":
            write_jsonl(DATA / "arc_c.jsonl", arc("ARC-Challenge", "test", "arc_challenge"))
        elif task == "arc_e":
            write_jsonl(DATA / "arc_e.jsonl", arc("ARC-Easy", "validation", "arc_easy"))
        elif task == "piqa":
            write_jsonl(DATA / "piqa.jsonl", piqa())
        elif task == "winogrande":
            write_jsonl(DATA / "winogrande.jsonl", winogrande())
        elif task == "hellaswag":
            write_jsonl(DATA / "hellaswag.jsonl", hellaswag())
        else:
            raise SystemExit(f"unknown task {task!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
