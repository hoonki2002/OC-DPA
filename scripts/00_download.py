"""Download what the scripts read: the two OLMoE checkpoints and the datasets, into the default Hugging Face cache.

  python scripts/00_download.py                 # everything below (about 42 GB of models, 0.5 GB of datasets)
  python scripts/00_download.py --what quality  # base checkpoint + WikiText-2 (KL, routing traces, residual table, GPTQ copies)
  python scripts/00_download.py --what downstream   # the Instruct checkpoint + HellaSwag, WinoGrande, BoolQ, GSM8K, HumanEval
  python scripts/00_download.py --dry-run       # list what would be fetched and its size

  what           model                                   used by
  hardware       nothing                                  scripts/10-13 (the inputs are in data/)
  quality        allenai/OLMoE-1B-7B-0125   (base, 28 GB, fp32) scripts/01-06, 20, 21, 22 (HellaSwag, WinoGrande, BoolQ)
  downstream     allenai/OLMoE-1B-7B-0125-Instruct (14 GB) scripts/23 (GSM8K and HumanEval)

The scripts look for the checkpoints in ~/.cache/huggingface/hub (ocdpa/models.py): keep the default cache location
(or link it there) and run them after this script, which switches the offline mode of the datasets library off.
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--what", choices=["all", "quality", "downstream"], default="all")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

MODELS = {"quality": ["allenai/OLMoE-1B-7B-0125"], "downstream": ["allenai/OLMoE-1B-7B-0125-Instruct"]}
DATASETS = {"quality": [("wikitext", "wikitext-2-raw-v1")],
            "downstream": [("Rowan/hellaswag", None), ("allenai/winogrande", "winogrande_xl"), ("aps/super_glue", "boolq"),
                           ("openai/gsm8k", "main"), ("openai/openai_humaneval", None)]}
# HellaSwag, WinoGrande and BoolQ are multiple-choice tasks of the quality model too (Table III)
DATASETS["quality"] += DATASETS["downstream"][:3]
PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"]
parts = ["quality", "downstream"] if args.what == "all" else [args.what]
seen = set()

from huggingface_hub import HfApi, snapshot_download
for part in parts:
    for repo in MODELS[part]:
        if args.dry_run:
            info = HfApi().model_info(repo, files_metadata=True)
            size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith((".safetensors", ".json")))
            print(f"model    {repo}: {size / 1e9:.1f} GB")
        else:
            print(f"downloading {repo} ...", flush=True)
            snapshot_download(repo, allow_patterns=PATTERNS)
    for name, config in DATASETS[part]:
        if (name, config) in seen:
            continue
        seen.add((name, config))
        print(f"dataset  {name} {config or ''}", flush=True)
        if not args.dry_run:
            from datasets import load_dataset
            load_dataset(name, config) if config else load_dataset(name)
print("done" if not args.dry_run else "dry run")
