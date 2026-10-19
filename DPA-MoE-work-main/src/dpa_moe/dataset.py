"""Calibration DataLoader.

Upstream's `CachedDataset` expects HF `default_data_collator` output —
a dict with key "input_ids" of shape [B, L]. We tokenize the user's
calibration text files, concatenate, then carve into seq_len blocks.
"""
from __future__ import annotations

import pathlib

import torch


def build_calibration_loader(tokenizer, text_files: list[pathlib.Path],
                              seq_len: int, num_samples: int, batch_size: int):
    """Return a torch DataLoader matching upstream `CachedDataset`'s
    contract (yields dicts of {"input_ids": [B, L] int64}).

    `text_files` are concatenated with double-newline separators before
    tokenization. We slice the resulting token stream into `num_samples`
    contiguous blocks of `seq_len` tokens each (a standard GPTQ-style
    calibration batch). If the corpus is too short we repeat it.
    """
    from torch.utils.data import Dataset, DataLoader
    from transformers import default_data_collator

    text = "\n\n".join(pathlib.Path(p).read_text() for p in text_files)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    needed = seq_len * num_samples
    if ids.numel() < needed:
        repeats = -(-needed // ids.numel())
        ids = ids.repeat(repeats)[:needed]
    else:
        ids = ids[:needed]
    samples = ids.view(num_samples, seq_len)

    class _DS(Dataset):
        def __len__(self): return num_samples
        def __getitem__(self, idx):
            return {"input_ids": samples[idx].long(),
                    "attention_mask": torch.ones(seq_len, dtype=torch.long)}

    return DataLoader(_DS(), batch_size=batch_size, collate_fn=default_data_collator)
