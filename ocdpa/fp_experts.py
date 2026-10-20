"""Full-precision expert stage that streams the released weights from disk.

Needed only for the reference pass: a 30 B model's expert tensors are 29 B
parameters, so they are read one layer at a time, used, and dropped rather
than held resident.
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn
from safetensors import safe_open

from ocdpa.models import snapshot


class FPExperts(nn.Module):
    def __init__(self, model_id, layer, num_experts, act_fn, device="cuda",
                 dtype=torch.float32):
        super().__init__()
        self.num_experts, self.act_fn = num_experts, act_fn
        self.device, self.dtype = device, dtype
        root = snapshot(model_id)
        index = root / "model.safetensors.index.json"
        wmap = json.load(open(index))["weight_map"] if index.exists() else {}
        self.prefix = f"model.layers.{layer}.mlp.experts."
        self.files = {}
        for key, shard in wmap.items():
            if key.startswith(self.prefix):
                self.files.setdefault(root / shard, []).append(key)

    def _weights(self, expert):
        out = {}
        for path, keys in self.files.items():
            want = [k for k in keys if k.startswith(f"{self.prefix}{expert}.")]
            if not want:
                continue
            with safe_open(path, framework="pt") as handle:
                for k in want:
                    out[k.split(".")[-2]] = handle.get_tensor(k).to(self.device, self.dtype)
        return out

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
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
            y = (mid @ w["down_proj"].T) * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, y.to(final.dtype))
            del w
        return final
