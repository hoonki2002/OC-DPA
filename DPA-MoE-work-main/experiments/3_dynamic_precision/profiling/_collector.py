"""Routed-activation collector for per-expert residual profiling."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class Collector:
    """Vectorised routed-activation collector.

    All accumulation stays on GPU during the forward pass. Per layer
    ℓ we store the concatenated ``x_in`` rows from every batch plus the
    ``(top_k_index, top_k_score)`` routing decisions. After forward,
    ``gather_per_expert(L)`` produces the per-expert sample tensors on
    demand (so we can free one layer's raw buffers as soon as its
    residuals are computed).
    """

    def __init__(self, n_layers: int, n_experts: int, hidden: int,
                 top_k: int, max_per_expert: int, device):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.hidden = hidden
        self.top_k = top_k
        self.max_per_expert = max_per_expert
        self.device = device
        # Raw buffers, one per layer.
        self.X_chunks: list[list] = [[] for _ in range(n_layers)]
        self.Idx_chunks: list[list] = [[] for _ in range(n_layers)]
        self.Score_chunks: list[list] = [[] for _ in range(n_layers)]
        # Aggregates (GPU, float32):
        self.W_sum = torch.zeros((n_layers, n_experts), dtype=torch.float32, device=device)
        self.N = torch.zeros((n_layers, n_experts), dtype=torch.int64, device=device)

    def add_batch(self, layer: int, x_in: torch.Tensor,
                  scores: torch.Tensor, idx: torch.Tensor) -> None:
        """x_in: (T, hidden) bf16; scores: (T, top_k) f32; idx: (T, top_k) int."""
        self.X_chunks[layer].append(x_in)
        self.Idx_chunks[layer].append(idx)
        self.Score_chunks[layer].append(scores)
        flat_e = idx.reshape(-1)
        flat_w = scores.reshape(-1)
        self.W_sum[layer].scatter_add_(0, flat_e.long(), flat_w.float())
        ones = torch.ones_like(flat_e, dtype=torch.int64)
        self.N[layer].scatter_add_(0, flat_e.long(), ones)

    def gather_per_expert(self, layer: int) -> list:
        """Per expert e, return ``(X_e, w_e)`` or None if unrouted.

        X_e : (n_e_i, hidden) bf16 — routed token's hidden state
        w_e : (n_e_i,)       f32  — routing weight for that token at e

        Reservoir = first max_per_expert tokens routed.  w_e is kept for
        diagnostic dumps but is NOT folded into R (see PROBLEM.md rung 2).
        """
        X = torch.cat(self.X_chunks[layer], dim=0)
        Idx = torch.cat(self.Idx_chunks[layer], dim=0)
        Sc  = torch.cat(self.Score_chunks[layer], dim=0)
        T, K = Idx.shape
        per_expert: list = [None] * self.n_experts
        flat_e = Idx.reshape(-1)
        token_idx = torch.arange(T, device=Idx.device).unsqueeze(1).expand(T, K).reshape(-1)
        flat_w = Sc.reshape(-1)
        sorted_e, perm = torch.sort(flat_e)
        sorted_token = token_idx[perm]
        sorted_w     = flat_w[perm]
        unique_e, counts = torch.unique_consecutive(sorted_e, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        for ue, st, ct in zip(unique_e.tolist(), starts.tolist(), counts.tolist()):
            n_take = min(ct, self.max_per_expert)
            rows = sorted_token[st : st + n_take]
            ws   = sorted_w[st : st + n_take]
            per_expert[int(ue)] = (X[rows], ws.float())
        return per_expert

    def free_layer(self, layer: int) -> None:
        self.X_chunks[layer] = []
        self.Idx_chunks[layer] = []
        self.Score_chunks[layer] = []


def make_gate_hook(layer_idx: int, mlp_module, collector: Collector):
    """Hook for supported MoE routers.

    Stores raw tensors on GPU for batched post-processing — no per-expert
    Python loop at hook time.
    """
    def hook(_module, args, output):
        x_in = args[0].detach().reshape(-1, args[0].shape[-1])
        if isinstance(output, tuple) and len(output) >= 3:
            out0_is_float = torch.is_tensor(output[0]) and torch.is_floating_point(output[0])
            out1_is_float = torch.is_tensor(output[1]) and torch.is_floating_point(output[1])
            out2_is_float = torch.is_tensor(output[2]) and torch.is_floating_point(output[2])
            if out1_is_float and torch.is_tensor(output[2]) and not out2_is_float:
                # Qwen router: (router_logits, topk_scores, topk_indices)
                scores, idx = output[1], output[2]
            elif torch.is_tensor(output[0]) and not out0_is_float and out1_is_float:
                # DeepSeek MoEGate: (topk_idx, topk_weight, aux_loss)
                idx, scores = output[0], output[1]
            else:
                raise TypeError(
                    "unsupported router output tuple shapes/types: "
                    f"{[(getattr(o, 'shape', None), getattr(o, 'dtype', None)) for o in output]}"
                )
        else:
            raise TypeError(f"unsupported router output type: {type(output)!r}")
        scores = scores.detach().reshape(x_in.shape[0], -1).float()
        idx = idx.detach().reshape(x_in.shape[0], -1).to(torch.int32)
        collector.add_batch(layer_idx, x_in, scores, idx)
    return hook


def expert_forward(X: torch.Tensor, Wg: torch.Tensor, Wu: torch.Tensor,
                   Wd: torch.Tensor, act=F.silu) -> torch.Tensor:
    """Qwen3-MoE expert: y = down(silu(gate(x)) * up(x)).

    X:  (n, hidden) bf16
    Wg: (intermediate, hidden) bf16
    Wu: (intermediate, hidden) bf16
    Wd: (hidden, intermediate) bf16
    """
    g = F.linear(X, Wg)
    u = F.linear(X, Wu)
    z = act(g) * u
    return F.linear(z, Wd)
