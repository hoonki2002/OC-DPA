"""Chunk-byte unpacking and per-expert residual computation."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
from safetensors import safe_open

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parent.parent.parent   # profiling → 3_dynamic_precision → experiments → DPA-MoE

from dpa_moe.format import HEADER_STRUCT, MAGIC  # noqa: E402

sys.path.insert(0, str(_HERE))
from _collector import expert_forward  # noqa: E402


def unpack_chunk_bytes(b: bytes) -> dict:
    """Parse one packed-byte stream (ss_anybcq) into (B, Alpha, beta)."""
    h = HEADER_STRUCT.unpack(b[:HEADER_STRUCT.size])
    magic, version, gs, base_p, target_p, flags, n, m, n_groups_per_row, d1 = h
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:x}")
    P = target_p - base_p
    fp16 = 2
    cursor = HEADER_STRUCT.size
    beta = np.frombuffer(b[cursor:cursor + d1 * fp16], dtype=np.float16).astype(np.float32)
    cursor += d1 * fp16
    sign_bytes = (d1 * gs + 7) // 8
    alpha_bytes = d1 * fp16
    B_arr = np.empty((d1, gs, P), dtype=np.int8)
    Alpha = np.empty((d1, P), dtype=np.float32)
    for p in range(P):
        sb = np.frombuffer(b[cursor:cursor + sign_bytes], dtype=np.uint8)
        cursor += sign_bytes
        bits = np.unpackbits(sb, bitorder="big")[:d1 * gs]
        B_arr[:, :, p] = (bits.astype(np.int8) * 2 - 1).reshape(d1, gs)
        Alpha[:, p] = np.frombuffer(b[cursor:cursor + alpha_bytes],
                                     dtype=np.float16).astype(np.float32)
        cursor += alpha_bytes
    return dict(B=B_arr, Alpha=Alpha, beta=beta,
                n_out=n, m_in=m, n_groups_per_row=n_groups_per_row,
                P=P, gs=gs)


def reconstruct_W_K(cb: dict, K: int, device,
                    dtype=torch.bfloat16) -> torch.Tensor:
    """Reconstruct (n_out, m_in) weight at K kept planes.

    W[r, g*gs+j] = β[r*ng+g] + Σ_{p<K} α[r*ng+g, p] · B[r*ng+g, j, p]
    """
    B = torch.from_numpy(cb["B"]).to(device=device, dtype=torch.float32)
    A = torch.from_numpy(cb["Alpha"]).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(cb["beta"]).to(device=device, dtype=torch.float32)
    d1, gs, P = B.shape
    assert 0 < K <= P
    W = b.unsqueeze(1).expand(d1, gs).contiguous()
    if K > 0:
        W = W + (A[:, :K].unsqueeze(1) * B[:, :, :K]).sum(dim=2)
    n_out = cb["n_out"]
    ng = cb["n_groups_per_row"]
    m_in = cb["m_in"]
    return W.view(n_out, ng * gs)[:, :m_in].to(dtype)


def _plane_deltas(cb: dict, device):
    """Pre-compute α·B per plane for incremental reconstruction.

    Returns (base, deltas, n_out, ng, gs, m_in, P) where:
      base   : (d1, gs)    — β baseline
      deltas : (d1, gs, P) — per-plane α·B increments
    """
    B = torch.from_numpy(cb["B"]).to(device=device, dtype=torch.float32)
    A = torch.from_numpy(cb["Alpha"]).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(cb["beta"]).to(device=device, dtype=torch.float32)
    d1, gs, P = B.shape
    deltas = A.unsqueeze(1) * B
    base = b.unsqueeze(1).expand(d1, gs).contiguous()
    return base, deltas, cb["n_out"], cb["n_groups_per_row"], gs, cb["m_in"], P


def compute_layer_residuals(
    layer_idx: int,
    safet_path: pathlib.Path,
    hf_mlp,
    collector_layer: list,
    K_min: int,
    K_max: int,
    base_p: int,
    device,
    log,
) -> np.ndarray:
    """Compute R: (n_experts, K_max − K_min + 1) bare squared-L2 residuals.

    Indexing convention (PROBLEM.md §1):
        K  =  total bitplanes used  ∈ [K_min = base_p, K_max = target_p]
        W_e^{(K)} = β + Σ_{p < K − base_p} α_p · B_p

    R[e, K-K_min] = E_calib[‖ΔE_e(K, x_t)‖²] — bare (gate weight NOT folded
    in); the allocator applies gate² at runtime (see PROBLEM.md rung 2).
    """
    n_experts = len(collector_layer)
    n_K = K_max - K_min + 1
    R = np.full((n_experts, n_K), np.nan, dtype=np.float32)

    with safe_open(safet_path, framework="pt") as fp:
        keys = set(fp.keys())
        with torch.no_grad():
            experts = hf_mlp.experts
            fused_experts = hasattr(experts, "gate_up_proj")
            for e in range(n_experts):
                entry = collector_layer[e]
                if entry is None:
                    continue
                X, w_per_tok = entry
                if X is None or X.numel() == 0:
                    continue
                Xg = X.to(device=device, dtype=torch.bfloat16)
                if fused_experts:
                    gate_up = experts.gate_up_proj    # (E, 2I, H)
                    down_p  = experts.down_proj       # (E, H, I)
                    _, two_I, _ = gate_up.shape
                    I = two_I // 2
                    Wg_orig = gate_up[e, :I, :].to(device=device, dtype=torch.bfloat16)
                    Wu_orig = gate_up[e, I:, :].to(device=device, dtype=torch.bfloat16)
                    Wd_orig = down_p[e].to(device=device, dtype=torch.bfloat16)
                else:
                    expert = experts[e]
                    Wg_orig = expert.gate_proj.weight.to(device=device, dtype=torch.bfloat16)
                    Wu_orig = expert.up_proj.weight.to(device=device, dtype=torch.bfloat16)
                    Wd_orig = expert.down_proj.weight.to(device=device, dtype=torch.bfloat16)
                y_orig = expert_forward(Xg, Wg_orig, Wu_orig, Wd_orig).to(torch.float32)
                cbs = {}
                for kind in ("gate", "up", "down"):
                    key = f"e{e:03d}.{kind}.bytes"
                    if key not in keys:
                        raise KeyError(key)
                    b = fp.get_tensor(key).numpy().tobytes()
                    cbs[kind] = unpack_chunk_bytes(b)
                base_g, dlt_g, n_g, ng_g, gs_g, m_g, P_g = _plane_deltas(cbs["gate"], device)
                base_u, dlt_u, n_u, ng_u, gs_u, m_u, P_u = _plane_deltas(cbs["up"],   device)
                base_d, dlt_d, n_d, ng_d, gs_d, m_d, P_d = _plane_deltas(cbs["down"], device)
                W_g = base_g.clone()
                W_u = base_u.clone()
                W_d = base_d.clone()
                for K in range(K_min, K_max + 1):
                    k_extras = K - base_p
                    if k_extras >= 1:
                        W_g = W_g + dlt_g[:, :, k_extras - 1]
                        W_u = W_u + dlt_u[:, :, k_extras - 1]
                        W_d = W_d + dlt_d[:, :, k_extras - 1]
                    Wg = W_g.view(n_g, ng_g * gs_g)[:, :m_g].to(torch.bfloat16)
                    Wu = W_u.view(n_u, ng_u * gs_u)[:, :m_u].to(torch.bfloat16)
                    Wd = W_d.view(n_d, ng_d * gs_d)[:, :m_d].to(torch.bfloat16)
                    y_K = expert_forward(Xg, Wg, Wu, Wd).to(torch.float32)
                    delta = (y_orig.reshape(Xg.shape[0], -1) -
                             y_K.reshape(Xg.shape[0], -1))
                    sq_l2 = (delta * delta).sum(dim=-1)
                    R[e, K - K_min] = float(sq_l2.mean().item())
                del W_g, W_u, W_d, dlt_g, dlt_u, dlt_d, base_g, base_u, base_d
                del Wg_orig, Wu_orig, Wd_orig, y_orig
                torch.cuda.empty_cache()
        del keys
    log(f"  layer {layer_idx}: residuals done")
    return R
