"""The any-precision encoder: binary-coded quantization with scales that nest across bit-widths.

An expert matrix W (rows x columns) is cut into groups of `group_size` inputs per row and each group is
written as

    W^(k) ~ beta + sum_{p=1..k} alpha_p * S_p ,    S_p in {-1, +1}^group_size ,

so that the first k sign planes with their scales form a valid k-bit model for every k from the base
precision up to the target precision. The fit follows the progressive precision expansion of AnyBCQ, here weight-only and with one scale set shared by all precisions:

  1. greedy initialisation of `base_precision` planes (each plane takes the sign of the residual, its scale
     is the mean magnitude of the residual),
  2. alternating refinement of the base planes: the scales and the offset beta are re-fitted by least
     squares (conjugate gradient), then the signs are re-assigned by the nearest representable value
     (binary search over the 2^k levels),
  3. one plane at a time, a new sign plane is added and scales and offset are re-fitted, so that the earlier
     planes keep their meaning; the signs are re-assigned at the end by the same binary search.

The kernels (greedy initialisation, least-squares refinement, nearest-level assignment, conjugate gradient)
are adapted from the reference implementation of AnyBCQ,

    AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs (ICLR 2026),
    https://github.com/naver-aics/anybcq, Copyright (c) 2025-present NAVER Cloud Corp., Apache License 2.0,
    which itself refers to https://github.com/allenbai01/ProxQuant.

Changes: only the asymmetric (offset) variant without sample weights is kept, the kernels and the progressive
expansion are gathered in one module, and unrelated code (training, packing, evaluation) is dropped. The
upstream license and notice are in `licenses/`.
"""
from __future__ import annotations

import numpy as np
import torch

FP16_MAX = torch.finfo(torch.float16).max


def _sign_vectors(n_bits):
    """All 2^n sign vectors of length n, +1 first (the order the nearest-level search relies on)."""
    vecs = [[]]
    for _ in range(n_bits):
        vecs = [[1.0] + v for v in vecs] + [[-1.0] + v for v in vecs]
    return vecs


def _nearest_signs(w, alpha):
    """For every entry of w (rows x cols) the sign vector whose level sum_p alpha_p S_p is nearest to it.

    alpha: (rows, n_bits). The 2^n_bits levels of each row are sorted and an entry is located by a binary
    search over their midpoints. Returns (rows, cols, n_bits) in {-1, +1}.
    """
    n_bits = alpha.shape[-1]
    codes = torch.from_numpy(np.vstack(_sign_vectors(n_bits)).astype(np.float32)).to(w.device)
    rows, cols = w.shape
    row_idx = torch.arange(rows, dtype=torch.long, device=w.device).view(rows, 1).repeat([1, cols]).view(-1)
    levels = alpha.mm(codes.t())
    levels_sorted, order = torch.sort(levels)
    flat = w.reshape(-1)
    left = torch.zeros(rows * cols, dtype=torch.long, device=w.device)
    right = torch.ones(rows * cols, dtype=torch.long, device=w.device) * (2 ** n_bits - 1)
    for _ in range(n_bits):
        mid_left = torch.div(left + right - 1, 2, rounding_mode="trunc")
        mid_right = mid_left + 1
        mid_val = (levels_sorted[row_idx, mid_left] + levels_sorted[row_idx, mid_right]) / 2
        go_left = flat < mid_val
        right[go_left] = mid_left[go_left]
        left[~go_left] = mid_right[~go_left]
    return codes[order[row_idx, left].view(rows, cols), :]


def _conjugate_gradient(A, b, x):
    """Solve A x = b for a batch of small symmetric systems (A: rows x k x k) with k CG steps from x."""
    rows, k, _ = A.shape
    x = x.clone().view(rows, k, 1)
    b = b.view(rows, k, 1)
    r = b - A.bmm(x)
    rr_new = r.transpose(1, 2).bmm(r)
    p = r.clone()
    for _ in range(k):
        rr = rr_new
        Ap = A.bmm(p)
        step = rr / (p.transpose(1, 2).bmm(Ap) + 1e-6)
        x += step * p
        r -= step * Ap
        rr_new = r.transpose(1, 2).bmm(r)
        p = r + rr_new / (rr + 1e-6) * p
    return x.view(rows, k)


def _fit_scales(w, B, alpha, beta):
    """Least-squares scales and offset for fixed signs: minimise ||w - beta - B alpha||^2 per row."""
    rows, cols = w.shape
    k = B.shape[-1]
    Bt = B.transpose(1, 2)
    ones = torch.ones((rows, cols), device=w.device, dtype=w.dtype)
    A = torch.zeros((rows, k + 1, k + 1), device=w.device, dtype=w.dtype)
    A[:, :k, :k] = Bt.bmm(B)
    A[:, :k, -1] = A[:, -1, :k] = Bt.bmm(ones.unsqueeze(-1)).view(rows, k)
    A[:, -1, -1] = ones.sum(dim=1)
    rhs = torch.zeros((rows, k + 1), device=w.device, dtype=w.dtype)
    rhs[:, :k] = Bt.bmm(w.unsqueeze(-1)).view(rows, k)
    rhs[:, -1] = (ones * w).sum(dim=1)
    x0 = torch.zeros_like(rhs)
    x0[:, :k] = alpha
    x0[:, -1] = beta
    return _conjugate_gradient(A, rhs, x0)


def _greedy_init(w, n_bits):
    """Each plane takes the sign of the residual and its mean magnitude as scale; beta is the row mean."""
    rows, cols = w.shape
    B = torch.zeros((rows, cols, n_bits), device=w.device)
    alpha = torch.zeros((rows, n_bits), device=w.device)
    beta = w.mean(dim=1, keepdim=True)
    r = (w - beta).clone()
    for i in range(n_bits):
        b = r.sign()
        a = r.abs().mean(dim=1, keepdim=True)
        r -= b * a
        B[:, :, i] = b
        alpha[:, i] = a.view(-1)
    return B, alpha, beta.view(-1)


def _refine(w, B, alpha, beta, use_bst=True):
    """One round: re-fit scales and offset, sort the scales, re-assign the signs."""
    w = w.float()
    with torch.no_grad():
        theta = _fit_scales(w, B, alpha, beta)
        k = B.shape[-1]
        alpha_new, _ = theta[:, :k].abs().sort(descending=True)
        beta_new = theta[:, -1]
        r = w - beta_new.view(-1, 1)
        if use_bst:
            B_new = _nearest_signs(r, alpha_new)
        else:
            B_new = torch.zeros_like(B)
            for i in range(k):
                B_new[:, :, i] = r.sign()
                r -= B_new[:, :, i] * alpha_new[:, i].view([-1, 1])
    return B_new, alpha_new, beta_new


def _add_plane(w, B_fixed, alpha, beta, rounds=20, use_bst=True):
    """Extend the representation by one plane: alternate new-sign / re-fit, keeping the earlier planes."""
    w = w.float()
    rows, cols = w.shape
    k_new = B_fixed.shape[2] + 1
    alpha_ext = torch.cat([alpha.float(), torch.zeros(rows, 1, device=w.device)], dim=1)
    centered = w - beta.float().view(-1, 1)
    for _ in range(rounds):
        zero_plane = torch.zeros(rows, cols, 1, device=w.device)
        recon = torch.einsum("ijk,ik->ij", torch.cat([B_fixed.float(), zero_plane], dim=-1), alpha_ext)
        new_plane = (centered - recon).sign()
        B_all = torch.cat([B_fixed.float(), new_plane.unsqueeze(-1)], dim=-1)
        theta = _fit_scales(w, B_all, alpha_ext, beta.float())
        alpha_ext = theta[:, :k_new].clamp(-FP16_MAX, FP16_MAX)
        beta = theta[:, -1]
        centered = w - beta.view(-1, 1)
    if use_bst and k_new <= 8:
        alpha_ext, _ = alpha_ext.abs().sort(descending=True)
        B_all = _nearest_signs(centered, alpha_ext)
    return B_all, alpha_ext, beta


def compute_codebook(W, *, base_precision=2, target_precision=8, group_size=128,
                     refine_rounds=20, expansion_rounds=20, use_bst=True):
    """Encode one weight matrix W (out_features x in_features) into nested sign planes.

    Returns (B, alpha, beta, n, m, groups_per_row) with
        B      (n * groups_per_row, group_size, target_precision)  signs in {-1, +1}, plane 1 first
        alpha  (n * groups_per_row, target_precision)              descending scales
        beta   (n * groups_per_row,)                               offsets
        n, m   the shape of W
    W stays on the device it was given. Input columns are zero-padded to a multiple of group_size.
    """
    W_f = W.detach().float()
    n, m = W_f.shape
    groups_per_row = -(-m // group_size)
    padded = torch.zeros(n, groups_per_row * group_size, device=W_f.device)
    padded[:, :m] = W_f
    w = padded.reshape(-1, group_size)

    B, alpha, beta = _greedy_init(w, base_precision)
    if refine_rounds > 0 and base_precision > 1:
        for _ in range(refine_rounds):
            B, alpha, beta = _refine(w, B, alpha, beta, use_bst=use_bst)
    for _ in range(base_precision + 1, target_precision + 1):
        B, alpha, beta = _add_plane(w, B, alpha, beta, rounds=expansion_rounds, use_bst=use_bst)
    return B, alpha, beta, n, m, groups_per_row
