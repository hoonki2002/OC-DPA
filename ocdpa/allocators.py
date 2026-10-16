"""Ways to spend a fixed plane budget over a token's routed experts.

All of them satisfy sum_e k_e = B with k_e in [KMIN, KMAX], so the ADC
conversion count -- and therefore the energy -- is identical across them and
only the distribution differs.

  uniform    every routed expert gets B / n
  DPA        per-token knapsack on gate^2 * residual, the published objective
  DPAGreedy  the same optimum from a marginal-gain rule (exact when the
             residual curves are convex)
  DPAHardware the greedy rule on the allocator's fixed-point, log-domain
             datapath (what the on-chip allocator computes)
"""
from __future__ import annotations

import torch

KMIN, KMAX = 2, 8


def knapsack(cost, budget, n, kmin=KMIN, kmax=KMAX):
    """cost[i][k] -> minimal total cost with sum k = budget. Returns list of k."""
    INF = float("inf")
    f = [[INF] * (budget + 1) for _ in range(n + 1)]
    back = [[0] * (budget + 1) for _ in range(n + 1)]
    f[0][0] = 0.0
    for i in range(1, n + 1):
        row_prev, row, bk = f[i - 1], f[i], back[i]
        for b in range(budget + 1):
            best, best_k = INF, 0
            for k in range(kmin, min(kmax, b) + 1):
                prev = row_prev[b - k]
                if prev < INF:
                    c = prev + cost[i - 1][k]
                    if c < best:
                        best, best_k = c, k
            row[b], bk[b] = best, best_k
    if f[n][budget] == INF:
        return None
    ks, b = [0] * n, budget
    for i in range(n, 0, -1):
        ks[i - 1] = back[i][b]; b -= back[i][b]
    return ks


class Uniform:
    def __init__(self, kbar):
        self.kbar = kbar

    def __call__(self, layer, expert, tokens, pos, gates, residual):
        return torch.full_like(tokens, self.kbar)


def batched_knapsack(cost, budget, kmin=KMIN, kmax=KMAX):
    """Solve one multi-choice knapsack per row of `cost` on the device.

    cost: (T, n, kmax+1). Returns (T, n) integer plane counts summing to
    `budget`. The serial version spends more time in the allocator than the
    array spends converting, which is the wrong shape for a sweep.
    """
    T, n, _ = cost.shape
    INF = torch.finfo(cost.dtype).max / 4
    f = cost.new_full((T, budget + 1), INF)
    f[:, 0] = 0.0
    back = torch.zeros((T, n, budget + 1), dtype=torch.uint8, device=cost.device)
    for i in range(n):
        nxt = cost.new_full((T, budget + 1), INF)
        choice = torch.zeros((T, budget + 1), dtype=torch.uint8, device=cost.device)
        for k in range(kmin, kmax + 1):
            if k > budget:
                break
            cand = f[:, : budget + 1 - k] + cost[:, i, k].unsqueeze(1)
            better = cand < nxt[:, k:]
            nxt[:, k:] = torch.where(better, cand, nxt[:, k:])
            choice[:, k:] = torch.where(better, torch.full_like(choice[:, k:], k),
                                        choice[:, k:])
        f, back[:, i] = nxt, choice
    ks = torch.zeros((T, n), dtype=torch.long, device=cost.device)
    b = torch.full((T,), budget, dtype=torch.long, device=cost.device)
    rows = torch.arange(T, device=cost.device)
    for i in range(n - 1, -1, -1):
        k = back[rows, i, b].long()
        ks[:, i] = k
        b = b - k
    return ks


class DPA:
    """Per-dispatch knapsack on gate^2 * residual, as published."""

    def __init__(self, kbar, residual, topk, device="cuda"):
        self.kbar, self.topk = kbar, topk
        self.res = torch.as_tensor(residual, dtype=torch.float32, device=device)
        self.plan = {}

    def prepare(self, layer, top_k_index, top_k_weights):
        w = (top_k_weights.float() ** 2)                       # (T, topk)
        cost = w.unsqueeze(-1) * self.res[layer][top_k_index]  # (T, topk, K+1)
        self.plan[layer] = batched_knapsack(cost, self.kbar * self.topk)

    def __call__(self, layer, expert, tokens, pos, gates, residual):
        return self.plan[layer][tokens, pos]


def batched_greedy(cost, budget, kmin=KMIN, kmax=KMAX, hull=True):
    """Marginal-gain allocator: the knapsack without the DP table.

    Every expert starts at kmin -- which the budget always admits, since
    B = kbar * n with kbar >= kmin -- and the remaining B - n*kmin units go one
    at a time to whichever expert currently offers the largest drop in cost.
    With `hull`, each expert's cost curve is first replaced by its lower convex
    hull, which makes the marginal gains non-increasing in k and the greedy
    choice exactly optimal for the continuous relaxation; the integer solution
    it returns differs from the DP's by at most one unit of budget.

    In hardware this is a max-select over n candidates repeated B - n*kmin
    times -- a priority encoder and an accumulator -- instead of the DP's
    O(n*B*krange) table walk.
    """
    T, n, _ = cost.shape
    c = cost[:, :, kmin:kmax + 1].clone()                     # (T, n, K)
    K = c.shape[2]
    if hull:
        # Lower convex hull along k: force the decrements to be non-increasing
        # by averaging any adjacent pair that violates it (pool-adjacent-
        # violators on the first difference).
        for _ in range(K):
            d = c[:, :, :-1] - c[:, :, 1:]                    # gains, (T, n, K-1)
            bad = d[:, :, :-1] < d[:, :, 1:]                  # gain grows with k
            if not bool(bad.any()):
                break
            mid = (c[:, :, :-2] + c[:, :, 2:]) / 2
            c[:, :, 1:-1] = torch.where(bad, mid, c[:, :, 1:-1])
    gains = c[:, :, :-1] - c[:, :, 1:]                        # (T, n, K-1)
    ks = torch.full((T, n), kmin, dtype=torch.long, device=cost.device)
    rows = torch.arange(T, device=cost.device)
    NEG = torch.finfo(cost.dtype).min / 4
    for _ in range(budget - n * kmin):
        idx = (ks - kmin).clamp(max=K - 2)                    # gain of k -> k+1
        g = gains[rows.unsqueeze(1), torch.arange(n, device=cost.device).unsqueeze(0), idx]
        g = torch.where(ks < kmax, g, torch.full_like(g, NEG))
        pick = g.argmax(1)
        ks[rows, pick] += 1
    return ks


class DPAGreedy(DPA):
    """DPA with the DP replaced by the marginal-gain allocator."""

    def __init__(self, kbar, residual, topk, device="cuda", hull=True):
        super().__init__(kbar, residual, topk, device)
        self.hull = hull

    def prepare(self, layer, top_k_index, top_k_weights):
        w = (top_k_weights.float() ** 2)
        cost = w.unsqueeze(-1) * self.res[layer][top_k_index]
        self.plan[layer] = batched_greedy(cost, self.kbar * self.topk, hull=self.hull)


# --------------------------------------------------------------------------
# Hardware allocator: the greedy above, in the arithmetic a circuit can hold.
#
# The comparison that drives the allocator is g_i^2 * dR_i(k_i) against the same
# product for every other routed expert. Only the ordering matters, so the
# datapath works in the log domain, which turns the multiply into an add -- the
# only arithmetic primitive NeuroSim's library offers -- and compresses gains
# that span many orders of magnitude into a few bits. With the gate
# g_i = exp(z_i) / Z (z_i the pre-softmax router logit), log g_i = z_i - log Z and the
# normaliser Z is common to all candidates, so the term is 2 z_i up to a constant.
# Subtracting the largest routed logit z_1 as well keeps it non-positive and bounds
# the word width:
#
#   s_i = 2 (z_i - z_1) + log dR[L, e_i, k_i]
#
# The second term is read from the residual LUT, which stores log gains rather than
# residuals so that no subtractor is needed at run time. The model hands the
# experts the gates, not the logits, so z_i - z_1 is recovered as log g_i - log g_max,
# which is the same number.
# --------------------------------------------------------------------------

def fixed(x, frac_bits, int_bits):
    """Two's-complement fixed point, saturating."""
    lo, hi = -(2.0 ** (int_bits - 1)), 2.0 ** (int_bits - 1) - 2.0 ** -frac_bits
    return torch.clamp(torch.round(x * 2 ** frac_bits) / 2 ** frac_bits, lo, hi)


class DPAHardware(DPA):
    """Greedy marginal-gain allocation in quantised log arithmetic (logs in base 2).

    `lut_frac`/`lut_int` size the stored log gains, `gate_frac`/`gate_int` the logit term
    2 (z_i - z_1). The default widths are the ones measured to leave the allocation unchanged.
    """

    def __init__(self, kbar, residual, topk, device="cuda",
                 lut_frac=3, lut_int=7, gate_frac=3, gate_int=5):
        super().__init__(kbar, residual, topk, device)
        self.gate_frac, self.gate_int = gate_frac, gate_int
        r = self.res                                        # (L, E, KMAX+1)
        gain = (r[:, :, KMIN:KMAX] - r[:, :, KMIN + 1:KMAX + 1]).clamp_min(0)
        # The LUT is normalised per layer so the stored codes use the full
        # range; a per-layer constant shifts every expert of a dispatch alike
        # and so cannot change the ordering.
        ref = gain.amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        self.lut = fixed(torch.log2((gain / ref).clamp_min(1e-30)), lut_frac, lut_int)
        self.n_entries = int(self.lut.numel())
        self.lut_bits = lut_frac + lut_int

    def prepare(self, layer, top_k_index, top_k_weights):
        g = top_k_weights.float().clamp_min(1e-30)                 # (T, n)
        lg = torch.log2(g)                                         # z_i up to a common constant
        s_gate = fixed(2.0 * (lg - lg.amax(1, keepdim=True)), self.gate_frac, self.gate_int)   # 2 (z_i - z_1), (T, n)
        lut = self.lut[layer][top_k_index]                         # (T, n, K-1)
        T, n = g.shape
        ks = torch.full((T, n), KMIN, dtype=torch.long, device=g.device)
        rows = torch.arange(T, device=g.device)
        cols = torch.arange(n, device=g.device)
        NEG = -1e9
        steps = self.kbar * self.topk - n * KMIN
        # Index in the low bits of the compared word: the max-select returns the
        # winning expert directly and ties break deterministically.
        for _ in range(steps):
            idx = (ks - KMIN).clamp(max=lut.shape[2] - 1)
            s = s_gate + lut[rows.unsqueeze(1), cols.unsqueeze(0), idx]
            s = torch.where(ks < KMAX, s, torch.full_like(s, NEG))
            self.plan_steps = steps
            ks[rows, s.argmax(1)] += 1
        self.plan[layer] = ks
