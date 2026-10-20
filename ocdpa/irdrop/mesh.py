"""IR-drop solver for a resistive crossbar performing a parallel analog MVM.

Wire resistance is not a perturbation at these array sizes: 128 cells of 6 kohm
in parallel present ~47 ohm to a line whose own resistance is of the same
order, so the drop has to be solved rather than estimated.

Node model. Wordline i is driven at its left end through r_row per cell pitch;
bitline j is terminated at virtual ground through r_col per cell pitch. For
every cell there are two unknown node voltages, on the wordline and on the
bitline, coupled by the cell conductance. Column current is read at the
grounded end of the bitline.

Parameters follow NeuroSim's 22 nm wire model and RRAM cell:
  unitLengthWireResistance = rho / (w^2 * AR),  rho = 4.03e-8, w = 40 nm, AR = 1.9
  1T1R adds a series access resistance of resistanceOn * IR_DROP_TOLERANCE.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

RHO, AR, FEATURE = 4.03e-8, 1.9, 40e-9
UNIT_R = RHO / (FEATURE * FEATURE * AR)
R_ON, R_OFF, IR_DROP_TOLERANCE = 6e3, 6e3 * 150, 0.25

CELLS = {
    "1T1R": (4, 12, R_ON * IR_DROP_TOLERANCE),
    "crossbar": (2, 2, 0.0),
}


def solve(conductance, v_in, r_row, r_col):
    """Column currents under wire resistance. conductance: (N, M) siemens."""
    n, m = conductance.shape
    idx_w = lambda i, j: i * m + j
    idx_b = lambda i, j: n * m + i * m + j
    size = 2 * n * m
    A = lil_matrix((size, size))
    b = np.zeros(size)
    gw, gc = 1.0 / r_row, 1.0 / r_col
    for i in range(n):
        for j in range(m):
            g = conductance[i, j]
            row = idx_w(i, j)
            A[row, idx_w(i, j)] += g
            A[row, idx_b(i, j)] -= g
            if j == 0:
                A[row, idx_w(i, j)] += gw
                b[row] += gw * v_in[i]
            else:
                A[row, idx_w(i, j)] += gw
                A[row, idx_w(i, j - 1)] -= gw
            if j < m - 1:
                A[row, idx_w(i, j)] += gw
                A[row, idx_w(i, j + 1)] -= gw
            row = idx_b(i, j)
            A[row, idx_b(i, j)] += g
            A[row, idx_w(i, j)] -= g
            if i > 0:
                A[row, idx_b(i, j)] += gc
                A[row, idx_b(i - 1, j)] -= gc
            A[row, idx_b(i, j)] += gc
            if i < n - 1:
                A[row, idx_b(i + 1, j)] -= gc
    v = spsolve(csr_matrix(A), b)
    vb = v[n * m:].reshape(n, m)
    return vb[n - 1, :] * gc


def ideal(conductance, v_in):
    return conductance.T @ v_in


def experiment(n, m, cell, binary, rng, density=0.5):
    r_pitch, c_pitch, r_series = CELLS[cell]
    r_row, r_col = UNIT_R * r_pitch * FEATURE, UNIT_R * c_pitch * FEATURE
    if binary:
        on = rng.random((n, m)) < density
        res = np.where(on, R_ON, R_OFF) + r_series
    else:
        level = rng.integers(0, 4, (n, m))
        g = (1 / R_OFF) + level * ((1 / R_ON) - (1 / R_OFF)) / 3
        res = 1.0 / g + r_series
    conductance = 1.0 / res
    v_in = rng.random(n) * 0.5
    got, want = solve(conductance, v_in, r_row, r_col), ideal(conductance, v_in)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))
