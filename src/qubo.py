"""QUBO / Ising encoding of the module-selection problem.

Variables z = [x_0..x_{n-1}, s_0..s_{K-1}], K = ceil(log2(T+1)) slack bits.
Energy: E(z) = -sum b_i x_i + P1 * sum_(i,j) x_j (1 - x_i) + P2 * (sum t_i x_i + sum 2^k s_k - T)^2
Qubit q <-> bit q of the basis-state index (little-endian, same as Qiskit).
"""
import math
import numpy as np
from .solvers import all_bitstrings, plan_arrays


def default_penalties(inst, c1=1.5, c2=1.0):
    bmax = max(inst["b"])
    return c1 * bmax, c2 * bmax


def build_qubo(inst, P1, P2):
    n, T = inst["n"], inst["T"]
    K = max(1, math.ceil(math.log2(T + 1)))
    N = n + K
    a = np.array(list(inst["t"]) + [2 ** k for k in range(K)], dtype=float)
    Q = np.zeros((N, N))
    for i in range(n):
        Q[i, i] -= inst["b"][i]
    for i, j in inst["E"]:                      # x_j - x_i x_j
        Q[j, j] += P1
        Q[min(i, j), max(i, j)] -= P1
    for i in range(N):                          # (a.z - T)^2 expanded, z^2 = z
        Q[i, i] += P2 * (a[i] ** 2 - 2 * T * a[i])
        for j in range(i + 1, N):
            Q[i, j] += 2 * P2 * a[i] * a[j]
    const = P2 * T ** 2
    return Q, const, K, N


def cost_diagonal(Q, const, N):
    Z = all_bitstrings(N).astype(float)
    return (Z @ Q * Z).sum(axis=1) + const, Z


def to_ising(Q, const):
    """z = (1 - s)/2. Returns h (N,), J {(i,j): Jij}, offset with E = offset + sum h_i s_i + sum J_ij s_i s_j."""
    N = Q.shape[0]
    h = -np.diag(Q) / 2
    J, off = {}, np.trace(Q) / 2 + const
    for i in range(N):
        for j in range(i + 1, N):
            if Q[i, j] != 0:
                J[(i, j)] = Q[i, j] / 4
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
                off += Q[i, j] / 4
    return h, J, off


def ising_diagonal(h, J, off, N):
    Z = all_bitstrings(N)
    S = 1 - 2 * Z.astype(float)
    E = off + S @ h
    for (i, j), v in J.items():
        E += v * S[:, i] * S[:, j]
    return E


def plan_info(inst, N):
    """Benefit and x-feasibility for each of the 2^N basis states (x = low n bits)."""
    X = all_bitstrings(N)[:, :inst["n"]]
    return plan_arrays(inst, X)
