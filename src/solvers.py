"""Classical methods. Every solver returns {"x", "benefit", "time", "feasible"}."""
import numpy as np
from .metrics import package


def all_bitstrings(n):
    k = np.arange(2 ** n)
    return ((k[:, None] >> np.arange(n)) & 1).astype(np.int8)   # column q = bit q


def plan_arrays(inst, X):
    """Benefit and feasibility of every row of X (shape (K, n))."""
    t, b = np.array(inst["t"]), np.array(inst["b"])
    ben = X @ b
    feas = (X @ t) <= inst["T"]
    for i, j in inst["E"]:
        feas &= X[:, j] <= X[:, i]
    return ben, feas


def exact_solver(inst):
    X = all_bitstrings(inst["n"])
    ben, feas = plan_arrays(inst, X)
    ben = np.where(feas, ben, -np.inf)
    return package(inst, X[int(np.argmax(ben))])


def _parents(inst):
    par = {i: [] for i in range(inst["n"])}
    for i, j in inst["E"]:
        par[j].append(i)
    return par


def fixed_curriculum(inst):
    """Modules in index order; take a module if its prerequisites are taken and it fits."""
    par, t, T = _parents(inst), inst["t"], inst["T"]
    x, used = [0] * inst["n"], 0
    for i in range(inst["n"]):
        if all(x[p] for p in par[i]) and used + t[i] <= T:
            x[i], used = 1, used + t[i]
    return package(inst, x)


def greedy_basic(inst):
    """Best benefit/time ratio among modules whose prerequisites are already selected."""
    par, t, b, T = _parents(inst), inst["t"], inst["b"], inst["T"]
    x, used = [0] * inst["n"], 0
    while True:
        cands = [i for i in range(inst["n"]) if not x[i] and all(x[p] for p in par[i])
                 and used + t[i] <= T]
        if not cands:
            break
        i = max(cands, key=lambda k: b[k] / t[k])
        x[i], used = 1, used + t[i]
    return package(inst, x)


def greedy_bundle(inst):
    """Prerequisite-aware: score a module together with all its missing ancestors."""
    par, t, b, T = _parents(inst), inst["t"], inst["b"], inst["T"]

    def missing_bundle(i, x):
        seen, stack = set(), [i]
        while stack:
            k = stack.pop()
            if x[k] or k in seen:
                continue
            seen.add(k)
            stack.extend(par[k])
        return seen

    x, used = [0] * inst["n"], 0
    while True:
        best, best_ratio = None, -1.0
        for i in range(inst["n"]):
            if x[i]:
                continue
            bun = missing_bundle(i, x)
            tt = sum(t[k] for k in bun)
            if used + tt <= T:
                ratio = sum(b[k] for k in bun) / tt
                if ratio > best_ratio:
                    best, best_ratio = bun, ratio
        if best is None:
            break
        for k in best:
            x[k] = 1
        used += sum(t[k] for k in best)
    return package(inst, x)
