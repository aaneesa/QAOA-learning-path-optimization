"""Metrics exactly as defined in the Problem Statement."""
import numpy as np


def total_benefit(inst, x):
    return float(np.dot(inst["b"], x))


def time_used(inst, x):
    return int(np.dot(inst["t"], x))


def utilization(inst, x):
    return time_used(inst, x) / inst["T"]


def prereq_fraction(inst, x):
    """Fraction of prerequisite edges satisfied (x_j <= x_i)."""
    E = inst["E"]
    return 1.0 if not E else sum(x[j] <= x[i] for i, j in E) / len(E)


def is_feasible(inst, x):
    return time_used(inst, x) <= inst["T"] and prereq_fraction(inst, x) == 1.0


def optimality_gap(b_opt, b_method):
    return float("nan") if b_opt <= 0 else (b_opt - b_method) / b_opt


def package(inst, x):
    x = [int(v) for v in x]
    return dict(x=x, benefit=total_benefit(inst, x), time=time_used(inst, x),
                feasible=bool(is_feasible(inst, x)))
