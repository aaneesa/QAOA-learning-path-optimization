"""Instance generation, benefit model, JSON I/O."""
import json
import numpy as np

PROFILES = ("beginner", "partial", "goal_focused")


def benefit(w, m, r):
    """b_i = w_i (1 - m_i) r_i"""
    return np.asarray(w) * (1 - np.asarray(m)) * np.asarray(r)


def generate_instance(n, density, rho, profile, seed, t_max=4, w_max=10):
    """Random instance. Edge (i, j) with i < j means i is a prerequisite of j (DAG by construction).
    T = rho * sum(t). Module indices are the 'fixed curriculum' order."""
    rng = np.random.default_rng(seed)
    t = rng.integers(1, t_max + 1, size=n)
    w = rng.integers(1, w_max + 1, size=n).astype(float)
    if profile == "beginner":
        m, r = rng.uniform(0.0, 0.2, n), rng.uniform(0.5, 1.0, n)
    elif profile == "partial":
        m, r = rng.uniform(0.0, 1.0, n), rng.uniform(0.5, 1.0, n)
    elif profile == "goal_focused":
        m = rng.uniform(0.0, 0.6, n)
        focus = rng.random(n) < 0.4
        r = np.where(focus, rng.uniform(0.9, 1.0, n), rng.uniform(0.1, 0.3, n))
    else:
        raise ValueError(f"unknown profile {profile}")
    E = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < density]
    T = max(int(t.min()), int(round(rho * t.sum())))
    b = np.round(benefit(w, m, r), 4)
    return dict(n=int(n), t=[int(v) for v in t], b=[float(v) for v in b],
                E=[(int(i), int(j)) for i, j in E], T=int(T),
                density=float(density), rho=float(rho), profile=profile, seed=int(seed))


def trap_instance():
    """Hand-made instance: optimum 16, basic greedy 11, fixed 11, bundle greedy 15."""
    return dict(n=6, t=[2, 2, 2, 1, 3, 3], b=[3, 3, 3, 2, 1, 12], E=[(4, 5)], T=8,
                density=0.0, rho=0.0, profile="trap", seed=0)


def save_instances(instances, path):
    with open(path, "w") as f:
        json.dump(instances, f)


def load_instances(path):
    with open(path) as f:
        data = json.load(f)
    for d in data:
        d["E"] = [tuple(e) for e in d["E"]]
    return data
