"""Generate the fixed project dataset. Run once:  python -m src.make_dataset
Outputs (in data/):
  instances_main.json   main set: n=8, 3 profiles x 3 rho x 2 densities x 30 seeds + trap
  instances_scaling.json scalability set: n in {6,8,10,12}, rho 0.5, density 0.3, 3 profiles x 20 seeds
  case_study.json       hand-built 10-module data-science path, 3 learner profiles
  instances_main.csv / instances_scaling.csv   flat copies for inspection
"""
import csv
import os
from .instances import (PROFILES, generate_instance, trap_instance, benefit,
                        save_instances)

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def main_set():
    out = []
    for profile in PROFILES:
        for rho in (0.3, 0.5, 0.7):
            for density in (0.2, 0.4):
                for s in range(30):
                    inst = generate_instance(8, density, rho, profile, seed=10_000 + len(out))
                    inst["id"] = len(out)
                    out.append(inst)
    t = trap_instance(); t["id"] = len(out); out.append(t)
    return out


def scaling_set():
    out = []
    for n in (6, 8, 10, 12):
        for profile in PROFILES:
            for s in range(20):
                inst = generate_instance(n, 0.3, 0.5, profile, seed=20_000 + len(out))
                inst["id"] = 1000 + len(out)
                out.append(inst)
    return out


# ---- illustrative case study: data-science learning path (values are assumptions, not real data)
MODULES = ["Python basics", "Linear algebra", "Probability & statistics", "Data handling (pandas)",
           "Calculus & optimization", "Supervised learning", "Model evaluation",
           "Neural networks", "Deep learning for vision", "ML deployment"]
T_MOD = [2, 3, 3, 2, 3, 4, 2, 4, 3, 2]
W_MOD = [6, 7, 8, 6, 6, 9, 7, 9, 6, 5]
EDGES = [(0, 3), (1, 4), (2, 5), (3, 5), (5, 6), (4, 7), (5, 7), (7, 8), (6, 9)]
LEARNERS = {
    "beginner":     dict(m=[0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0], r=[0.8] * 10),
    "partial":      dict(m=[0.9, 0.4, 0.7, 0.6, 0.3, 0.1, 0, 0, 0, 0], r=[0.7] * 10),
    "goal_vision":  dict(m=[0.9, 0.4, 0.7, 0.6, 0.3, 0.1, 0, 0, 0, 0],
                         r=[1.0, 1.0, 0.4, 0.5, 1.0, 1.0, 0.3, 1.0, 1.0, 0.2]),
}


def case_study():
    out = []
    for name, L in LEARNERS.items():
        b = [round(float(v), 4) for v in benefit(W_MOD, L["m"], L["r"])]
        out.append(dict(id=2000 + len(out), n=10, t=T_MOD, b=b, E=EDGES,
                        T=round(0.5 * sum(T_MOD)), density=len(EDGES) / 45, rho=0.5,
                        profile=name, seed=0, modules=MODULES, mastery=L["m"], relevance=L["r"]))
    return out


def to_csv(instances, path):
    keys = ["id", "n", "profile", "density", "rho", "T", "seed", "t", "b", "E"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for d in instances:
            w.writerow([d[k] if k not in ("t", "b", "E") else str(d[k]) for k in keys])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main, scal, case = main_set(), scaling_set(), case_study()
    save_instances(main, f"{OUT}/instances_main.json")
    save_instances(scal, f"{OUT}/instances_scaling.json")
    save_instances(case, f"{OUT}/case_study.json")
    to_csv(main, f"{OUT}/instances_main.csv")
    to_csv(scal, f"{OUT}/instances_scaling.csv")
    print(len(main), len(scal), len(case))
