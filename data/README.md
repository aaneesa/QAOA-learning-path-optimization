# Dataset

Synthetic. No real learner data. Regenerate with `python -m src.make_dataset` (fixed seeds).

| File | Contents |
|---|---|
| instances_main.json/.csv | 541 instances: n=8, 3 profiles x rho {0.3,0.5,0.7} x density {0.2,0.4} x 30 seeds, plus 1 hand-made trap instance (last, optimum 16) |
| instances_scaling.json/.csv | 240 instances: n in {6,8,10,12} x 3 profiles x 20 seeds, rho=0.5, density=0.3 (10 to 17 qubits) |
| case_study.json | Illustrative 10-module data-science path with 3 learner profiles (values are assumptions) |

Fields: id, n, t (study times), b (benefit = w(1-m)r), E (edges (i,j): i is a prerequisite of j), T (budget), density, rho, profile, seed.
Generator ranges: t in 1..4, w in 1..10; mastery/relevance distributions per profile are in src/instances.py.
