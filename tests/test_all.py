import numpy as np
from src.instances import trap_instance, generate_instance
from src.solvers import exact_solver, fixed_curriculum, greedy_basic, greedy_bundle
from src.qubo import build_qubo, cost_diagonal, to_ising, ising_diagonal, plan_info
from src.qaoa import run_qaoa


def test_trap_values():
    inst = trap_instance()
    assert exact_solver(inst)["benefit"] == 16
    assert greedy_basic(inst)["benefit"] == 11
    assert fixed_curriculum(inst)["benefit"] == 11
    assert greedy_bundle(inst)["benefit"] == 15


def test_all_solvers_feasible_and_bounded():
    for s in range(10):
        inst = generate_instance(8, 0.3, 0.5, "partial", s)
        opt = exact_solver(inst)["benefit"]
        for f in (fixed_curriculum, greedy_basic, greedy_bundle):
            r = f(inst)
            assert r["feasible"] and r["benefit"] <= opt + 1e-9


def test_ising_matches_qubo_and_ground_state():
    inst = trap_instance()
    Q, c, K, N = build_qubo(inst, 18.0, 12.0)
    E, _ = cost_diagonal(Q, c, N)
    h, J, off = to_ising(Q, c)
    assert np.allclose(E, ising_diagonal(h, J, off, N))
    ben, feas = plan_info(inst, N)
    k = int(np.argmin(E))
    assert feas[k] and np.isclose(ben[k], 16)


def test_qaoa_runs():
    r = run_qaoa(trap_instance(), p=2, restarts=2, maxiter=60, seed=1)
    assert 0 <= r["feasible_rate"] <= 1 and r["qubits"] == 10


def test_uniform_baseline_fields():
    r = run_qaoa(trap_instance(), p=1, restarts=1, maxiter=20, seed=0)
    assert r["p_opt_uniform"] > 0 and r["lift_p_opt"] > 0
