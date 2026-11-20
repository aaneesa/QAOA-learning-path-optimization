"""Statevector QAOA (numpy) + Qiskit circuit for resource counts."""
import numpy as np
from scipy.optimize import minimize
from .qubo import build_qubo, cost_diagonal, to_ising, plan_info, default_penalties


def qaoa_state(E, gammas, betas, N):
    psi = np.full(2 ** N, 2 ** (-N / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E)
        psi = psi.reshape([2] * N)
        c, s = np.cos(b), -1j * np.sin(b)
        U = np.array([[c, s], [s, c]])
        for ax in range(N):
            psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [ax])), 0, ax)
        psi = psi.reshape(-1)
    return psi


def run_qaoa(inst, p, P1=None, P2=None, restarts=5, maxiter=200, shots=2000, seed=0):
    if P1 is None or P2 is None:
        P1, P2 = default_penalties(inst)
    Q, const, K, N = build_qubo(inst, P1, P2)
    E, _ = cost_diagonal(Q, const, N)
    En = (E - E.min()) / (E.max() - E.min())            # normalised so gamma ~ O(1)
    ben, feas = plan_info(inst, N)
    b_opt = ben[feas].max()
    rng = np.random.default_rng(seed)

    def f(theta):
        psi = qaoa_state(En, theta[:p], theta[p:], N)
        return float(np.real(np.vdot(psi, En * psi)))

    best = None
    for _ in range(restarts):
        res = minimize(f, rng.uniform(0, np.pi, 2 * p), method="COBYLA", options=dict(maxiter=maxiter))
        if best is None or res.fun < best.fun:
            best = res
    psi = qaoa_state(En, best.x[:p], best.x[p:], N)
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    idx = rng.choice(len(probs), size=shots, p=probs)
    ok = feas[idx]
    best_b = float(ben[idx][ok].max()) if ok.any() else 0.0
    k0 = int(np.argmin(E))
    opt_mask = feas & np.isclose(ben, b_opt)
    # uniform-random baseline with the same shots (what QAOA must beat to mean anything)
    idx_u = rng.choice(len(probs), size=shots)
    ok_u = feas[idx_u]
    best_u = float(ben[idx_u][ok_u].max()) if ok_u.any() else 0.0
    return dict(p=p, P1=P1, P2=P2, qubits=N, benefit=best_b, B_opt=float(b_opt),
                gap=(b_opt - best_b) / b_opt if b_opt > 0 else float("nan"),
                feasible_rate=float(ok.mean()),
                p_opt=float(probs[opt_mask].sum()),
                p_opt_uniform=float(opt_mask.mean()),
                feasible_rate_uniform=float(feas.mean()),
                gap_uniform=(b_opt - best_u) / b_opt if b_opt > 0 else float("nan"),
                lift_p_opt=float(probs[opt_mask].sum() / opt_mask.mean()),
                exp_energy=float(best.fun),
                qubo_min_is_opt=bool(feas[k0] and np.isclose(ben[k0], b_opt)),
                nfev=int(best.nfev))


def circuit_resources(inst, p, P1=None, P2=None):
    """Transpiled depth / two-qubit gates (all-to-all connectivity, basis cx/rz/sx/x)."""
    from qiskit import QuantumCircuit, transpile
    if P1 is None or P2 is None:
        P1, P2 = default_penalties(inst)
    Q, const, K, N = build_qubo(inst, P1, P2)
    h, J, _ = to_ising(Q, const)
    qc = QuantumCircuit(N)
    qc.h(range(N))
    for _ in range(p):
        for (i, j), v in J.items():
            qc.rzz(0.5 * v, i, j)
        for i in range(N):
            qc.rz(0.5 * h[i], i)
        for i in range(N):
            qc.rx(0.3, i)
    tq = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    return dict(qubits=N, depth=tq.depth(), two_qubit_gates=int(tq.count_ops().get("cx", 0)))
