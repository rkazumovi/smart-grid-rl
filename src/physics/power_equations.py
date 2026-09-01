"""
AC Power Flow — Newton-Raphson solver, implemented from scratch.

Math reference (see notebooks/power_flow_math.ipynb for the full derivation):

    P_k = |V_k| * sum_j |V_j| * (G_kj*cos(d_k-d_j) + B_kj*sin(d_k-d_j))
    Q_k = |V_k| * sum_j |V_j| * (G_kj*sin(d_k-d_j) - B_kj*cos(d_k-d_j))

Bus types:
    'slack' : |V|, delta fixed (reference bus). P, Q solved for.
    'pv'    : P, |V| fixed (generator bus). delta, Q solved for.
    'pq'    : P, Q fixed (load bus). |V|, delta solved for.

Newton-Raphson iterates:
    [dP; dQ] = J [d_delta; d_V]
    [delta; V]_(n+1) = [delta; V]_n + J^-1 [dP; dQ]
until max(|dP|, |dQ|) < tol.
"""

import numpy as np


def calculate_power_injections(V: np.ndarray, delta: np.ndarray, G: np.ndarray, B: np.ndarray):
    """Compute P_k, Q_k at every bus from the current voltage state (V, delta)."""
    n = len(V)
    P = np.zeros(n)
    Q = np.zeros(n)
    for k in range(n):
        for j in range(n):
            angle = delta[k] - delta[j]
            P[k] += V[k] * V[j] * (G[k, j] * np.cos(angle) + B[k, j] * np.sin(angle))
            Q[k] += V[k] * V[j] * (G[k, j] * np.sin(angle) - B[k, j] * np.cos(angle))
    return P, Q


def compute_jacobian(V: np.ndarray, delta: np.ndarray, G: np.ndarray, B: np.ndarray,
                      P: np.ndarray, Q: np.ndarray, pv_pq: list, pq: list):
    """
    Build the Newton-Raphson Jacobian.

    Row/col ordering:
        rows: dP for buses in pv_pq (all non-slack), then dQ for buses in pq
        cols: d_delta for buses in pv_pq, then d_V for buses in pq
    """
    n = len(V)
    npv_pq = len(pv_pq)
    npq = len(pq)

    H = np.zeros((npv_pq, npv_pq))   # dP/d_delta
    N = np.zeros((npv_pq, npq))      # dP/d_V
    M = np.zeros((npq, npv_pq))      # dQ/d_delta
    L = np.zeros((npq, npq))         # dQ/d_V

    for a, k in enumerate(pv_pq):
        for b, m in enumerate(pv_pq):
            if k == m:
                H[a, b] = -Q[k] - B[k, k] * V[k] ** 2
            else:
                angle = delta[k] - delta[m]
                H[a, b] = V[k] * V[m] * (G[k, m] * np.sin(angle) - B[k, m] * np.cos(angle))

    for a, k in enumerate(pv_pq):
        for b, m in enumerate(pq):
            if k == m:
                N[a, b] = P[k] / V[k] + G[k, k] * V[k]
            else:
                angle = delta[k] - delta[m]
                N[a, b] = V[k] * (G[k, m] * np.cos(angle) + B[k, m] * np.sin(angle))

    for a, k in enumerate(pq):
        for b, m in enumerate(pv_pq):
            if k == m:
                M[a, b] = P[k] - G[k, k] * V[k] ** 2
            else:
                angle = delta[k] - delta[m]
                M[a, b] = -V[k] * V[m] * (G[k, m] * np.cos(angle) + B[k, m] * np.sin(angle))

    for a, k in enumerate(pq):
        for b, m in enumerate(pq):
            if k == m:
                L[a, b] = Q[k] / V[k] - B[k, k] * V[k]
            else:
                angle = delta[k] - delta[m]
                L[a, b] = V[k] * (G[k, m] * np.sin(angle) - B[k, m] * np.cos(angle))

    J = np.block([[H, N], [M, L]])
    return J


def newton_raphson_power_flow(G: np.ndarray, B: np.ndarray, P_spec: np.ndarray, Q_spec: np.ndarray,
                               bus_types: list, V_init: np.ndarray, delta_init: np.ndarray,
                               tol: float = 1e-6, max_iter: int = 20, verbose: bool = True):
    """
    Solve the AC power flow equations for voltage magnitudes and angles.

    bus_types[k] in {'slack', 'pv', 'pq'} for each bus k.
    Returns: V, delta, n_iterations, converged (bool), history (list of max mismatch per iter)
    """
    n = len(V_init)
    V = V_init.copy()
    delta = delta_init.copy()

    slack = [k for k in range(n) if bus_types[k] == 'slack']
    pv = [k for k in range(n) if bus_types[k] == 'pv']
    pq = [k for k in range(n) if bus_types[k] == 'pq']
    pv_pq = sorted(pv + pq)

    assert len(slack) == 1, "Exactly one slack bus is required."

    history = []
    for it in range(1, max_iter + 1):
        P, Q = calculate_power_injections(V, delta, G, B)

        dP = np.array([P_spec[k] - P[k] for k in pv_pq])
        dQ = np.array([Q_spec[k] - Q[k] for k in pq])
        mismatch = np.concatenate([dP, dQ]) if len(pq) > 0 else dP
        max_mismatch = np.max(np.abs(mismatch)) if len(mismatch) > 0 else 0.0
        history.append(max_mismatch)

        if verbose:
            print(f"  iter {it}: max mismatch = {max_mismatch:.3e}")

        if max_mismatch < tol:
            return V, delta, it, True, history

        J = compute_jacobian(V, delta, G, B, P, Q, pv_pq, pq)
        dx = np.linalg.solve(J, mismatch)

        d_delta = dx[:len(pv_pq)]
        d_V = dx[len(pv_pq):]

        for a, k in enumerate(pv_pq):
            delta[k] += d_delta[a]
        for a, k in enumerate(pq):
            V[k] += d_V[a]

    return V, delta, max_iter, False, history


if __name__ == "__main__":
    # Classic 3-bus textbook test system: 1 slack, 1 PV, 1 PQ bus.
    # Each of the 3 lines has series impedance z = 0.02 + j0.06 pu
    # -> series admittance y = 1/z = 5 - j15 pu (no line charging, for simplicity).
    G = np.array([
        [10.0, -5.0, -5.0],
        [-5.0, 10.0, -5.0],
        [-5.0, -5.0, 10.0],
    ])
    B = np.array([
        [-30.0, 15.0, 15.0],
        [15.0, -30.0, 15.0],
        [15.0, 15.0, -30.0],
    ])

    bus_types = ['slack', 'pv', 'pq']
    P_spec = np.array([0.0, 0.5, -0.6])     # bus 0 (slack) P is solved, not specified
    Q_spec = np.array([0.0, 0.0, -0.3])     # only PQ bus's Q is used

    V_init = np.array([1.05, 1.02, 1.00])
    delta_init = np.array([0.0, 0.0, 0.0])

    print("Solving 3-bus AC power flow with Newton-Raphson...\n")
    V, delta, n_iter, converged, history = newton_raphson_power_flow(
        G, B, P_spec, Q_spec, bus_types, V_init, delta_init
    )

    print(f"\nConverged: {converged} in {n_iter} iterations")
    print("\nFinal bus voltages:")
    for k in range(len(V)):
        print(f"  Bus {k} ({bus_types[k]:5s}): |V| = {V[k]:.5f} pu, angle = {np.degrees(delta[k]):.4f} deg")

    P_final, Q_final = calculate_power_injections(V, delta, G, B)
    print("\nFinal power injections (computed from converged V, delta):")
    for k in range(len(V)):
        print(f"  Bus {k}: P = {P_final[k]:+.5f} pu, Q = {Q_final[k]:+.5f} pu")

    print("\nSanity check — power balance (sum of P should be ~0, i.e. slack supplies the losses):")
    print(f"  sum(P) = {np.sum(P_final):.6f} pu  (should be small and POSITIVE = line losses)")