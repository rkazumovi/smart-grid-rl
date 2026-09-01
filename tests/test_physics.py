"""
Validates the from-scratch Newton-Raphson AC power flow solver (src/physics/power_equations.py)
against pandapower's own trusted solver, on the real IEEE 14-bus benchmark network.

Run as a pytest test:   pytest tests/test_physics.py -v
Run standalone:         python tests/test_physics.py
"""

import sys
import os
import numpy as np
import pandapower as pp
import pandapower.networks as pn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from physics.power_equations import newton_raphson_power_flow


def build_ieee14_inputs():
    """Load the IEEE 14-bus case and build (G, B, P_spec, Q_spec, bus_types, V_init, delta_init)."""
    net = pn.case14()
    pp.runpp(net)  # solve once with pandapower to get Ybus and ground-truth results

    baseMVA = net.sn_mva
    n = len(net.bus)

    Ybus = net._ppc["internal"]["Ybus"].toarray()
    G, B = Ybus.real, Ybus.imag

    bus_types = ["pq"] * n
    V_init = np.ones(n)
    delta_init = np.zeros(n)
    P_spec = np.zeros(n)
    Q_spec = np.zeros(n)

    for _, row in net.load.iterrows():
        b = int(row.bus)
        P_spec[b] -= row.p_mw
        Q_spec[b] -= row.q_mvar

    for _, row in net.gen.iterrows():
        b = int(row.bus)
        P_spec[b] += row.p_mw
        bus_types[b] = "pv"
        V_init[b] = row.vm_pu

    for _, row in net.ext_grid.iterrows():
        b = int(row.bus)
        bus_types[b] = "slack"
        V_init[b] = row.vm_pu
        delta_init[b] = np.radians(row.va_degree)

    P_spec /= baseMVA
    Q_spec /= baseMVA

    return net, G, B, P_spec, Q_spec, bus_types, V_init, delta_init


def test_ieee14_matches_pandapower():
    """Our solver's converged (|V|, angle) must match pandapower's to tight tolerance."""
    net, G, B, P_spec, Q_spec, bus_types, V_init, delta_init = build_ieee14_inputs()

    V, delta, n_iter, converged, history = newton_raphson_power_flow(
        G, B, P_spec, Q_spec, bus_types, V_init, delta_init, tol=1e-8, max_iter=30, verbose=False
    )

    assert converged, "Newton-Raphson failed to converge on IEEE 14-bus"
    assert n_iter <= 10, f"Should converge quickly (quadratic convergence), took {n_iter} iterations"

    pp_v = net.res_bus.vm_pu.to_numpy()
    pp_a = net.res_bus.va_degree.to_numpy()
    our_a = np.degrees(delta)

    max_v_diff = np.max(np.abs(V - pp_v))
    max_a_diff = np.max(np.abs(our_a - pp_a))

    assert max_v_diff < 1e-4, f"Voltage magnitude mismatch too large: {max_v_diff}"
    assert max_a_diff < 1e-3, f"Voltage angle mismatch too large: {max_a_diff}"


if __name__ == "__main__":
    net, G, B, P_spec, Q_spec, bus_types, V_init, delta_init = build_ieee14_inputs()

    print("Bus types:", bus_types)
    print("\nSolving IEEE 14-bus with our from-scratch Newton-Raphson solver (flat start)...\n")
    V, delta, n_iter, converged, history = newton_raphson_power_flow(
        G, B, P_spec, Q_spec, bus_types, V_init, delta_init, tol=1e-8, max_iter=30
    )

    print(f"\nConverged: {converged} in {n_iter} iterations\n")
    print(f"{'Bus':>4} {'type':>6} {'Ours |V|':>10} {'PP |V|':>10} {'diff':>10}   "
          f"{'Ours ang':>10} {'PP ang':>10} {'diff':>10}")

    max_v_diff, max_a_diff = 0, 0
    for k in range(len(V)):
        our_v, pp_v = V[k], net.res_bus.vm_pu.iloc[k]
        our_a, pp_a = np.degrees(delta[k]), net.res_bus.va_degree.iloc[k]
        v_diff, a_diff = abs(our_v - pp_v), abs(our_a - pp_a)
        max_v_diff, max_a_diff = max(max_v_diff, v_diff), max(max_a_diff, a_diff)
        print(f"{k:>4} {bus_types[k]:>6} {our_v:>10.5f} {pp_v:>10.5f} {v_diff:>10.2e}   "
              f"{our_a:>10.4f} {pp_a:>10.4f} {a_diff:>10.2e}")

    print(f"\nMax |V| difference:    {max_v_diff:.2e} pu")
    print(f"Max angle difference:  {max_a_diff:.2e} deg")
    print("PASS" if max_v_diff < 1e-4 and max_a_diff < 1e-3 else "FAIL")