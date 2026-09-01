"""
Battery storage model: nonlinear SoC-dependent efficiency, integrated with RK4.

dE/dt = eta(SoC) * charge_power   - discharge_power / eta(SoC)
eta(SoC) = eta_max - k*(SoC - 0.5)^2      (worst near empty/full, best at mid-charge)
"""

import numpy as np


class BatteryModel:
    def __init__(self, capacity_mwh=10.0, max_power_mw=2.5,
                 eta_max=0.97, k=0.12, soc_min=0.05, soc_max=0.95, soc_init=0.5):
        self.capacity = capacity_mwh
        self.max_power = max_power_mw
        self.eta_max, self.k = eta_max, k
        self.soc_min, self.soc_max = soc_min, soc_max
        self.E = soc_init * capacity_mwh

    def reset(self, soc=0.5):
        self.E = soc * self.capacity
        return self.E

    def efficiency(self, E):
        soc = E / self.capacity
        return self.eta_max - self.k * (soc - 0.5) ** 2

    def _dEdt(self, E, u):
        """u > 0: charging at power u (MW). u < 0: discharging at power |u| (MW)."""
        eta = self.efficiency(E)
        charge = max(u, 0.0)
        discharge = max(-u, 0.0)
        return eta * charge - discharge / eta

    def step_rk4(self, u, dt_hours):
        u = np.clip(u, -self.max_power, self.max_power)
        E = self.E
        k1 = self._dEdt(E, u)
        k2 = self._dEdt(E + dt_hours / 2 * k1, u)
        k3 = self._dEdt(E + dt_hours / 2 * k2, u)
        k4 = self._dEdt(E + dt_hours * k3, u)
        E_new = E + dt_hours / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        E_new_clipped = np.clip(E_new, self.soc_min * self.capacity, self.soc_max * self.capacity)
        self.E = E_new_clipped
        return self.E / self.capacity  # return new SoC

    def step_euler(self, u, dt_hours):
        u = np.clip(u, -self.max_power, self.max_power)
        E_new = self.E + dt_hours * self._dEdt(self.E, u)
        self.E = np.clip(E_new, self.soc_min * self.capacity, self.soc_max * self.capacity)
        return self.E / self.capacity


def fine_reference(capacity, eta_max, k, E0, u, dt_hours, n_substeps=100000):
    """Ground truth: Euler with a huge number of substeps."""
    E = E0
    h = dt_hours / n_substeps
    for _ in range(n_substeps):
        soc = E / capacity
        eta = eta_max - k * (soc - 0.5) ** 2
        charge, discharge = max(u, 0.0), max(-u, 0.0)
        dEdt = eta * charge - discharge / eta
        E += h * dEdt
    return E


if __name__ == "__main__":
    batt_rk4 = BatteryModel()
    batt_euler = BatteryModel()
    E0 = batt_rk4.E

    dt = 1.0  # 1-hour coarse timestep -- deliberately coarse to expose integration error
    u = 2.0   # charging at 2 MW

    rk4_soc = batt_rk4.step_rk4(u, dt)
    euler_soc = batt_euler.step_euler(u, dt)
    ref_E = fine_reference(batt_rk4.capacity, batt_rk4.eta_max, batt_rk4.k, E0, u, dt)
    ref_soc = ref_E / batt_rk4.capacity

    print(f"Starting SoC: {E0/batt_rk4.capacity:.6f}")
    print(f"After 1 coarse step charging at {u} MW for {dt}h:")
    print(f"  RK4 SoC:            {rk4_soc:.6f}")
    print(f"  Euler SoC:          {euler_soc:.6f}")
    print(f"  Fine reference SoC: {ref_soc:.6f}  (100,000 substeps, ground truth)")
    print(f"\n  RK4 error:   {abs(rk4_soc - ref_soc):.3e}")
    print(f"  Euler error: {abs(euler_soc - ref_soc):.3e}")
    print(f"  RK4 is {abs(euler_soc - ref_soc) / max(abs(rk4_soc - ref_soc), 1e-15):.1f}x more accurate than Euler at the same step size")

    # Sanity: discharge test + bounds clipping
    batt = BatteryModel(soc_init=0.06)
    for _ in range(5):
        soc = batt.step_rk4(-2.5, 1.0)  # discharge hard
    print(f"\nDischarge-to-floor test: SoC clipped at soc_min={batt.soc_min} -> final SoC = {soc:.4f}")
    assert soc >= batt.soc_min - 1e-9