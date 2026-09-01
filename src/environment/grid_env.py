"""
GridEnv: a custom Gymnasium environment wrapping the from-scratch Newton-Raphson
power flow solver, stochastic renewable models, battery dynamics, and price-elastic
demand response into a single RL control problem on the IEEE 14-bus network.

Physics-informed reward shaping: Newton-Raphson is capped at a fixed iteration
budget, and the FINAL power-flow mismatch magnitude (not just a converged/failed
flag) is used directly as a continuous penalty term -- the agent is pushed toward
Kirchhoff-consistent dispatch decisions, not just "did the solver technically converge."
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandapower as pp
import pandapower.networks as pn

from physics.power_equations import newton_raphson_power_flow, calculate_power_injections
from environment.renewable_model import WindModel, SolarModel
from environment.battery_model import BatteryModel
from environment.demand_model import DemandResponseModel, diurnal_baseline_load


class GridEnv(gym.Env):
    metadata = {"render_modes": []}

    # Bus role assignments on the IEEE 14-bus network (0-indexed)
    GEN_BUSES = [1, 2, 5, 7]      # controllable PV buses
    WIND_BUS = 3                  # PQ bus hosting distributed wind
    SOLAR_BUS = 4                 # PQ bus hosting distributed solar
    BATTERY_BUS = 8               # PQ bus hosting the battery

    GEN_CAPACITY_MW = {1: 80.0, 2: 60.0, 5: 40.0, 7: 40.0}
    GEN_COST_A = {1: 0.02, 2: 0.025, 5: 0.03, 7: 0.015}    # $/MW^2/h (quadratic term)
    GEN_COST_B = {1: 20.0, 2: 22.0, 5: 25.0, 7: 18.0}      # $/MW/h (linear term)
    GEN_EMISSION = {1: 0.6, 2: 0.7, 5: 0.5, 7: 0.9}        # tons CO2 / MWh (fossil mix)
    SLACK_COST_A, SLACK_COST_B, SLACK_EMISSION = 0.018, 30.0, 0.65

    WIND_CAPACITY_MW = 30.0
    SOLAR_CAPACITY_MW = 20.0
    BATTERY_CAPACITY_MWH = 10.0
    BATTERY_MAX_POWER_MW = 2.5

    V_MIN, V_MAX = 0.95, 1.05
    LAMBDA_VOLTAGE = 500.0
    LAMBDA_PHYSICS = 200.0
    CARBON_PRICE = 40.0  # $/ton CO2

    def __init__(self, dt_hours=1.0, max_steps=24, seed=None):
        super().__init__()
        self.dt_hours = dt_hours
        self.max_steps = max_steps

        net = pn.case14()
        pp.runpp(net)
        self.n_bus = len(net.bus)
        self.baseMVA = net.sn_mva
        Ybus = net._ppc["internal"]["Ybus"].toarray()
        self.G, self.B = Ybus.real, Ybus.imag

        self.bus_types = ["pq"] * self.n_bus
        self.base_load_p = np.zeros(self.n_bus)  # MW, from the original case (used as peak load)
        self.base_load_q = np.zeros(self.n_bus)  # MVAr
        for _, row in net.load.iterrows():
            b = int(row.bus)
            self.base_load_p[b] += row.p_mw
            self.base_load_q[b] += row.q_mvar
        for _, row in net.gen.iterrows():
            self.bus_types[int(row.bus)] = "pv"
        for _, row in net.ext_grid.iterrows():
            self.slack_bus = int(row.bus)
            self.bus_types[self.slack_bus] = "slack"
            self.slack_v = row.vm_pu

        self._rng = np.random.default_rng(seed)
        self.wind = WindModel(rng=self._rng)
        self.solar = SolarModel(rng=self._rng)
        self.battery = BatteryModel(capacity_mwh=self.BATTERY_CAPACITY_MWH,
                                     max_power_mw=self.BATTERY_MAX_POWER_MW)
        self.demand = DemandResponseModel()

        n_gen = len(self.GEN_BUSES)
        # action: [gen_P (n_gen), gen_V (n_gen), battery_power (1), price_multiplier (1)], all in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * n_gen + 2,), dtype=np.float32)

        # obs: |V| (n_bus), angle (n_bus), wind power, solar power, battery SoC, sin/cos hour
        obs_dim = 2 * self.n_bus + 2 + 1 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.wind.rng = self._rng
            self.solar.rng = self._rng

        self.step_count = 0
        self.hour = 0.0
        self.wind.reset()
        self.battery.reset(soc=0.5)
        self._last_V = np.ones(self.n_bus)
        self._last_delta = np.zeros(self.n_bus)

        obs = self._build_observation(power_flow_ok=True)
        info = {}
        return obs, info

    def _build_observation(self, power_flow_ok):
        wind_power = self.wind.power_output()
        solar_power = self.solar.step(self.hour % 24)
        soc = self.battery.E / self.battery.capacity
        hour_sin = np.sin(2 * np.pi * (self.hour % 24) / 24)
        hour_cos = np.cos(2 * np.pi * (self.hour % 24) / 24)
        obs = np.concatenate([
            self._last_V, self._last_delta,
            [wind_power, solar_power, soc, hour_sin, hour_cos],
        ]).astype(np.float32)
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        n_gen = len(self.GEN_BUSES)
        gen_p_action = action[:n_gen]
        gen_v_action = action[n_gen:2 * n_gen]
        battery_action = action[2 * n_gen]
        price_action = action[2 * n_gen + 1]

        # --- rescale normalized actions to physical units ---
        gen_P_mw = {}
        for i, bus in enumerate(self.GEN_BUSES):
            frac = (gen_p_action[i] + 1) / 2  # [-1,1] -> [0,1]
            gen_P_mw[bus] = frac * self.GEN_CAPACITY_MW[bus]
        gen_V_pu = {}
        for i, bus in enumerate(self.GEN_BUSES):
            gen_V_pu[bus] = self.V_MIN + (gen_v_action[i] + 1) / 2 * (self.V_MAX - self.V_MIN)
        battery_power_mw = battery_action * self.BATTERY_MAX_POWER_MW
        price = 50.0 * (1 + price_action)  # price_action in [-1,1] -> price in [0, 100] $/MWh

        # --- advance stochastic processes ---
        self.wind.step()
        wind_power_mw = self.wind.power_output() * self.WIND_CAPACITY_MW
        solar_power_mw = self.solar.step(self.hour % 24) * self.SOLAR_CAPACITY_MW
        battery_soc_new = self.battery.step_rk4(battery_power_mw, self.dt_hours)

        # --- build bus injections for this timestep ---
        P_spec = np.zeros(self.n_bus)
        Q_spec = np.zeros(self.n_bus)
        load_factor = diurnal_baseline_load(self.hour % 24)
        for b in range(self.n_bus):
            if self.base_load_p[b] > 0 or self.base_load_q[b] > 0:
                baseline_p = self.base_load_p[b] * load_factor / 0.8  # normalize so factor~0.8 matches original case value
                actual_p = self.demand.response(baseline_p, price)
                scale = actual_p / self.base_load_p[b] if self.base_load_p[b] > 0 else 1.0
                P_spec[b] -= actual_p
                Q_spec[b] -= self.base_load_q[b] * load_factor / 0.8 * scale

        for bus, p_mw in gen_P_mw.items():
            P_spec[bus] += p_mw

        P_spec[self.WIND_BUS] += wind_power_mw
        P_spec[self.SOLAR_BUS] += solar_power_mw
        P_spec[self.BATTERY_BUS] -= battery_power_mw  # charging draws from grid (negative injection)

        P_spec /= self.baseMVA
        Q_spec /= self.baseMVA

        V_init = np.ones(self.n_bus)
        delta_init = np.zeros(self.n_bus)
        V_init[self.slack_bus] = self.slack_v
        for bus in self.GEN_BUSES:
            V_init[bus] = gen_V_pu[bus]

        # --- physics-informed power flow solve (capped iterations) ---
        V, delta, n_iter, converged, history = newton_raphson_power_flow(
            self.G, self.B, P_spec, Q_spec, self.bus_types, V_init, delta_init,
            tol=1e-6, max_iter=15, verbose=False,
        )
        physics_residual = history[-1] if len(history) > 0 else 0.0
        self._last_V, self._last_delta = V, delta

        # --- reward components ---
        P_final, _ = calculate_power_injections(V, delta, self.G, self.B)
        slack_P_mw = P_final[self.slack_bus] * self.baseMVA

        gen_cost = 0.0
        carbon_cost = 0.0
        for bus, p_mw in gen_P_mw.items():
            gen_cost += (self.GEN_COST_A[bus] * p_mw ** 2 + self.GEN_COST_B[bus] * p_mw) * self.dt_hours
            carbon_cost += self.GEN_EMISSION[bus] * p_mw * self.dt_hours * self.CARBON_PRICE
        if slack_P_mw > 0:  # importing from the main grid
            gen_cost += (self.SLACK_COST_A * slack_P_mw ** 2 + self.SLACK_COST_B * slack_P_mw) * self.dt_hours
            carbon_cost += self.SLACK_EMISSION * slack_P_mw * self.dt_hours * self.CARBON_PRICE

        voltage_violation = np.sum(np.maximum(0, V - self.V_MAX) ** 2 + np.maximum(0, self.V_MIN - V) ** 2)

        reward = -(gen_cost + carbon_cost
                   + self.LAMBDA_VOLTAGE * voltage_violation
                   + self.LAMBDA_PHYSICS * physics_residual)

        self.step_count += 1
        self.hour += self.dt_hours
        terminated = not converged
        truncated = self.step_count >= self.max_steps

        obs = self._build_observation(power_flow_ok=converged)
        info = {
            "converged": converged, "n_iter": n_iter, "physics_residual": physics_residual,
            "gen_cost": gen_cost, "carbon_cost": carbon_cost, "voltage_violation": voltage_violation,
            "slack_P_mw": slack_P_mw, "battery_soc": battery_soc_new,
        }
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = GridEnv(seed=0)
    obs, info = env.reset(seed=0)
    print(f"Observation space: {env.observation_space.shape}, Action space: {env.action_space.shape}")
    print(f"Initial obs shape: {obs.shape}\n")

    rewards = []
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        ep_reward = 0.0
        for t in range(env.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if t == 0:
                print(f"Episode {ep} step 0: reward={reward:.3f}, converged={info['converged']}, "
                      f"n_iter={info['n_iter']}, physics_residual={info['physics_residual']:.2e}, "
                      f"gen_cost={info['gen_cost']:.2f}, carbon_cost={info['carbon_cost']:.2f}, "
                      f"voltage_violation={info['voltage_violation']:.4f}")
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        print(f"Episode {ep} finished after {t+1} steps, total reward = {ep_reward:.2f}, "
              f"final converged={info['converged']}")

    print(f"\nOver {len(rewards)} random-action episodes: mean reward = {np.mean(rewards):.2f}")
    print("Environment ran without crashing across random actions.")