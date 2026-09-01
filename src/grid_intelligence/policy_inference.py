"""
Runs one of the trained RL policies (PPO, SAC, or MARL/MADDPG) on GridEnv and decodes
its raw action vector into a plain-language description, for the Optimization Advisor
agent's "proposed_action" field.

Scope note, stated plainly rather than left implicit: GridEnv (src/environment/grid_env.py)
is a self-contained IEEE 14-bus test-system simulator -- synthetic wind/solar/battery/
demand models, its own reward function -- and is NOT connected to real Elia grid
telemetry the way src/forecasting/ is. Component 3's load/wind/solar numbers describe the
real Belgian grid; the RL policies here were trained and benchmarked entirely on a
physically realistic but synthetic 14-bus stand-in, at a completely different scale
(generator capacities in the tens of MW here, vs. thousands of MW for real Belgian load).
So "what does the policy recommend" below means "what does the policy actually recommend
in a representative scenario sampled from its own training/eval environment" -- a
genuine, real inference call against the real trained checkpoint, not a projection of
that action onto the real grid state reported elsewhere in the same pipeline run.
Bridging those two would need a grid model calibrated to the real Belgian network, which
is a substantially larger undertaking than this project built.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from environment.grid_env import GridEnv


def decode_action(env: GridEnv, action: np.ndarray) -> dict:
    """Mirrors GridEnv.step()'s own action-rescaling formulas exactly, so the
    description matches what the environment will actually do with this action --
    copied from grid_env.py's step() rather than re-derived, to avoid a second,
    possibly-drifting copy of the same math."""
    action = np.clip(action, -1.0, 1.0)
    n_gen = len(env.GEN_BUSES)
    gen_p_action = action[:n_gen]
    battery_action = action[2 * n_gen]
    price_action = action[2 * n_gen + 1]

    gen_dispatch_mw = {}
    for i, bus in enumerate(env.GEN_BUSES):
        frac = (gen_p_action[i] + 1) / 2  # [-1, 1] -> [0, 1]
        gen_dispatch_mw[bus] = float(frac * env.GEN_CAPACITY_MW[bus])
    battery_power_mw = float(battery_action * env.BATTERY_MAX_POWER_MW)  # + discharge, - charge
    price_usd_per_mwh = float(50.0 * (1 + price_action))

    return {
        "gen_dispatch_mw": gen_dispatch_mw,
        "total_gen_mw": sum(gen_dispatch_mw.values()),
        "battery_power_mw": battery_power_mw,
        "price_signal_usd_per_mwh": price_usd_per_mwh,
    }


def describe_action(decoded: dict) -> str:
    battery = decoded["battery_power_mw"]
    if battery > 0.05:
        battery_desc = f"discharge the battery at {battery:.2f} MW"
    elif battery < -0.05:
        battery_desc = f"charge the battery at {abs(battery):.2f} MW"
    else:
        battery_desc = "leave the battery roughly idle"

    gen_desc = ", ".join(f"bus {bus}: {mw:.1f} MW" for bus, mw in decoded["gen_dispatch_mw"].items())
    return (
        f"{battery_desc}; dispatch generators at ({gen_desc}), totaling "
        f"{decoded['total_gen_mw']:.1f} MW; set the demand-response price signal to "
        f"${decoded['price_signal_usd_per_mwh']:.1f}/MWh. (This is the trained {{policy}} "
        f"policy's actual recommendation on a representative scenario from its own IEEE "
        f"14-bus training environment -- see policy_inference.py's module docstring for "
        f"why this is not a projection onto the real Belgian grid state reported above.)"
    )


def get_policy_action(policy_name: str, seed: int = 0):
    """Loads the requested trained policy the same way benchmark.py does, resets GridEnv
    with `seed`, takes one deterministic step, and returns (decoded_dict, description,
    reward, info). Raises FileNotFoundError with a clear message if the checkpoint for
    `policy_name` isn't present under outputs/ -- callers (run_pipeline.py) catch this
    and fall back to a placeholder rather than crashing the whole pipeline."""
    from training.benchmark import load_sb3_policy, load_marl_policy
    from stable_baselines3 import PPO, SAC

    if policy_name == "PPO":
        policy = load_sb3_policy(PPO, "outputs/ppo_gridenv_final.zip", "outputs/vecnormalize_ppo.pkl")
    elif policy_name == "SAC":
        policy = load_sb3_policy(SAC, "outputs/sac_gridenv_final.zip", "outputs/vecnormalize_sac.pkl")
    elif policy_name == "MARL":
        marl_path = "outputs/marl_actors_best.pt" if os.path.exists("outputs/marl_actors_best.pt") \
            else "outputs/marl_actors.pt"
        policy = load_marl_policy(marl_path)
    else:
        raise ValueError(f"Unknown policy_name '{policy_name}' -- use 'PPO', 'SAC', or 'MARL'.")

    if policy is None:
        raise FileNotFoundError(
            f"No trained checkpoint found for '{policy_name}' under outputs/ -- "
            f"run the corresponding training script first."
        )

    env = GridEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    action = policy(obs)
    decoded = decode_action(env, action)
    description = describe_action(decoded).replace("{policy}", policy_name)
    _, reward, _, _, info = env.step(action)
    return decoded, description, reward, info


if __name__ == "__main__":
    policy_name = sys.argv[1] if len(sys.argv) > 1 else "MARL"
    decoded, description, reward, info = get_policy_action(policy_name, seed=0)
    print(f"Policy: {policy_name}")
    print(f"Decoded action: {decoded}")
    print(f"\nDescription:\n{description}")
    print(f"\nStep reward: {reward:.3f}")
    print(f"Info: {info}")