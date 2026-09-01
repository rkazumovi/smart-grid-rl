"""
RL training loop for GridEnv: trains PPO, evaluates against random and
heuristic-dispatch baselines, reports % cost reduction.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environment.grid_env import GridEnv
from agents.ppo_agent import build_ppo_agent


def make_vec_env(seed=0):
    def _init():
        env = GridEnv(seed=seed)
        return Monitor(env)
    vec_env = DummyVecEnv([_init])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    return vec_env


def evaluate_policy_cost(policy_fn, n_episodes=10, seed_offset=1000):
    """
    Run n_episodes on a fresh (unwrapped) GridEnv using policy_fn(obs) -> action.
    Returns mean total (gen_cost + carbon_cost) per episode -- the actual $ cost,
    not the shaped RL reward (which also includes voltage/physics penalty terms).
    """
    env = GridEnv()
    total_costs = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_cost = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_cost += info["gen_cost"] + info["carbon_cost"]
            done = terminated or truncated
        total_costs.append(ep_cost)
    return np.mean(total_costs), np.std(total_costs)


def random_policy(obs):
    return np.random.uniform(-1, 1, size=10)


def heuristic_policy(obs):
    """Dispatch all generators at a moderate fixed fraction, battery idle, price at reference."""
    return np.array([0.0] * 4 + [0.0] * 4 + [0.0] + [0.0])  # mid-range dispatch (frac=0.5), V mid, no battery, ref price


if __name__ == "__main__":
    print("=== Baseline evaluation (before training) ===")
    random_cost, random_std = evaluate_policy_cost(random_policy, n_episodes=10)
    heuristic_cost, heuristic_std = evaluate_policy_cost(heuristic_policy, n_episodes=10)
    print(f"Random policy:    mean cost/episode = ${random_cost:,.2f} (+/- {random_std:,.2f})")
    print(f"Heuristic policy: mean cost/episode = ${heuristic_cost:,.2f} (+/- {heuristic_std:,.2f})")

    print("\n=== Training PPO (smoke test: 20,000 timesteps) ===")
    vec_env = make_vec_env(seed=0)
    model = build_ppo_agent(vec_env, verbose=1)
    model.learn(total_timesteps=20000, progress_bar=False)

    def ppo_policy(obs):
        norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        return action[0]

    print("\n=== PPO evaluation (after 20,000-step smoke test) ===")
    ppo_cost, ppo_std = evaluate_policy_cost(ppo_policy, n_episodes=10)
    print(f"PPO policy:       mean cost/episode = ${ppo_cost:,.2f} (+/- {ppo_std:,.2f})")

    improvement_vs_random = (random_cost - ppo_cost) / random_cost * 100
    improvement_vs_heuristic = (heuristic_cost - ppo_cost) / heuristic_cost * 100
    print(f"\nPPO vs random baseline:    {improvement_vs_random:+.1f}% cost change")
    print(f"PPO vs heuristic baseline: {improvement_vs_heuristic:+.1f}% cost change")
    print("\n(Note: this is a short smoke test (20k steps) to verify the training loop works end-to-end.")
    print(" Reaching the >20% cost-reduction target will need substantially longer training --")
    print(" hundreds of thousands of timesteps -- which we'll run as a proper background job next.)")