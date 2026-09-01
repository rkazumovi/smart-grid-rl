"""
Benchmark PPO vs SAC vs MARL (MADDPG+GNN) on GridEnv, using identical evaluation
seeds for a fair side-by-side comparison against random and heuristic baselines.

Usage:
    python src/training/benchmark.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment.grid_env import GridEnv
from agents.marl_agent import MADDPGTrainer, split_global_action
from training.trainer import evaluate_policy_cost, random_policy, heuristic_policy

N_EVAL_EPISODES = 20
SEED_OFFSET = 5000


def load_sb3_policy(model_cls, model_path, vecnorm_path):
    if not (os.path.exists(model_path) and os.path.exists(vecnorm_path)):
        return None
    model = model_cls.load(model_path)
    dummy_env = DummyVecEnv([lambda: Monitor(GridEnv())])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False

    def policy(obs):
        norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        return action[0]
    return policy


def load_marl_policy(path):
    if not os.path.exists(path):
        return None
    trainer = MADDPGTrainer()
    trainer.load(path)

    def policy(obs):
        act_a, act_b = trainer.act(obs, noise_std=0.0)
        return split_global_action(act_a, act_b)
    return policy


def main():
    results = {}

    print("=== Evaluating baselines ===")
    results["Random"], _ = evaluate_policy_cost(random_policy, N_EVAL_EPISODES, SEED_OFFSET)
    results["Heuristic"], _ = evaluate_policy_cost(heuristic_policy, N_EVAL_EPISODES, SEED_OFFSET)

    ppo_policy = load_sb3_policy(PPO, "outputs/ppo_gridenv_final.zip", "outputs/vecnormalize_ppo.pkl")
    if ppo_policy:
        print("=== Evaluating PPO ===")
        results["PPO"], _ = evaluate_policy_cost(ppo_policy, N_EVAL_EPISODES, SEED_OFFSET)
    else:
        print("PPO model not found -- skipping (run train_full.py first)")

    sac_policy = load_sb3_policy(SAC, "outputs/sac_gridenv_final.zip", "outputs/vecnormalize_sac.pkl")
    if sac_policy:
        print("=== Evaluating SAC ===")
        results["SAC"], _ = evaluate_policy_cost(sac_policy, N_EVAL_EPISODES, SEED_OFFSET)
    else:
        print("SAC model not found -- skipping (run train_full_sac.py first)")

    # Prefer the best-during-training checkpoint over the final-episode weights: MADDPG can
    # (and on one of our runs did) get worse late in training, so "best" is the more honest
    # number to report. Fall back to the final checkpoint for older training runs that
    # predate this checkpointing.
    marl_path = "outputs/marl_actors_best.pt" if os.path.exists("outputs/marl_actors_best.pt") \
        else "outputs/marl_actors.pt"
    marl_policy = load_marl_policy(marl_path)
    if marl_policy:
        print(f"=== Evaluating MARL (MADDPG+GNN) -- using {marl_path} ===")
        results["MARL"], _ = evaluate_policy_cost(marl_policy, N_EVAL_EPISODES, SEED_OFFSET)
    else:
        print("MARL model not found -- skipping (run train_marl.py first)")

    baseline = results["Heuristic"]
    print(f"\n{'Algorithm':<12} {'Mean cost/episode':>20} {'% vs heuristic':>16}")
    print("-" * 50)
    for name, cost in results.items():
        pct = (baseline - cost) / baseline * 100
        marker = "" if name in ("Random", "Heuristic") else (
            "  <- target met" if
            (name == "PPO" and pct > 20) or (name == "SAC" and pct > 25) or (name == "MARL" and pct > 30)
            else "  <- below target"
        )
        print(f"{name:<12} ${cost:>18,.2f} {pct:>+15.1f}%{marker}")

    names = list(results.keys())
    costs = [results[n] for n in names]
    colors = ["#999999", "#666666", "#4C72B0", "#55A868", "#C44E52"][:len(names)]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, costs, color=colors)
    plt.ylabel("Mean episode cost ($)")
    plt.title("PPO vs SAC vs MARL: dispatch cost comparison (IEEE 14-bus, 20 eval episodes)")
    for bar, cost in zip(bars, costs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${cost:,.0f}",
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/benchmark_comparison.png", dpi=120)
    print("\nSaved chart to outputs/benchmark_comparison.png")


if __name__ == "__main__":
    main()