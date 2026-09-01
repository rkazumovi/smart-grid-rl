"""
Full PPO training run on GridEnv: checkpointing, TensorBoard logging, and
periodic real-$-cost evaluation (via CostEvalCallback).

Usage:
    python src/training/train_full.py --timesteps 300000
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from environment.grid_env import GridEnv
from agents.ppo_agent import build_ppo_agent
from training.trainer import make_vec_env, evaluate_policy_cost, random_policy, heuristic_policy
from training.callbacks import CostEvalCallback


def main(total_timesteps=300000, eval_freq=10000, checkpoint_freq=50000):
    os.makedirs("outputs/checkpoints/ppo", exist_ok=True)
    os.makedirs("outputs/tensorboard/ppo", exist_ok=True)

    print("=== Baseline evaluation (before training) ===")
    random_cost, _ = evaluate_policy_cost(random_policy, n_episodes=20)
    heuristic_cost, _ = evaluate_policy_cost(heuristic_policy, n_episodes=20)
    print(f"Random policy:    mean cost/episode = ${random_cost:,.2f}")
    print(f"Heuristic policy: mean cost/episode = ${heuristic_cost:,.2f}")

    vec_env = make_vec_env(seed=0)
    model = build_ppo_agent(vec_env, verbose=1, tensorboard_log="outputs/tensorboard/ppo")

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq, save_path="outputs/checkpoints/ppo", name_prefix="ppo_gridenv"
    )
    cost_eval_cb = CostEvalCallback(eval_env_fn=lambda: GridEnv(), eval_freq=eval_freq, n_eval_episodes=5, verbose=1)

    print(f"\n=== Training PPO for {total_timesteps:,} timesteps ===")
    print("Monitor progress live in another terminal with:")
    print("  tensorboard --logdir outputs/tensorboard/ppo")
    model.learn(total_timesteps=total_timesteps, callback=CallbackList([checkpoint_cb, cost_eval_cb]))

    model.save("outputs/ppo_gridenv_final")
    vec_env.save("outputs/vecnormalize_ppo.pkl")
    print("\nSaved final model to outputs/ppo_gridenv_final.zip and outputs/vecnormalize_ppo.pkl")

    def ppo_policy(obs):
        norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        return action[0]

    print("\n=== Final PPO evaluation ===")
    ppo_cost, ppo_std = evaluate_policy_cost(ppo_policy, n_episodes=20)
    print(f"PPO policy: mean cost/episode = ${ppo_cost:,.2f} (+/- {ppo_std:,.2f})")
    print(f"vs random:    {(random_cost - ppo_cost) / random_cost * 100:+.1f}%")
    print(f"vs heuristic: {(heuristic_cost - ppo_cost) / heuristic_cost * 100:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--checkpoint-freq", type=int, default=50000)
    args = parser.parse_args()
    main(args.timesteps, args.eval_freq, args.checkpoint_freq)