"""
Full SAC training run on GridEnv: checkpointing, TensorBoard logging, and
periodic real-$-cost evaluation (via CostEvalCallback).

Usage:
    python src/training/train_full_sac.py --timesteps 150000
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from environment.grid_env import GridEnv
from agents.sac_agent import build_sac_agent
from training.trainer import make_vec_env, evaluate_policy_cost, random_policy, heuristic_policy
from training.callbacks import CostEvalCallback


def main(total_timesteps=150000, eval_freq=10000, checkpoint_freq=50000):
    os.makedirs("outputs/checkpoints/sac", exist_ok=True)
    os.makedirs("outputs/tensorboard/sac", exist_ok=True)

    print("=== Baseline evaluation (before training) ===")
    random_cost, _ = evaluate_policy_cost(random_policy, n_episodes=20)
    heuristic_cost, _ = evaluate_policy_cost(heuristic_policy, n_episodes=20)
    print(f"Random policy:    mean cost/episode = ${random_cost:,.2f}")
    print(f"Heuristic policy: mean cost/episode = ${heuristic_cost:,.2f}")

    vec_env = make_vec_env(seed=0)
    model = build_sac_agent(vec_env, verbose=1, tensorboard_log="outputs/tensorboard/sac")

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq, save_path="outputs/checkpoints/sac", name_prefix="sac_gridenv"
    )
    cost_eval_cb = CostEvalCallback(eval_env_fn=lambda: GridEnv(), eval_freq=eval_freq, n_eval_episodes=5, verbose=1)

    print(f"\n=== Training SAC for {total_timesteps:,} timesteps ===")
    print("Note: SAC updates every environment step, so it runs slower (wall-clock) per")
    print("timestep than PPO but is typically more sample-efficient per interaction.")
    print("Monitor progress live in another terminal with:")
    print("  tensorboard --logdir outputs/tensorboard/sac")
    model.learn(total_timesteps=total_timesteps, callback=CallbackList([checkpoint_cb, cost_eval_cb]))

    model.save("outputs/sac_gridenv_final")
    vec_env.save("outputs/vecnormalize_sac.pkl")
    print("\nSaved final model to outputs/sac_gridenv_final.zip and outputs/vecnormalize_sac.pkl")

    def sac_policy(obs):
        norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        return action[0]

    print("\n=== Final SAC evaluation ===")
    sac_cost, sac_std = evaluate_policy_cost(sac_policy, n_episodes=20)
    print(f"SAC policy: mean cost/episode = ${sac_cost:,.2f} (+/- {sac_std:,.2f})")
    print(f"vs random:    {(random_cost - sac_cost) / random_cost * 100:+.1f}%")
    print(f"vs heuristic: {(heuristic_cost - sac_cost) / heuristic_cost * 100:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--checkpoint-freq", type=int, default=50000)
    args = parser.parse_args()
    main(args.timesteps, args.eval_freq, args.checkpoint_freq)