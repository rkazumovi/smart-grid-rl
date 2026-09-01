"""
Training loop for the MADDPG multi-agent setup on GridEnv.

Usage:
    python src/training/train_marl.py --episodes 500 --seed 0
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from environment.grid_env import GridEnv
from agents.marl_agent import MADDPGTrainer, split_global_action
from training.trainer import evaluate_policy_cost, random_policy, heuristic_policy


def evaluate_marl_cost(trainer, n_episodes=10, seed_offset=1000):
    env = GridEnv()
    costs = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_cost, done = 0.0, False
        while not done:
            act_a, act_b = trainer.act(obs, noise_std=0.0)  # deterministic at eval time
            full_action = split_global_action(act_a, act_b)
            obs, reward, terminated, truncated, info = env.step(full_action)
            ep_cost += info["gen_cost"] + info["carbon_cost"]
            done = terminated or truncated
        costs.append(ep_cost)
    return np.mean(costs), np.std(costs)


def main(n_episodes=500, batch_size=128, warmup_episodes=20, noise_std=0.3, seed=0, eval_every=50):
    env = GridEnv(seed=seed)
    trainer = MADDPGTrainer(seed=seed)
    print(f"(seed={seed} -- rerun with the same --seed to reproduce this exact run)")
    os.makedirs("outputs", exist_ok=True)

    print("=== Baseline evaluation ===")
    random_cost, _ = evaluate_policy_cost(random_policy, n_episodes=10)
    heuristic_cost, _ = evaluate_policy_cost(heuristic_policy, n_episodes=10)
    print(f"Random cost: ${random_cost:,.2f}  Heuristic cost: ${heuristic_cost:,.2f}")

    # MADDPG is not guaranteed to improve monotonically -- track the best checkpoint by
    # actual deterministic eval cost, not just whatever the final episode happens to produce.
    best_cost = float("inf")

    print(f"\n=== Training MADDPG for {n_episodes} episodes ===")
    ep_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward, done = 0.0, False
        current_noise = noise_std * max(0.0, 1.0 - ep / (n_episodes * 0.8))  # anneal exploration
        while not done:
            act_a, act_b = trainer.act(obs, noise_std=current_noise)
            full_action = split_global_action(act_a, act_b)
            next_obs, reward, terminated, truncated, info = env.step(full_action)
            done = terminated or truncated
            trainer.buffer.push(obs, act_a, act_b, reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward
            if ep >= warmup_episodes:
                trainer.update(batch_size=batch_size)
        ep_rewards.append(ep_reward)
        if (ep + 1) % eval_every == 0:
            recent = np.mean(ep_rewards[-eval_every:])
            eval_cost, _ = evaluate_marl_cost(trainer, n_episodes=5, seed_offset=9000 + ep)
            print(f"Episode {ep + 1}/{n_episodes}: mean reward (last {eval_every}) = {recent:,.1f}, "
                  f"noise = {current_noise:.3f}, eval cost = ${eval_cost:,.2f}")
            if eval_cost < best_cost:
                best_cost = eval_cost
                trainer.save("outputs/marl_actors_best.pt")
                print(f"  -> new best (${best_cost:,.2f}), checkpoint saved to outputs/marl_actors_best.pt")

    print("\n=== MADDPG evaluation (deterministic policy, final episode's weights) ===")
    marl_cost, marl_std = evaluate_marl_cost(trainer, n_episodes=10)
    print(f"MADDPG cost: ${marl_cost:,.2f} (+/- {marl_std:,.2f})")
    print(f"vs random:    {(random_cost - marl_cost) / random_cost * 100:+.1f}%")
    print(f"vs heuristic: {(heuristic_cost - marl_cost) / heuristic_cost * 100:+.1f}%")

    trainer.save("outputs/marl_actors.pt")
    if marl_cost < best_cost:
        best_cost = marl_cost
        trainer.save("outputs/marl_actors_best.pt")
    print("\n" + "=" * 60)
    print("SAVED: outputs/marl_actors.pt  (final-episode weights)")
    print(f"SAVED: outputs/marl_actors_best.pt  (best eval cost overall: ${best_cost:,.2f})")
    print("=" * 60)

    return trainer, ep_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n_episodes=args.episodes, seed=args.seed)