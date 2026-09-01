"""Custom training callbacks for GridEnv agents."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CostEvalCallback(BaseCallback):
    """
    Periodically evaluates the actual $ cost (gen_cost + carbon_cost from the
    environment's info dict) on fresh episodes -- NOT the shaped RL reward,
    which also bakes in voltage/physics penalty terms and reward normalization.
    Logs the result to TensorBoard as eval/mean_episode_cost_usd.
    """

    def __init__(self, eval_env_fn, eval_freq=10000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            env = self.eval_env_fn()
            costs = []
            for ep in range(self.n_eval_episodes):
                obs, _ = env.reset(seed=100000 + ep)
                ep_cost = 0.0
                done = False
                while not done:
                    norm_obs = obs.reshape(1, -1)
                    if hasattr(self.model.get_env(), "normalize_obs"):
                        norm_obs = self.model.get_env().normalize_obs(norm_obs)
                    action, _ = self.model.predict(norm_obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action[0])
                    ep_cost += info["gen_cost"] + info["carbon_cost"]
                    done = terminated or truncated
                costs.append(ep_cost)
            mean_cost = float(np.mean(costs))
            self.logger.record("eval/mean_episode_cost_usd", mean_cost)
            if self.verbose:
                print(f"[CostEvalCallback] step={self.num_timesteps}: mean episode cost = ${mean_cost:,.2f}")
        return True