"""PPO agent configuration for GridEnv."""
from stable_baselines3 import PPO

def build_ppo_agent(vec_env, **kwargs):
    defaults = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=240,          # 10 episodes worth of steps (episode length = 24)
        batch_size=60,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,      # Generalized Advantage Estimation
        clip_range=0.2,       # PPO clipped surrogate objective
        ent_coef=0.0,
        verbose=0,
    )
    defaults.update(kwargs)
    return PPO(env=vec_env, **defaults)