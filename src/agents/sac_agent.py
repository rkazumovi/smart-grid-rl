"""SAC agent configuration for GridEnv."""
from stable_baselines3 import SAC


def build_sac_agent(env, **kwargs):
    defaults = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=500,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",   # automatic entropy/temperature tuning (dual optimization)
        verbose=0,
    )
    defaults.update(kwargs)
    return SAC(env=env, **defaults)