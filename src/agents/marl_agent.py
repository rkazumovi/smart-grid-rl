"""
Multi-Agent RL (MADDPG-style) with GNN-based inter-agent communication.

Two cooperative agents jointly control the IEEE 14-bus GridEnv:
  Agent A: generators at buses 1, 2   (4 action dims: P, V setpoints)
  Agent B: generators at buses 5, 7, plus battery and price  (6 action dims)

Centralized training, decentralized execution (CTDE):
  - Each actor sees only its own local observation, but exchanges one learned
    message with the other agent via a graph attention layer (GATConv) before
    committing to an action -- extends the GNN expertise from Project 7.
  - Each critic is centralized: it sees the full global observation and BOTH
    agents' actions during training only (discarded at deployment). This is
    the classic MADDPG fix for the non-stationarity every agent otherwise
    sees from the other agent's policy changing underneath it.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

AGENT_ACTION_SLICES = {"A": [0, 1, 4, 5], "B": [2, 3, 6, 7, 8, 9]}
ACTION_DIMS = {"A": 4, "B": 6}
OBS_DIM = 33
TOTAL_ACTION_DIM = 10


def split_global_action(action_a: np.ndarray, action_b: np.ndarray) -> np.ndarray:
    """Combine each agent's sub-action into the 10-dim action GridEnv expects."""
    full = np.zeros(TOTAL_ACTION_DIM, dtype=np.float32)
    full[AGENT_ACTION_SLICES["A"]] = action_a
    full[AGENT_ACTION_SLICES["B"]] = action_b
    return full


class GNNCommActor(nn.Module):
    """Local obs -> embedding -> GAT message passing with the other agent -> action."""

    def __init__(self, obs_dim, action_dim, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, embed_dim), nn.ReLU())
        self.gat = GATConv(embed_dim, embed_dim, heads=1)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, action_dim), nn.Tanh()
        )

    def encode(self, obs):
        return self.encoder(obs)

    def communicate_and_act(self, own_embed, other_embed):
        """own_embed, other_embed: (batch, embed_dim). Returns action: (batch, action_dim)."""
        batch = own_embed.shape[0]
        x = torch.stack([own_embed, other_embed], dim=1).reshape(batch * 2, -1)
        base_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=x.device)
        edge_index = torch.cat([base_edges + 2 * i for i in range(batch)], dim=1)
        out = self.gat(x, edge_index).reshape(batch, 2, -1)[:, 0, :]  # this agent's updated embedding
        return self.decoder(out)


class CentralizedCritic(nn.Module):
    """Q(global_obs, action_A, action_B) -- training-time only."""

    def __init__(self, obs_dim=OBS_DIM, total_action_dim=TOTAL_ACTION_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + total_action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs, action_a, action_b):
        return self.net(torch.cat([global_obs, action_a, action_b], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, act_a, act_b, reward, next_obs, done):
        item = (obs, act_a, act_b, reward, next_obs, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        obs, act_a, act_b, reward, next_obs, done = zip(*[self.buffer[i] for i in idx])
        return (np.array(obs), np.array(act_a), np.array(act_b),
                np.array(reward), np.array(next_obs), np.array(done))

    def __len__(self):
        return len(self.buffer)


class MADDPGTrainer:
    def __init__(self, obs_dim=OBS_DIM, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.01,
                 grad_clip_norm=1.0, device="cpu", seed=None):
        """
        actor_lr < critic_lr on purpose: in DDPG-family methods the critic must track a
        moving target (the actor) accurately before the actor's policy gradient through it
        means anything. A critic that learns faster than the actor changes keeps that
        gradient signal trustworthy; matching learning rates (as an earlier version of this
        trainer did) lets the actor chase a noisy, half-formed critic and can visibly
        destabilize training run-to-run (see notebooks/rl_theory.ipynb for the writeup).

        seed: fixes torch's and numpy's global RNGs so network initialization, exploration
        noise, and replay-buffer sampling are reproducible across runs with the same seed.
        MADDPG (like DDPG/TD3) is known to be sensitive to initial conditions, so leaving
        this unset means two runs of identical code and hyperparameters can converge to
        very different final policies -- which is exactly what happened between our first
        (+13.3% vs heuristic) and second (-26.4%) 500-episode runs.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.gamma, self.tau, self.device = gamma, tau, device
        self.grad_clip_norm = grad_clip_norm

        self.actor_a = GNNCommActor(obs_dim, ACTION_DIMS["A"]).to(device)
        self.actor_b = GNNCommActor(obs_dim, ACTION_DIMS["B"]).to(device)
        self.target_actor_a = GNNCommActor(obs_dim, ACTION_DIMS["A"]).to(device)
        self.target_actor_b = GNNCommActor(obs_dim, ACTION_DIMS["B"]).to(device)
        self.target_actor_a.load_state_dict(self.actor_a.state_dict())
        self.target_actor_b.load_state_dict(self.actor_b.state_dict())

        self.critic_a = CentralizedCritic().to(device)
        self.critic_b = CentralizedCritic().to(device)
        self.target_critic_a = CentralizedCritic().to(device)
        self.target_critic_b = CentralizedCritic().to(device)
        self.target_critic_a.load_state_dict(self.critic_a.state_dict())
        self.target_critic_b.load_state_dict(self.critic_b.state_dict())

        params_a = list(self.actor_a.parameters())
        params_b = list(self.actor_b.parameters())
        self.actor_a_opt = torch.optim.Adam(params_a, lr=actor_lr)
        self.actor_b_opt = torch.optim.Adam(params_b, lr=actor_lr)
        self.critic_a_opt = torch.optim.Adam(self.critic_a.parameters(), lr=critic_lr)
        self.critic_b_opt = torch.optim.Adam(self.critic_b.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer()

    def save(self, path):
        torch.save({
            "actor_a": self.actor_a.state_dict(),
            "actor_b": self.actor_b.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_a.load_state_dict(checkpoint["actor_a"])
        self.actor_b.load_state_dict(checkpoint["actor_b"])
        self.target_actor_a.load_state_dict(checkpoint["actor_a"])
        self.target_actor_b.load_state_dict(checkpoint["actor_b"])

    def act(self, obs, actor_a=None, actor_b=None, noise_std=0.0):
        actor_a = actor_a or self.actor_a
        actor_b = actor_b or self.actor_b
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embed_a = actor_a.encode(obs_t)
            embed_b = actor_b.encode(obs_t)
            action_a = actor_a.communicate_and_act(embed_a, embed_b).cpu().numpy()[0]
            action_b = actor_b.communicate_and_act(embed_b, embed_a).cpu().numpy()[0]
        if noise_std > 0:
            action_a = np.clip(action_a + np.random.normal(0, noise_std, size=action_a.shape), -1, 1)
            action_b = np.clip(action_b + np.random.normal(0, noise_std, size=action_b.shape), -1, 1)
        return action_a, action_b

    def _soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) + s_param.data * self.tau)

    def update(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return None
        obs, act_a, act_b, reward, next_obs, done = self.buffer.sample(batch_size)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_a = torch.as_tensor(act_a, dtype=torch.float32, device=self.device)
        act_b = torch.as_tensor(act_b, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_embed_a = self.target_actor_a.encode(next_obs)
            next_embed_b = self.target_actor_b.encode(next_obs)
            next_act_a = self.target_actor_a.communicate_and_act(next_embed_a, next_embed_b)
            next_act_b = self.target_actor_b.communicate_and_act(next_embed_b, next_embed_a)
            target_q_a = reward + self.gamma * (1 - done) * self.target_critic_a(next_obs, next_act_a, next_act_b)
            target_q_b = reward + self.gamma * (1 - done) * self.target_critic_b(next_obs, next_act_a, next_act_b)

        critic_a_loss = nn.functional.mse_loss(self.critic_a(obs, act_a, act_b), target_q_a)
        self.critic_a_opt.zero_grad(); critic_a_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_a.parameters(), self.grad_clip_norm)
        self.critic_a_opt.step()

        critic_b_loss = nn.functional.mse_loss(self.critic_b(obs, act_a, act_b), target_q_b)
        self.critic_b_opt.zero_grad(); critic_b_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_b.parameters(), self.grad_clip_norm)
        self.critic_b_opt.step()

        embed_a = self.actor_a.encode(obs)
        embed_b_detached = self.actor_b.encode(obs).detach()
        new_act_a = self.actor_a.communicate_and_act(embed_a, embed_b_detached)
        actor_a_loss = -self.critic_a(obs, new_act_a, act_b).mean()
        self.actor_a_opt.zero_grad(); actor_a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_a.parameters(), self.grad_clip_norm)
        self.actor_a_opt.step()

        embed_b = self.actor_b.encode(obs)
        embed_a_detached = self.actor_a.encode(obs).detach()
        new_act_b = self.actor_b.communicate_and_act(embed_b, embed_a_detached)
        actor_b_loss = -self.critic_b(obs, act_a, new_act_b).mean()
        self.actor_b_opt.zero_grad(); actor_b_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_b.parameters(), self.grad_clip_norm)
        self.actor_b_opt.step()

        self._soft_update(self.target_actor_a, self.actor_a)
        self._soft_update(self.target_actor_b, self.actor_b)
        self._soft_update(self.target_critic_a, self.critic_a)
        self._soft_update(self.target_critic_b, self.critic_b)

        return {"critic_a_loss": critic_a_loss.item(), "critic_b_loss": critic_b_loss.item(),
                "actor_a_loss": actor_a_loss.item(), "actor_b_loss": actor_b_loss.item()}