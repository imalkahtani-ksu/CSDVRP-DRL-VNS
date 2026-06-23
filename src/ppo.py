"""
ppo.py — Proximal Policy Optimisation (PPO) agent for DRL-ALNS operator selection.

State  : 17-dimensional vector encoding solution quality, iteration progress,
         operator history, SA temperature, split/violation metrics.
Actions: 6 operator pairs {D1,D2,D3} × {R1,R2}
         0=(D1,R1), 1=(D1,R2), 2=(D2,R1), 3=(D2,R2), 4=(D3,R1), 5=(D3,R2)

Network: Actor-Critic with two shared-head MLPs (128→64).
Training: Clipped surrogate objective with generalised advantage estimation (GAE).
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Training uses GPU; inference runs on CPU (per-step latency beats GPU launch overhead
# for this small 17→128→64→6 model).
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFER_DEVICE  = torch.device("cpu")

STATE_DIM  = 17
N_ACTIONS  = 6    # 3 destroy × 2 repair

# Operator index → (destroy_op, repair_op)
ACTION_MAP = {
    0: (0, 0),  # D1, R1
    1: (0, 1),  # D1, R2
    2: (1, 0),  # D2, R1
    3: (1, 1),  # D2, R2
    4: (2, 0),  # D3, R1
    5: (2, 1),  # D3, R2
}

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 64),        nn.ReLU(),
        )
        self.actor  = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, x: torch.Tensor):
        feat   = self.shared(x)
        logits = self.actor(feat)
        value  = self.critic(feat)
        return logits, value.squeeze(-1)

    def get_action(self, state: np.ndarray
                   ) -> Tuple[int, float, float]:
        """Inference — tensor moved to whatever device the model weights are on."""
        with torch.no_grad():
            dev    = next(self.parameters()).device   # follows model (CPU or CUDA)
            s      = torch.FloatTensor(state).unsqueeze(0).to(dev)
            logits, value = self.forward(s)
            dist   = Categorical(logits=logits)
            action = dist.sample()
            lp     = dist.log_prob(action)
        return int(action.item()), float(lp.item()), float(value.item())

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(states)
        dist    = Categorical(logits=logits)
        lp      = dist.log_prob(actions)
        entropy = dist.entropy()
        return lp, values, entropy

class PPOAgent:
    def __init__(
        self,
        lr:           float = 3e-4,
        clip_eps:     float = 0.2,
        vf_coef:      float = 0.5,
        ent_coef:     float = 0.01,
        gae_lambda:   float = 0.95,
        gamma:        float = 0.99,
        n_epochs:     int   = 4,
        batch_size:   int   = 64,
        n_actions:    int   = N_ACTIONS,
        state_dim:    int   = STATE_DIM,
    ):
        self.n_actions  = n_actions
        self.net        = ActorCritic(state_dim=state_dim,
                                      n_actions=n_actions).to(DEVICE)
        self.optimizer  = optim.Adam(self.net.parameters(), lr=lr)
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.gae_lambda = gae_lambda
        self.gamma      = gamma
        self.n_epochs   = n_epochs
        self.batch_size = batch_size

        # Rollout buffer
        self._states:   List[np.ndarray] = []
        self._actions:  List[int]        = []
        self._rewards:  List[float]      = []
        self._values:   List[float]      = []
        self._log_probs: List[float]     = []
        self._dones:    List[bool]       = []

    # inference
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        return self.net.get_action(state)

    # buffer management
    def store(self, state, action, reward, value, log_prob, done):
        self._states.append(state.copy())
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._log_probs.append(log_prob)
        self._dones.append(done)

    def clear_buffer(self):
        self._states.clear(); self._actions.clear(); self._rewards.clear()
        self._values.clear(); self._log_probs.clear(); self._dones.clear()

    # GAE advantage computation
    def _compute_gae(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        rewards  = np.array(self._rewards, dtype=np.float32)
        values   = np.array(self._values + [last_value], dtype=np.float32)
        dones    = np.array(self._dones,   dtype=np.float32)

        T        = len(rewards)
        adv      = np.zeros(T, dtype=np.float32)
        gae      = 0.0

        for t in reversed(range(T)):
            mask    = 1.0 - dones[t]
            delta   = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae     = delta + self.gamma * self.gae_lambda * mask * gae
            adv[t]  = gae

        returns = adv + values[:T]
        return adv, returns

    # PPO update
    def update(self, last_value: float = 0.0):
        if len(self._states) < 2:
            return {}

        adv, returns = self._compute_gae(last_value)
        # Normalise advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states   = torch.FloatTensor(np.array(self._states)).to(DEVICE)
        actions  = torch.LongTensor(self._actions).to(DEVICE)
        old_lp   = torch.FloatTensor(self._log_probs).to(DEVICE)
        adv_t    = torch.FloatTensor(adv).to(DEVICE)
        ret_t    = torch.FloatTensor(returns).to(DEVICE)

        N = len(self._states)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                batch = idx[start: start + self.batch_size]
                lp, values, entropy = self.net.evaluate(states[batch], actions[batch])

                ratio      = torch.exp(lp - old_lp[batch])
                surr1      = ratio * adv_t[batch]
                surr2      = torch.clamp(ratio, 1 - self.clip_eps,
                                         1 + self.clip_eps) * adv_t[batch]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (values - ret_t[batch]).pow(2).mean()
                ent_loss    = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += (-ent_loss.item())

        self.clear_buffer()
        return metrics

    # persistence
    def save(self, path: str):
        torch.save({
            "model":     self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str, for_inference: bool = False):
        """Load model weights.
        for_inference=True: move net to CPU so select_action has no GPU overhead.
        for_inference=False (default): model stays on DEVICE for continued training.
        """
        target = torch.device("cpu") if for_inference else DEVICE
        ckpt = torch.load(path, map_location=target)
        self.net.load_state_dict(ckpt["model"])
        if not for_inference:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.net.to(target)
        self.net.eval()
