from __future__ import annotations

import numpy as np
import torch
from torch import nn


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Core(nn.Module):
    """Наблюдение -> латент. memory=True: на вход ещё предыдущее действие, дальше GRU.

    Без памяти агент видит только текущий цвет (бейзлайн для POMDP).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128, memory: bool = True):
        super().__init__()
        self.memory = memory
        self.hidden = hidden
        in_dim = obs_dim + (n_actions if memory else 0)
        self.encoder = nn.Sequential(layer_init(nn.Linear(in_dim, hidden)), nn.Tanh())
        if memory:
            self.gru = nn.GRUCell(hidden, hidden)
        else:
            self.mlp = nn.Sequential(layer_init(nn.Linear(hidden, hidden)), nn.Tanh())

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    def step(self, obs, prev_action, starts, h):
        """Один шаг для батча сред. starts=1 — начало эпизода, память обнуляется."""
        if not self.memory:
            return self.mlp(self.encoder(obs)), h
        h = h * (1.0 - starts).unsqueeze(-1)
        h = self.gru(self.encoder(torch.cat([obs, prev_action], dim=-1)), h)
        return h, h

    def unroll(self, obs, prev_action, starts, h):
        """Прогон по последовательности (T, B, ...) — для обучения на сохранённом роллауте."""
        if not self.memory:
            return self.mlp(self.encoder(obs)), h
        outs = []
        for t in range(obs.shape[0]):
            out, h = self.step(obs[t], prev_action[t], starts[t], h)
            outs.append(out)
        return torch.stack(outs), h


class ActorCritic(nn.Module):
    """Общий Core и две головы: актор (логиты действий) и критик (V(s))."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128, memory: bool = True):
        super().__init__()
        self.n_actions = n_actions
        self.core = Core(obs_dim, n_actions, hidden, memory)
        self.actor = layer_init(nn.Linear(hidden, n_actions), gain=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), gain=1.0)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return self.core.initial_state(batch, device)

    def step(self, obs, prev_action, starts, h):
        z, h = self.core.step(obs, prev_action, starts, h)
        return self.actor(z), self.critic(z).squeeze(-1), h

    def unroll(self, obs, prev_action, starts, h):
        z, h = self.core.unroll(obs, prev_action, starts, h)
        return self.actor(z), self.critic(z).squeeze(-1), h
