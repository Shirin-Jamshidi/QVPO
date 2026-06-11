# qvpo_cartpole/critic.py
"""
Twin Q-networks for QVPO.

Paper §5: "the practical implementation of QVPO follows SAC in critic part,
which utilizes doubled Q networks and only use the minimum for policy and critic update."

Q(s, a) → scalar    input: cat(state, action)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Single Q-network  Q_ω(s, a) → R."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = list(self.net.children())[-1]
        nn.init.uniform_(last.weight, -3e-3, 3e-3)
        nn.init.zeros_(last.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1)."""
        return self.net(torch.cat([state, action], dim=-1))


class TwinQNetwork(nn.Module):
    """
    Twin critics Q1, Q2.
    Exposes:
      q_min(s, a)   → min(Q1, Q2)    shape (B,)
      both(s, a)    → Q1, Q2          each (B, 1)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q1_vals, Q2_vals), each (B, 1)."""
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns min(Q1, Q2) as (B,)."""
        q1, q2 = self.both(state, action)
        return torch.min(q1, q2).squeeze(-1)

    def critic_loss(
        self,
        state:      torch.Tensor,   # (B, state_dim)
        action:     torch.Tensor,   # (B, action_dim)
        target:     torch.Tensor,   # (B,)  TD target
    ) -> torch.Tensor:
        """MSE loss on both heads."""
        q1, q2 = self.both(state, action)
        return F.mse_loss(q1.squeeze(-1), target) + F.mse_loss(q2.squeeze(-1), target)
