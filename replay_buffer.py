# qvpo_cartpole/replay_buffer.py
"""
Standard uniform replay buffer for QVPO (online RL, no offline data).

All tensors live on CPU; batches are moved to device during sampling.
Sized to match the paper's off-policy setup (1M capacity for MuJoCo;
we use 300k which is more than enough for CartPole).
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        capacity:   int = 300_000,
        device:     torch.device = torch.device("cpu"),
    ):
        self.capacity   = capacity
        self.device     = device
        self.ptr        = 0
        self.size       = 0

        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

    def add(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       float,
    ):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states":      torch.FloatTensor(self.states[idx]).to(self.device),
            "actions":     torch.FloatTensor(self.actions[idx]).to(self.device),
            "rewards":     torch.FloatTensor(self.rewards[idx]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[idx]).to(self.device),
            "dones":       torch.FloatTensor(self.dones[idx]).to(self.device),
        }

    def __len__(self) -> int:
        return self.size
