# qvpo_cartpole/agent.py
"""
QVPO Agent — full implementation of Algorithm 1 (NeurIPS 2024, arXiv 2405.16173v3).

Four paper components, all implemented:

  1. Q-weighted VLO loss  L(θ)                       §4.1 / Eq.(5-6)
     Weights each noisy-diffusion MSE term by ω_eq(s,a).

  2. Q-weight transformation  ω_eq = qadv            §4.2 / Eq.(9)
     ω_eq(s,a) = max(A(s,a), 0)
     where A(s,a) = Q(s,a) - V(s),  V(s) ≈ mean Q over Nd policy samples.
     Best-advantage action selected per state for training.

  3. Diffusion entropy regularisation  L_ent(θ)      §4.3 / Eq.(10)
     Uniform-action samples pushed through same diffusion loss,
     weighted by ω_ent(s) = ω_ent · ω_eq(s, a_max).

  4. K-efficient behavior policy  π^K_θ              §4.4
     At interaction time, draw K_b action candidates; act with argmax_Q.
     For TD target, use K_t < K_b candidates (avoids overestimation).

Critic follows SAC (twin Q, min for both policy + critic updates).
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .diffusion import GaussianDiffusion, EpsilonNet
    from .critic import TwinQNetwork
    from .replay_buffer import ReplayBuffer
except ImportError:
    from diffusion import GaussianDiffusion, EpsilonNet
    from critic import TwinQNetwork
    from replay_buffer import ReplayBuffer


class QVPO:
    """
    Parameters
    ----------
    state_dim, action_dim   : environment dimensions
    action_scale            : maps tanh-clipped network output → real action range
                              (for CartPole continuous: 1.0, scaled externally)
    device                  : cuda device
    -- Diffusion --
    n_diffusion_steps (T)   : paper uses T=5 for MuJoCo; T=5 fine for CartPole
    beta_min, beta_max      : linear noise schedule endpoints
    -- Policy update --
    n_policy_samples (Nd)   : # diffusion samples drawn per state for training   (paper: 64)
    n_uniform_samples (Ne)  : # uniform samples per state for entropy term       (paper: 64)
    omega_ent               : entropy regularisation coefficient                  (paper: 1.0)
    -- Behavior policy --
    K_b                     : action candidates for behavior policy               (paper: 10)
    K_t                     : action candidates for target policy                 (paper: 2)
    -- Critic --
    gamma                   : discount factor
    tau                     : soft-update rate for target networks
    lr_critic, lr_actor     : learning rates
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        device:       torch.device,
        # diffusion
        n_diffusion_steps: int   = 5,
        beta_min:          float = 0.1,
        beta_max:          float = 0.5,
        hidden_dim:        int   = 256,
        time_emb_dim:      int   = 16,
        # policy update
        n_policy_samples:  int   = 64,
        n_uniform_samples: int   = 64,
        omega_ent:         float = 1.0,
        # behavior policy
        K_b:               int   = 10,
        K_t:               int   = 2,
        # critic / RL
        gamma:      float = 0.99,
        tau:        float = 0.005,
        lr_critic:  float = 3e-4,
        lr_actor:   float = 3e-4,
    ):
        self.device    = device
        self.action_dim = action_dim
        self.gamma     = gamma
        self.tau       = tau
        self.K_b       = K_b
        self.K_t       = K_t
        self.Nd        = n_policy_samples
        self.Ne        = n_uniform_samples
        self.omega_ent = omega_ent

        # ── Diffusion schedule ────────────────────────────────────────────────
        self.diffusion = GaussianDiffusion(
            n_steps=n_diffusion_steps,
            beta_min=beta_min,
            beta_max=beta_max,
        ).to(device)

        # ── Epsilon-prediction network ε_θ ───────────────────────────────────
        self.eps_net = EpsilonNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            n_steps=n_diffusion_steps,
        ).to(device)

        # ── Twin critics + target copies ─────────────────────────────────────
        self.critic        = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # ── Optimisers ───────────────────────────────────────────────────────
        self.opt_actor  = optim.Adam(self.eps_net.parameters(),  lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(),   lr=lr_critic)

        # ── Logging ──────────────────────────────────────────────────────────
        self.total_steps = 0

    # ─────────────────────────────────────────────────────────────────────────
    # K-efficient behavior policy  (§4.4)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Draw K_b action candidates from the diffusion policy and return the one
        with highest Q-value  (K-efficient behavior policy, §4.4).

        state : (state_dim,)  numpy array
        returns : (action_dim,)  numpy array
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)

        # Sample K_b actions from diffusion policy
        # p_sample returns (1*K_b, action_dim)
        candidates = self.diffusion.p_sample(
            self.eps_net, s, n_samples=self.K_b
        )                                                           # (K_b, action_dim)

        # State tiled to match candidates
        s_rep = s.expand(self.K_b, -1)                             # (K_b, state_dim)
        q_vals = self.critic.q_min(s_rep, candidates)              # (K_b,)

        best_idx = q_vals.argmax()
        return candidates[best_idx].cpu().numpy()                   # (action_dim,)

    # ─────────────────────────────────────────────────────────────────────────
    # Critic update  (SAC-style twin Q + soft targets)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_critic(self, batch: dict) -> float:
        s  = batch["states"]       # (B, state_dim)
        a  = batch["actions"]      # (B, action_dim)
        r  = batch["rewards"]      # (B, 1)
        s_ = batch["next_states"]  # (B, state_dim)
        d  = batch["dones"]        # (B, 1)

        with torch.no_grad():
            # K_t-efficient target policy (§4.4)
            a_next = self.diffusion.p_sample(
                self.eps_net, s_, n_samples=self.K_t
            )                                                   # (B*K_t, action_dim)
            s_next_rep = s_.repeat_interleave(self.K_t, dim=0) # (B*K_t, state_dim)
            q_next = self.critic_target.q_min(
                s_next_rep, a_next
            ).view(-1, self.K_t).mean(dim=1, keepdim=True)     # (B, 1)  averaged

            td_target = r + self.gamma * (1.0 - d) * q_next    # (B, 1)

        loss_c = self.critic.critic_loss(s, a, td_target.squeeze(-1))
        self.opt_critic.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()
        return loss_c.item()

    # ─────────────────────────────────────────────────────────────────────────
    # Policy update  (Q-weighted VLO loss + entropy term)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_actor(self, batch: dict) -> tuple[float, float]:
        s = batch["states"]   # (B, state_dim)
        B = s.shape[0]

        # ── Step 1: Sample Nd actions from current diffusion policy ──────────
        with torch.no_grad():
            actions_nd = self.diffusion.p_sample(
                self.eps_net, s, n_samples=self.Nd
            )                                           # (B*Nd, action_dim)
            s_rep = s.repeat_interleave(self.Nd, dim=0) # (B*Nd, state_dim)

            # Q-values for all samples
            q_vals = self.critic.q_min(
                s_rep, actions_nd
            ).view(B, self.Nd)                          # (B, Nd)

            # V(s) ≈ mean Q over Nd samples  (§4.2 remark)
            v_s = q_vals.mean(dim=1, keepdim=True)      # (B, 1)

            # Advantage  A(s,a) = Q(s,a) - V(s)
            adv = q_vals - v_s                          # (B, Nd)

            # ── qadv weight transformation  ω_eq = max(A, 0)  (Eq. 9) ──────
            weights_nd = adv.clamp(min=0.0)             # (B, Nd)

            # Select best action per state (highest advantage, §4.2)
            best_idx = adv.argmax(dim=1)                # (B,)
            row_idx  = torch.arange(B, device=self.device)

            # Best action and its weight
            a_sel   = actions_nd.view(B, self.Nd, -1)[row_idx, best_idx]  # (B, action_dim)
            w_sel   = weights_nd[row_idx, best_idx]                         # (B,)

            # ── Entropy coefficient  ω_ent(s) = ω_ent · ω_eq(s, a_max) ─────
            omega_ent_s = self.omega_ent * w_sel        # (B,)

        # ── Step 2: Q-weighted VLO loss on selected actions  (Eq. 6) ────────
        loss_q = self.diffusion.q_weighted_vlo_loss(
            self.eps_net, a_sel, s, w_sel
        )

        # ── Step 3: Entropy regularisation loss  (Eq. 10) ────────────────────
        loss_e = self.diffusion.entropy_loss(
            self.eps_net, s, omega_ent_s, n_uniform=self.Ne
        )

        loss_total = loss_q + loss_e

        self.opt_actor.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(self.eps_net.parameters(), 1.0)
        self.opt_actor.step()

        return loss_q.item(), loss_e.item()

    # ─────────────────────────────────────────────────────────────────────────
    # Soft target update
    # ─────────────────────────────────────────────────────────────────────────

    def _soft_update(self):
        for p, p_tgt in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            p_tgt.data.lerp_(p.data, self.tau)

    # ─────────────────────────────────────────────────────────────────────────
    # Main training step  (Algorithm 1, lines 4-11)
    # ─────────────────────────────────────────────────────────────────────────

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """
        One gradient step over a sampled mini-batch.
        Returns a dict of scalar metrics for logging.
        """
        batch = replay_buffer.sample(batch_size)
        self.total_steps += 1

        loss_c          = self._update_critic(batch)
        loss_q, loss_e  = self._update_actor(batch)
        self._soft_update()

        return {
            "loss_critic":  loss_c,
            "loss_q_vlo":   loss_q,
            "loss_entropy": loss_e,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "eps_net":       self.eps_net.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.eps_net.load_state_dict(ckpt["eps_net"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
