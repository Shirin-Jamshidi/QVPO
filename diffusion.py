# qvpo_cartpole/diffusion.py
"""
QVPO Diffusion components — faithful to NeurIPS 2024 paper (arXiv 2405.16173v3).

Implements:
  - Linear variance schedule (β_1 … β_T), α_bar pre-computation
  - Epsilon-prediction noise network  ε_θ(√ᾱ_t · a + √(1-ᾱ_t)·ε, s, t)
  - Forward process  q(a_t | a_0)
  - Reverse DDPM sampler  p_θ(a_{t-1} | a_t, s)  used for policy rollout
  - Q-weighted VLO loss  L(θ) = E[ω_eq(s,a) · ||ε - ε_θ(…)||²]   (Eq. 6)
  - Entropy regularisation   L_ent(θ) = E[ω_ent(s) · ||ε - ε_θ(…)||²]  (Eq. 10)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding  (standard DDPM)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t : (B,) float/int  →  (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) *
            torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        args = t.float()[:, None] * freqs[None]          # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1) # (B, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Noise network  ε_θ(a_t, s, t)
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonNet(nn.Module):
    """
    MLP that predicts the noise ε added at diffusion step t.

    Input  : concat(a_t, s, t_emb)   — noisy action + state + time embedding
    Output : ε̂  ∈  R^{action_dim}   — predicted noise, same shape as action

    Architecture follows the official QVPO implementation style:
      3 hidden layers, Mish activations, LayerNorm for stability.
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        hidden_dim:   int = 256,
        time_emb_dim: int = 16,
        n_steps:      int = 5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps    = n_steps

        self.time_emb = SinusoidalPosEmb(time_emb_dim)

        in_dim = action_dim + state_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # small output scale
        last = list(self.net.children())[-1]
        nn.init.uniform_(last.weight, -1e-3, 1e-3)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        a_t:   torch.Tensor,   # (B, action_dim)  noisy action
        state: torch.Tensor,   # (B, state_dim)
        t:     torch.Tensor,   # (B,)  integer ∈ [1, T]
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)                              # (B, time_emb_dim)
        x = torch.cat([a_t, state, t_emb], dim=-1)           # (B, in_dim)
        return self.net(x)                                    # (B, action_dim)


# ─────────────────────────────────────────────────────────────────────────────
# DDPM schedule + forward / reverse processes
# ─────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    """
    DDPM schedule and sampling for continuous action space.

    All schedule tensors are pre-computed and moved to the correct device
    on the first use via `.to(device)`.

    Paper uses:  T = 5,  linear β schedule  (Appendix B)
    """

    def __init__(self, n_steps: int = 5, beta_min: float = 0.1, beta_max: float = 0.5):
        self.T = n_steps

        betas      = torch.linspace(beta_min, beta_max, n_steps)    # (T,)
        alphas     = 1.0 - betas                                     # (T,)
        alpha_bar  = torch.cumprod(alphas, dim=0)                    # (T,)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]]) # (T,) ᾱ_{t-1}

        # ---- store everything; moved to device lazily ----
        self.betas          = betas
        self.alphas         = alphas
        self.alpha_bar      = alpha_bar          # ᾱ_t
        self.alpha_bar_prev = alpha_bar_prev     # ᾱ_{t-1}
        self.sqrt_ab        = alpha_bar.sqrt()
        self.sqrt_1mab      = (1 - alpha_bar).sqrt()
        # Posterior variance  σ²_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t
        self.posterior_var  = (1 - alpha_bar_prev) / (1 - alpha_bar) * betas
        self._device        = None

    def to(self, device: torch.device) -> "GaussianDiffusion":
        for attr in [
            "betas", "alphas", "alpha_bar", "alpha_bar_prev",
            "sqrt_ab", "sqrt_1mab", "posterior_var"
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        self._device = device
        return self

    # ------------------------------------------------------------------
    # Forward process  q(a_t | a_0)
    # ------------------------------------------------------------------
    def q_sample(
        self,
        a0:    torch.Tensor,                     # (B, action_dim)
        t:     torch.Tensor,                     # (B,) int  in [1, T]
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (a_t, ε) where a_t = √ᾱ_t · a_0 + √(1-ᾱ_t) · ε."""
        if noise is None:
            noise = torch.randn_like(a0)
        s_ab  = self.sqrt_ab[t - 1].view(-1, 1)      # (B,1)  index 0-based
        s_1mab = self.sqrt_1mab[t - 1].view(-1, 1)
        return s_ab * a0 + s_1mab * noise, noise

    # ------------------------------------------------------------------
    # Reverse step  p_θ(a_{t-1} | a_t, s)   — used at inference time
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample_step(
        self,
        eps_net: EpsilonNet,
        a_t:     torch.Tensor,   # (B, action_dim)
        state:   torch.Tensor,   # (B, state_dim)
        t_val:   int,            # scalar in [1, T]
    ) -> torch.Tensor:
        """One reverse DDPM step; returns a_{t-1}."""
        B      = a_t.shape[0]
        t      = torch.full((B,), t_val, dtype=torch.long, device=a_t.device)
        eps_hat = eps_net(a_t, state, t)               # (B, action_dim)

        # Predicted a_0 from  a_t  and  ε̂
        s_ab   = self.sqrt_ab[t_val - 1]
        s_1mab = self.sqrt_1mab[t_val - 1]
        a0_hat = (a_t - s_1mab * eps_hat) / s_ab
        a0_hat = a0_hat.clamp(-1.0, 1.0)

        # Posterior mean  μ_θ(a_t, t)
        ab      = self.alpha_bar[t_val - 1]
        ab_prev = self.alpha_bar_prev[t_val - 1]
        beta_t  = self.betas[t_val - 1]
        mu = (ab_prev.sqrt() * beta_t * a0_hat +
              (1 - ab_prev) * (1 - beta_t).sqrt() * a_t) / (1 - ab)

        if t_val == 1:
            return mu
        noise = torch.randn_like(a_t)
        sigma = self.posterior_var[t_val - 1].sqrt()
        return mu + sigma * noise

    # ------------------------------------------------------------------
    # Full reverse chain  p_θ(a_0 | s)   — action sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(
        self,
        eps_net: EpsilonNet,
        state:   torch.Tensor,          # (B, state_dim)
        n_samples: int = 1,             # how many independent rollouts per state
    ) -> torch.Tensor:
        """
        Sample action(s) from the diffusion policy.

        If n_samples > 1, tiles `state` and returns (B*n_samples, action_dim).
        """
        B, device = state.shape[0], state.device

        if n_samples > 1:
            state = state.repeat_interleave(n_samples, dim=0)  # (B*N, state_dim)

        # Start from pure Gaussian noise
        a_t = torch.randn(state.shape[0], eps_net.action_dim, device=device)

        for t_val in reversed(range(1, self.T + 1)):
            a_t = self.p_sample_step(eps_net, a_t, state, t_val)

        return a_t.clamp(-1.0, 1.0)   # (B*n_samples, action_dim)

    # ------------------------------------------------------------------
    # Q-weighted VLO loss  L(θ)  —  Eq. (6) of the paper
    # ------------------------------------------------------------------
    def q_weighted_vlo_loss(
        self,
        eps_net:  EpsilonNet,
        a_sel:    torch.Tensor,   # (B, action_dim)  selected best action per state
        state:    torch.Tensor,   # (B, state_dim)
        weights:  torch.Tensor,   # (B,)  ω_eq(s, a)  — non-negative
    ) -> torch.Tensor:
        """
        L(θ) = E_{s, a~π_k, ε, t} [ ω_eq(s,a) · ||ε - ε_θ(√ᾱ_t·a + √(1-ᾱ_t)·ε, s, t)||² ]

        `a_sel`  is the single best-advantage action selected per state (§4.2).
        `weights` is ω_eq ≡ max(A(s,a), 0)  computed externally.
        """
        B = a_sel.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=a_sel.device)  # t ∈ [1,T]
        a_t, noise = self.q_sample(a_sel, t)
        eps_hat    = eps_net(a_t, state, t)
        per_sample = ((noise - eps_hat) ** 2).mean(dim=-1)  # (B,)
        return (weights * per_sample).mean()

    # ------------------------------------------------------------------
    # Entropy regularisation loss  L_ent(θ)  —  Eq. (10) of the paper
    # ------------------------------------------------------------------
    def entropy_loss(
        self,
        eps_net:    EpsilonNet,
        state:      torch.Tensor,   # (B, state_dim)
        omega_ent_s: torch.Tensor,  # (B,)  ω_ent(s) coefficient per state
        n_uniform:  int,            # Ne — number of uniform-random action samples
    ) -> torch.Tensor:
        """
        L_ent(θ) = E_{s, a~U, ε, t} [ ω_ent(s) · ||ε - ε_θ(√ᾱ_t·a + √(1-ᾱ_t)·ε, s, t)||² ]

        Actions sampled from U(-1, 1)^{action_dim} approximate the maximum-entropy
        uniform distribution over the action space.
        """
        B, device = state.shape[0], state.device

        # Tile state for Ne uniform samples per batch element
        state_rep     = state.repeat_interleave(n_uniform, dim=0)       # (B*Ne, state_dim)
        omega_rep     = omega_ent_s.repeat_interleave(n_uniform, dim=0)  # (B*Ne,)

        # Uniform actions in [-1, 1]^{action_dim}
        a_unif = torch.rand(B * n_uniform, eps_net.action_dim, device=device) * 2 - 1

        t = torch.randint(1, self.T + 1, (B * n_uniform,), device=device)
        a_t, noise = self.q_sample(a_unif, t)
        eps_hat    = eps_net(a_t, state_rep, t)
        per_sample = ((noise - eps_hat) ** 2).mean(dim=-1)   # (B*Ne,)
        return (omega_rep * per_sample).mean()
