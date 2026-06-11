# QVPO — CartPole-v1 Baseline
### NeurIPS 2024 · arXiv 2405.16173v3

Faithful implementation of **Q-weighted Variational Policy Optimization (QVPO)**
applied to CartPole-v1 as a standalone online-RL baseline.

---

## Paper Components Implemented

All four contributions from the QVPO paper are implemented exactly as described.

### 1 · Q-weighted VLO Loss  `diffusion.py → q_weighted_vlo_loss`  (§4.1, Eq. 6)

The core loss. Standard DDPM denoising MSE weighted per-sample by `ω_eq(s, a)`:

```
L(θ) = E_{s, a~π_k, ε, t} [ ω_eq(s,a) · ||ε - ε_θ(√ᾱ_t·a + √(1-ᾱ_t)·ε, s, t)||² ]
```

Only the **single best-advantage action** per state is used for training (§4.2 selection rule).

---

### 2 · `qadv` Q-weight Transformation  `agent.py → _update_actor`  (§4.2, Eq. 9)

Converts Q-values to non-negative weights while handling negative Q:

```
ω_eq(s, a) = max(A(s, a), 0)
A(s, a)    = Q(s, a) − V(s)
V(s)       ≈ (1/Nd) Σ Q(s, aᵢ)    [mean over Nd diffusion samples]
```

The action with the highest advantage is selected for training — this is the
"best sample selection" that improves training quality without extra networks.

---

### 3 · Diffusion Entropy Regularisation  `diffusion.py → entropy_loss`  (§4.3, Eq. 10)

Pushes the policy toward higher entropy (better exploration) by training it to
also fit **uniformly sampled** actions:

```
L_ent(θ) = E_{s, a~U(-1,1), ε, t} [ ω_ent(s) · ||ε - ε_θ(…)||² ]
ω_ent(s)  = ω_ent · ω_eq(s, a_max)
```

This is the paper's key insight: since the log-likelihood of a diffusion policy
is intractable, entropy is approximated by minimising KL to the uniform distribution
via the VLO objective on uniform samples.

---

### 4 · K-efficient Behavior Policy  `agent.py → select_action`  (§4.4)

Reduces the large variance of diffusion policies during environment interaction:

```
π^{K_b}_θ(a|s)  =  argmax_{a ∈ {a¹,…,a^{K_b} ~ π_θ}} Q(s, a)
```

A smaller `K_t < K_b` is used for the TD target to avoid overestimation:

```
y_t = r_t + γ · (1/K_t) Σ min(Q1, Q2)(s_{t+1}, aⁱ_{t+1})
```

---

## File Structure

```
qvpo_cartpole/
├── diffusion.py       # GaussianDiffusion schedule + EpsilonNet + losses (Eq. 5-10)
├── critic.py          # TwinQNetwork  (SAC-style double Q)
├── replay_buffer.py   # Standard uniform replay buffer
├── agent.py           # QVPO agent — Algorithm 1
└── train.py           # Training loop + CartPole continuous wrapper
```

---

## Setup & Run

```bash
# On sm120 (GPU machine)
pip install torch gymnasium

# Run with paper-default hyperparameters
python train.py

# Reproduce with specific seed
python train.py --seed 0 --total_steps 300000

# Ablate entropy term (set omega_ent=0 to remove it)
python train.py --omega_ent 0.0

# Ablate K-efficient policy (K_b=1 → no selection)
python train.py --K_b 1 --K_t 1
```

---

## Key Hyperparameters

| Argument | Default | Paper value | Description |
|---|---|---|---|
| `--n_diffusion_steps` | 5 | 5 | Diffusion chain length T |
| `--n_policy_samples` | 64 | 64 | Nd — samples for qadv weights |
| `--n_uniform_samples` | 64 | 64 | Ne — samples for entropy term |
| `--omega_ent` | 1.0 | 1.0 | Entropy regularisation coefficient |
| `--K_b` | 10 | 10 | Behavior policy action candidates |
| `--K_t` | 2 | 2 | Target policy action candidates |
| `--batch_size` | 256 | 256 | Mini-batch size |
| `--gamma` | 0.99 | 0.99 | Discount factor |
| `--tau` | 0.005 | 0.005 | Soft target update rate |

---

## CartPole Action Space Adaptation

CartPole-v1 is natively discrete (`Discrete(2)`). QVPO requires a continuous
action space. We wrap the environment with a 1-D continuous action in `[-1, 1]`:

```
a < 0  →  discrete 0  (push cart left)
a ≥ 0  →  discrete 1  (push cart right)
```

State observations (4-D: cart pos, cart vel, pole angle, pole angular vel) are used
unchanged. This is the cleanest adaptation that preserves original CartPole dynamics.

---

## Expected Performance

| Steps | Expected mean return (10 eval eps) |
|---|---|
| 10k  | ~150–250 |
| 50k  | ~350–450 |
| 150k | ~470–500 |
| 300k | ~490–500 |

CartPole-v1 max return = 500 (episode terminates at 500 steps).
