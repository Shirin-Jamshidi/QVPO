# qvpo_cartpole/train.py
"""
QVPO on CartPole-v1  —  standalone training script.

CartPole adaptation
-------------------
CartPole-v1's native action space is Discrete(2).  QVPO requires a
continuous action space (the diffusion model lives in R^{action_dim}).
We wrap the environment so that:
  - The policy outputs a continuous scalar a ∈ [-1, 1]
  - We map  a < 0  →  gym action 0 (push left)
           a ≥ 0  →  gym action 1 (push right)
This preserves the CartPole dynamics while giving QVPO a 1-D continuous
action space to model.

Algorithm 1 mapping (paper → this script)
-----------------------------------------
  Lines 1-3  : interaction loop  (collect_steps)
  Lines 4-11 : agent.train_step  (one gradient step per env step)

Run
---
  python train.py                         # defaults
  python train.py --seed 1 --K_b 10      # custom K_b
  python train.py --headless              # no eval renders
"""

import os
import random
import argparse
import numpy as np
import torch

import gymnasium as gym

from agent        import QVPO
from replay_buffer import ReplayBuffer


# ─────────────────────────────────────────────────────────────────────────────
# CartPole continuous-action wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CartPoleContinuousWrapper(gym.Wrapper):
    """
    Wraps CartPole-v1 to accept a continuous scalar action a ∈ [-1, 1].
      a < 0  →  discrete action 0  (push cart left)
      a ≥ 0  →  discrete action 1  (push cart right)

    Observation space is unchanged (4-D Box).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space      = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        discrete = int(action[0] >= 0.0)
        return self.env.step(discrete)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(agent: QVPO, n_episodes: int = 10, seed: int = 0) -> float:
    """Run n_episodes with the K_b-efficient policy; return mean episodic return."""
    env = CartPoleContinuousWrapper(gym.make("CartPole-v1"))
    returns = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_ret   = 0.0
        done     = False
        while not done:
            action = agent.select_action(state)
            state, reward, term, trunc, _ = env.step(action)
            ep_ret += reward
            done    = term or trunc
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    set_seed(cfg.seed)

    # ── Environment ──────────────────────────────────────────────────────────
    env = CartPoleContinuousWrapper(gym.make("CartPole-v1"))
    STATE_DIM  = env.observation_space.shape[0]   # 4
    ACTION_DIM = env.action_space.shape[0]         # 1

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = QVPO(
        state_dim          = STATE_DIM,
        action_dim         = ACTION_DIM,
        device             = device,
        n_diffusion_steps  = cfg.n_diffusion_steps,
        beta_min           = cfg.beta_min,
        beta_max           = cfg.beta_max,
        hidden_dim         = cfg.hidden_dim,
        time_emb_dim       = cfg.time_emb_dim,
        n_policy_samples   = cfg.n_policy_samples,
        n_uniform_samples  = cfg.n_uniform_samples,
        omega_ent          = cfg.omega_ent,
        K_b                = cfg.K_b,
        K_t                = cfg.K_t,
        gamma              = cfg.gamma,
        tau                = cfg.tau,
        lr_critic          = cfg.lr_critic,
        lr_actor           = cfg.lr_actor,
    )

    # ── Replay buffer ────────────────────────────────────────────────────────
    replay = ReplayBuffer(
        state_dim  = STATE_DIM,
        action_dim = ACTION_DIM,
        capacity   = cfg.replay_capacity,
        device     = device,
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    log = {k: [] for k in
           ["step", "ep_return", "loss_critic", "loss_q_vlo", "loss_entropy", "eval_return"]}

    # ── Interaction loop ─────────────────────────────────────────────────────
    state, _    = env.reset(seed=cfg.seed)
    ep_return   = 0.0
    ep_count    = 0
    global_step = 0

    print(f"\n{'='*60}")
    print(f"  QVPO — CartPole-v1  |  T={cfg.n_diffusion_steps}  Nd={cfg.n_policy_samples}"
          f"  Ne={cfg.n_uniform_samples}  K_b={cfg.K_b}  K_t={cfg.K_t}")
    print(f"{'='*60}\n")

    while global_step < cfg.total_steps:

        # ── Algorithm 1, line 2: K_b-efficient action selection ──────────────
        if global_step < cfg.warmup_steps:
            # Random warmup before diffusion policy is meaningful
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)             # (action_dim,)

        # ── Algorithm 1, line 3: step environment, store transition ──────────
        next_state, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        replay.add(state, action, reward, next_state, float(done))
        state      = next_state
        ep_return += reward
        global_step += 1

        if done:
            ep_count += 1
            log["step"].append(global_step)
            log["ep_return"].append(ep_return)

            if ep_count % cfg.log_interval == 0:
                recent = np.mean(log["ep_return"][-20:])
                print(f"  step={global_step:7d}  ep={ep_count:4d}  "
                      f"ret(last20)={recent:6.1f}  buffer={len(replay):6d}")

            ep_return = 0.0
            state, _  = env.reset()

        # ── Algorithm 1, lines 4-11: one gradient step per env step ──────────
        if (len(replay) >= cfg.batch_size and
                global_step >= cfg.warmup_steps):
            metrics = agent.train_step(replay, cfg.batch_size)
            for k, v in metrics.items():
                log[k].append(v)

        # ── Periodic evaluation ───────────────────────────────────────────────
        if global_step % cfg.eval_interval == 0 and global_step >= cfg.warmup_steps:
            eval_ret = evaluate(agent, n_episodes=cfg.eval_episodes, seed=cfg.seed + 999)
            log["eval_return"].append(eval_ret)
            print(f"\n  ── Eval @ step {global_step:,}  mean_return={eval_ret:.1f} ──\n")

            # Save checkpoint
            ckpt_path = os.path.join(cfg.save_dir, f"qvpo_{global_step}.pt")
            agent.save(ckpt_path)

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_ret = evaluate(agent, n_episodes=20, seed=cfg.seed + 1234)
    print(f"\n{'='*60}")
    print(f"  Final eval (20 eps):  mean={final_ret:.1f}")
    print(f"{'='*60}")

    agent.save(os.path.join(cfg.save_dir, "qvpo_final.pt"))
    np.save(os.path.join(cfg.save_dir, "log.npy"), log)
    env.close()
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("QVPO — CartPole-v1 baseline")

    # Diffusion model
    p.add_argument("--n_diffusion_steps",  type=int,   default=5,
                   help="Diffusion chain length T (paper: 5)")
    p.add_argument("--beta_min",           type=float, default=0.1,
                   help="Linear schedule lower bound")
    p.add_argument("--beta_max",           type=float, default=0.5,
                   help="Linear schedule upper bound")
    p.add_argument("--hidden_dim",         type=int,   default=256)
    p.add_argument("--time_emb_dim",       type=int,   default=16)

    # Policy update (§4.1-4.3)
    p.add_argument("--n_policy_samples",   type=int,   default=64,
                   help="Nd: policy samples per state for qadv weights (paper: 64)")
    p.add_argument("--n_uniform_samples",  type=int,   default=64,
                   help="Ne: uniform samples per state for entropy term (paper: 64)")
    p.add_argument("--omega_ent",          type=float, default=1.0,
                   help="Entropy regularisation coefficient (paper: 1.0)")

    # Behavior policy (§4.4)
    p.add_argument("--K_b",               type=int,   default=10,
                   help="K_b: behavior policy candidate actions (paper: 10)")
    p.add_argument("--K_t",               type=int,   default=2,
                   help="K_t: target policy candidate actions  (paper: 2)")

    # RL / critic
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--tau",               type=float, default=0.005)
    p.add_argument("--lr_critic",         type=float, default=3e-4)
    p.add_argument("--lr_actor",          type=float, default=3e-4)
    p.add_argument("--batch_size",        type=int,   default=256)
    p.add_argument("--replay_capacity",   type=int,   default=300_000)

    # Training schedule
    p.add_argument("--total_steps",       type=int,   default=300_000,
                   help="Total environment steps")
    p.add_argument("--warmup_steps",      type=int,   default=1_000,
                   help="Random-action steps before training begins")
    p.add_argument("--eval_interval",     type=int,   default=10_000)
    p.add_argument("--eval_episodes",     type=int,   default=10)
    p.add_argument("--log_interval",      type=int,   default=10,
                   help="Print every N episodes")

    # Misc
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--save_dir",          type=str,   default="checkpoints")

    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
