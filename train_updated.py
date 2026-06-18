# qvpo_cartpole/train.py
"""
QVPO on CartPole-v1  —  standalone training script.

CartPole adaptation
"""

import os
import random
import argparse
import numpy as np
import torch

import gymnasium as gym

from agent        import QVPO
from replay_buffer import ReplayBuffer
from metrics import MetricsTracker

# ─────────────────────────────────────────────────────────────────────────────
# Continuous CartPole environment
# ─────────────────────────────────────────────────────────────────────────────

import math


class ContinuousCartPoleEnv:
    """
    Continuous-force CartPole.

    Action: scalar force in [-force_mag, +force_mag].
    Dynamics and termination thresholds are identical to CartPole-v1.
    """

    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    TOTAL_MASS = MASSCART + MASSPOLE
    HALF_LEN = 0.5
    POLEMASS_LEN = MASSPOLE * HALF_LEN
    TAU = 0.02

    THETA_THRESHOLD = 12 * 2 * math.pi / 360
    X_THRESHOLD = 2.4

    def __init__(self, force_mag=10.0, max_steps=500, seed=42):
        self.force_mag = force_mag
        self.max_steps = max_steps
        self._rng = np.random.RandomState(seed)
        self.state = None
        self._step_count = 0

        self.observation_space_shape = (4,)
        self.action_space_low = np.array([-force_mag], dtype=np.float32)
        self.action_space_high = np.array([force_mag], dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.state = self._rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        self._step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        force = float(np.clip(action, -self.force_mag, self.force_mag))

        x, x_dot, theta, theta_dot = self.state

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        tmp = (force + self.POLEMASS_LEN * theta_dot ** 2 * sin_t) / self.TOTAL_MASS
        theta_acc = (self.GRAVITY * sin_t - cos_t * tmp) / (
            self.HALF_LEN * (4.0 / 3.0 - self.MASSPOLE * cos_t ** 2 / self.TOTAL_MASS)
        )
        x_acc = tmp - self.POLEMASS_LEN * theta_acc * cos_t / self.TOTAL_MASS

        x += self.TAU * x_dot
        x_dot += self.TAU * x_acc
        theta += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_count += 1

        terminated = abs(x) > self.X_THRESHOLD or abs(theta) > self.THETA_THRESHOLD
        truncated = self._step_count >= self.max_steps

        reward = 1.0 if not terminated else 0.0
        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        pass

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
    env = ContinuousCartPoleEnv(seed=seed)
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
    env = ContinuousCartPoleEnv(seed=cfg.seed)
    STATE_DIM = env.observation_space_shape[0]   # 4
    ACTION_DIM = 1         # 1

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
    tracker = MetricsTracker("QVPO")

    print(f"\n{'='*60}")
    print(f"  QVPO — CartPole-v1  |  T={cfg.n_diffusion_steps}  Nd={cfg.n_policy_samples}"
          f"  Ne={cfg.n_uniform_samples}  K_b={cfg.K_b}  K_t={cfg.K_t}")
    print(f"{'='*60}\n")

    while global_step < cfg.total_steps:

        # ── Algorithm 1, line 2: K_b-efficient action selection ──────────────
        if global_step < cfg.warmup_steps:
            # Random warmup before diffusion policy is meaningful
            action = np.random.uniform(-10.0, 10.0, size=(1,)).astype(np.float32)
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
            tracker.log_step(
                step=global_step,
                policy_loss=metrics["loss_q_vlo"],   # main one
                critic_loss=metrics["loss_critic"],
            )
            for k, v in metrics.items():
                log[k].append(v)
                
        if global_step % cfg.eval_interval == 0 and global_step > 0:

            returns = []

            env_eval = ContinuousCartPoleEnv(seed=cfg.seed + 999)

            for ep in range(cfg.eval_episodes):
                s, _ = env_eval.reset()
                done = False
                ep_ret = 0.0

                while not done:
                    a = agent.select_action(s)
                    s, r, term, trunc, _ = env_eval.step(a)
                    ep_ret += r
                    done = term or trunc

                returns.append(ep_ret)

            env_eval.close()

            tracker.log_eval(
                step=global_step,
                returns=returns   # ✅ THIS MUST BE A LIST
            )

    tracker.save("qvpo_metrics.npz")

    # ── Final evaluation (20 episodes, print each) ──────────────────────────
    env_eval = ContinuousCartPoleEnv(seed=cfg.seed + 1234)

    returns = []
    for ep in range(20):
        state, _ = env_eval.reset(seed=cfg.seed + 1234 + ep)
        ep_ret = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            state, reward, term, trunc, _ = env_eval.step(action)
            ep_ret += reward
            done = term or trunc

        returns.append(ep_ret)
        print(f"Eval Episode {ep + 1:2d}: return = {ep_ret:.1f}")

    env_eval.close()

    final_ret = float(np.mean(returns))

    print(f"\n{'='*60}")
    print("Final Evaluation Summary (20 episodes)")
    print(f"Mean Return: {final_ret:.1f}")
    print(f"{'='*60}")

    # Save final model + log
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
    p.add_argument("--total_steps",       type=int,   default=10_000,
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
