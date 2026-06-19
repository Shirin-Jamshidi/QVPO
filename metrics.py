# metrics.py

import time
import os
from typing import Optional, List
import numpy as np


# ─────────────────────────────────────────────────────────────

class MetricsTracker:
    def __init__(self, method_name: str):
        self.method_name = method_name
        self._t0 = time.time()

        # step-level logs
        self.steps = []
        self.critic_losses = []
        self.policy_losses = []
        self.wall_times = []

        # eval-level logs
        self.eval_steps = []
        self.eval_means = []
        self.eval_stds = []
        self.eval_mins = []
        self.eval_maxs = []
        self.eval_medians = []
        self.eval_iqms = []
        self.eval_wall_times = []

    # ─────────────────────────────────────────────────────────
    def log_step(self, step: int, policy_loss: float,
                 critic_loss: Optional[float] = None):
        self.steps.append(step)
        self.policy_losses.append(float(policy_loss))
        self.critic_losses.append(
            float(critic_loss) if critic_loss is not None else float("nan")
        )
        self.wall_times.append(time.time() - self._t0)

    # ─────────────────────────────────────────────────────────
    def log_eval(self, step: int, returns: List[float]):
        arr = np.array(returns, dtype=np.float64)

        self.eval_steps.append(step)
        self.eval_means.append(float(arr.mean()))
        self.eval_stds.append(float(arr.std()))
        self.eval_mins.append(float(arr.min()))
        self.eval_maxs.append(float(arr.max()))
        self.eval_medians.append(float(np.median(arr)))
        self.eval_iqms.append(float(_iqm(arr)))
        self.eval_wall_times.append(time.time() - self._t0)

    # ─────────────────────────────────────────────────────────
    def compute_stability_step(self, threshold=450.0):
        """
        Find first eval step where performance reaches threshold
        and stays above it for the rest of training.
        """
        if len(self.eval_means) == 0:
            return -1

        means = np.array(self.eval_means)
        steps = np.array(self.eval_steps)

        for i in range(len(means)):
            if means[i] < threshold:
                continue

            if np.all(means[i:] >= threshold):
                return int(steps[i])

        return -1  # never stabilized

    # ─────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        stability = self.compute_stability_step()

        np.savez(
            path,
            method_name=np.array([self.method_name]),
            steps=np.array(self.steps),
            critic_losses=np.array(self.critic_losses),
            policy_losses=np.array(self.policy_losses),
            wall_times=np.array(self.wall_times),

            eval_steps=np.array(self.eval_steps),
            eval_means=np.array(self.eval_means),
            eval_stds=np.array(self.eval_stds),
            eval_mins=np.array(self.eval_mins),
            eval_maxs=np.array(self.eval_maxs),
            eval_medians=np.array(self.eval_medians),
            eval_iqms=np.array(self.eval_iqms),
            eval_wall_times=np.array(self.eval_wall_times),

            stability_step=np.array([stability])
        )

    # ─────────────────────────────────────────────────────────
    def print_summary(self):
        if len(self.steps) == 0:
            print(f"[{self.method_name}] No data.")
            return

        stability = self.compute_stability_step()

        print(f"\n── {self.method_name} Summary ──")
        print(f"Total steps       : {len(self.steps)}")
        print(f"Wall time (min)   : {self.wall_times[-1]/60:.2f}")

        if self.eval_means:
            print(f"Final return      : {self.eval_means[-1]:.1f} ± {self.eval_stds[-1]:.1f}")
            print(f"Best return       : {max(self.eval_means):.1f}")
            print(f"Stability step    : {stability}")


# ─────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────

def print_comparison_table(npz_paths: List[str]):

    rows = []

    for path in npz_paths:
        if not os.path.exists(path):
            continue

        d = np.load(path, allow_pickle=True)

        if len(d["eval_means"]) == 0:
            continue

        stability = int(d["stability_step"][0]) if "stability_step" in d else -1

        rows.append({
            "method": str(d["method_name"][0]),
            "steps": int(d["steps"][-1]) if len(d["steps"]) else 0,
            "mean": float(d["eval_means"][-1]),
            "std": float(d["eval_stds"][-1]),
            "median": float(d["eval_medians"][-1]),
            "iqm": float(d["eval_iqms"][-1]),
            "best": float(d["eval_maxs"][-1]),
            "worst": float(d["eval_mins"][-1]),
            "peak": float(np.max(d["eval_means"])),
            "time": float(d["wall_times"][-1] / 60.0),
            "stability": stability
        })

    if not rows:
        print("No data to compare.")
        return

    # header
    headers = ["Method", "Steps", "Mean ± Std", "Median", "IQM",
               "Best", "Worst", "Peak", "Time", "Stability"]

    print("\n" + " | ".join(headers))
    print("-" * 110)

    for r in sorted(rows, key=lambda x: x["mean"], reverse=True):
        print(
            f"{r['method']:12} | "
            f"{r['steps']:7d} | "
            f"{r['mean']:6.1f} ± {r['std']:<5.1f} | "
            f"{r['median']:6.1f} | "
            f"{r['iqm']:6.1f} | "
            f"{r['best']:5.0f} | "
            f"{r['worst']:5.0f} | "
            f"{r['peak']:6.1f} | "
            f"{r['time']:6.1f} | "
            f"{r['stability']:9d}"
        )

    print("\nStability = first step where return >= 450 and stays there.")


# ─────────────────────────────────────────────────────────────

def _iqm(arr):
    q25, q75 = np.percentile(arr, [25, 75])
    middle = arr[(arr >= q25) & (arr <= q75)]
    return float(np.mean(middle)) if len(middle) else float(np.mean(arr))


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_comparison_table([
        "vanilla_diffusion_metrics.npz",
        "qvpo_metrics.npz",
        "qvpo+hy-q_metrics.npz",
    ])