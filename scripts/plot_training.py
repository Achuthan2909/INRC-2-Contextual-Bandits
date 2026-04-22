"""Training diagnostic plots for LinUCB (week-level or repair-level).

Reads the ``<checkpoint>.trajectory.json`` sidecar produced by either
``src/week_level/train.py`` or ``src/repair_level/train.py`` and writes
plots to ``plots/``:

1. ``reward_rolling.png``      — rolling-window mean reward vs training round
2. ``arm_usage_over_time.png`` — stacked arm picks per 100-round window
3. ``theta_norms.png``         — L2 norm of each arm's θ at snapshot rounds
4. ``final_theta_heatmap.png`` — final θ_a as a heatmap over features
5. ``penalty_reduction.png``   — per-instance Δpenalty (repair sidecars only)

Auto-adapts figure sizes + legend columns + heatmap annotation font to the
number of arms (4 for week-level, 18 for repair-level).

Example:
    python scripts/plot_training.py --sidecar runs/linucb_week_level.trajectory.json
    python scripts/plot_training.py --sidecar runs/linucb_repair_level.trajectory.json \\
        --out-dir plots/repair
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2 or len(x) < window:
        return x.astype(float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_reward_rolling(out_dir: Path, data: dict, window: int = 100) -> None:
    rewards = np.asarray(data["reward_trajectory_scaled"], dtype=float)
    if len(rewards) == 0:
        return
    warm = data.get("warm_start_rounds", 0)
    smoothed = _rolling_mean(rewards, window)
    xs = np.arange(window - 1, window - 1 + len(smoothed)) if len(smoothed) < len(rewards) else np.arange(len(rewards))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, smoothed, linewidth=1.2)
    if warm > 0 and warm < len(rewards):
        ax.axvline(warm, linestyle="--", linewidth=1, alpha=0.6, label=f"warm-start end ({warm})")
        ax.legend()
    ax.set_xlabel("Training round")
    ax.set_ylabel(f"Mean scaled reward (window={window})")
    ax.set_title("Rolling-mean reward over training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_rolling.png", dpi=130)
    plt.close(fig)


def plot_arm_usage_over_time(out_dir: Path, data: dict, window: int = 100) -> None:
    picks = data["pick_trajectory"]
    arm_names = data["arm_names"]
    if not picks:
        return
    # Bin into windows of `window` rounds, count picks per arm per bin.
    n_bins = (len(picks) + window - 1) // window
    counts = np.zeros((n_bins, len(arm_names)), dtype=float)
    name_idx = {n: i for i, n in enumerate(arm_names)}
    for i, name in enumerate(picks):
        counts[i // window, name_idx[name]] += 1
    counts = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(n_bins) * window
    bottom = np.zeros(n_bins)
    for j, name in enumerate(arm_names):
        ax.fill_between(xs, bottom, bottom + counts[:, j], step="post", alpha=0.7, label=name)
        bottom = bottom + counts[:, j]
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training round")
    ax.set_ylabel(f"Arm-pick fraction (window={window})")
    ax.set_title("Arm usage over training")
    ncol = 1 if len(arm_names) <= 8 else 2
    fontsize = 9 if len(arm_names) <= 8 else 7
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        ncol=ncol, fontsize=fontsize,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "arm_usage_over_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_theta_norms(out_dir: Path, data: dict) -> None:
    snaps = data.get("theta_snapshots", [])
    rounds = data.get("snapshot_rounds", [])
    arm_names = data["arm_names"]
    if not snaps or not rounds:
        return
    # snaps: list over time; each item is list[arm][feature]
    arr = np.asarray(snaps)  # shape (T, A, D)
    norms = np.linalg.norm(arr, axis=2)  # shape (T, A)

    fig, ax = plt.subplots(figsize=(8, 4))
    for j, name in enumerate(arm_names):
        ax.plot(rounds, norms[:, j], marker="o", markersize=3, label=name)
    ax.set_xlabel("Training round (at snapshot)")
    ax.set_ylabel("‖θ_a‖₂")
    ax.set_title("θ norm per arm (saturation diagnostic — flat = converged)")
    ax.grid(alpha=0.3)
    ncol = 1 if len(arm_names) <= 8 else 2
    fontsize = 9 if len(arm_names) <= 8 else 7
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        ncol=ncol, fontsize=fontsize,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "theta_norms.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_final_theta_heatmap(out_dir: Path, data: dict) -> None:
    snaps = data.get("theta_snapshots", [])
    arm_names = data["arm_names"]
    feature_labels = data["feature_labels"]
    if not snaps:
        return
    theta_final = np.asarray(snaps[-1])  # shape (A, D)

    n_arms = len(arm_names)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(feature_labels) + 4), 1 + 0.38 * n_arms))
    vmax = np.abs(theta_final).max() or 1.0
    im = ax.imshow(theta_final, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=30, ha="right")
    ax.set_yticks(range(n_arms))
    ax.set_yticklabels(arm_names, fontsize=8 if n_arms > 8 else 10)
    ann_fs = 8 if n_arms <= 8 else (6 if n_arms <= 14 else 5)
    for i in range(theta_final.shape[0]):
        for j in range(theta_final.shape[1]):
            ax.text(j, i, f"{theta_final[i, j]:.1f}", ha="center", va="center", fontsize=ann_fs)
    fig.colorbar(im, ax=ax, shrink=0.8, label="θ weight")
    ax.set_title("Final θ_a per arm × feature (red=positive, blue=negative)")
    fig.tight_layout()
    fig.savefig(out_dir / "final_theta_heatmap.png", dpi=130)
    plt.close(fig)


def plot_penalty_reduction(out_dir: Path, data: dict) -> None:
    """Per-instance initial vs final penalty (repair-level sidecars only)."""
    init = data.get("initial_penalties")
    final = data.get("final_penalties")
    if not init or not final:
        return
    init = np.asarray(init, dtype=float)
    final = np.asarray(final, dtype=float)
    n = min(len(init), len(final))
    init, final = init[:n], final[:n]
    delta = init - final
    xs = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(xs, init, marker=".", linewidth=1, label="initial P", alpha=0.7)
    ax1.plot(xs, final, marker=".", linewidth=1, label="final P", alpha=0.7)
    ax1.set_xlabel("Training instance #")
    ax1.set_ylabel("Penalty")
    ax1.set_title("Per-instance initial vs final penalty")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(xs, delta, marker=".", linewidth=1, color="tab:green")
    ax2.axhline(0, color="k", linewidth=0.6)
    ax2.set_xlabel("Training instance #")
    ax2.set_ylabel("ΔP (initial − final)")
    ax2.set_title(f"Penalty reduction per instance (mean={delta.mean():.0f})")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "penalty_reduction.png", dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training diagnostics for week-level LinUCB.")
    p.add_argument(
        "--sidecar",
        default="runs/linucb_week_level.trajectory.json",
        help="Path to the *.trajectory.json file written alongside the checkpoint.",
    )
    p.add_argument("--out-dir", default="plots", help="Where to write PNG files.")
    p.add_argument("--window", type=int, default=100, help="Rolling-window size (rounds).")
    args = p.parse_args()

    data = json.loads(Path(args.sidecar).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_reward_rolling(out_dir, data, window=args.window)
    plot_arm_usage_over_time(out_dir, data, window=args.window)
    plot_theta_norms(out_dir, data)
    plot_final_theta_heatmap(out_dir, data)
    plot_penalty_reduction(out_dir, data)

    print(f"Wrote plots to {out_dir}/:")
    print("  reward_rolling.png")
    print("  arm_usage_over_time.png")
    print("  theta_norms.png")
    print("  final_theta_heatmap.png")
    if data.get("initial_penalties") and data.get("final_penalties"):
        print("  penalty_reduction.png")

    # Quick textual summary
    picks = data["pick_trajectory"]
    c = Counter(picks)
    total = len(picks)
    print("\nFinal pick distribution:")
    for name, n in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {name}: {n} ({100*n/total:.1f}%)")


if __name__ == "__main__":
    main()
