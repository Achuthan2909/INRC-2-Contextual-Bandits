"""Cumulative regret and reward curves for the week-level LinUCB.

For each dev instance the script runs five policies in sequence:
  1. LinUCB (trained checkpoint, eval mode — no parameter updates)
  2. CoverageFirst always
  3. FatigueAware always
  4. WeekendBalancing always
  5. PreferenceRespecting always

All five policies are seeded identically and receive the same scenario /
initial history / week data, so the per-week rewards are directly comparable.

The oracle best-fixed-arm is determined after collection as whichever fixed
arm achieves the highest cumulative reward across all instances.

Output plots:
  • Cumulative reward curves for all policies (plus oracle highlighted)
  • Cumulative regret of LinUCB vs. oracle (should converge toward 0 if
    the bandit is learning the right context-arm mapping)

Usage:
    PYTHONPATH=src python scripts/plot_cumulative_regret.py \\
        --week-checkpoint runs/final_week.npz \\
        --split dev --max-instances 30 \\
        --out-reward plots/cumulative_reward.png \\
        --out-regret plots/cumulative_regret.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bandit.linucb import LinUCB
from data.splits import split_instances
from schedule.penalty import compute_penalty, _compute_week_history
from schedule.representation import DAY_NAMES_FULL, Schedule
from week_level.arms import (
    CoverageFirstArm,
    FatigueAwareArm,
    PreferenceRespectingArm,
    WeekendBalancingArm,
)
from week_level.arms.base import WeekArm
from week_level.context_builder import build_context


FIXED_ARM_NAMES = [
    "Coverage First",
    "Fatigue Aware",
    "Weekend Balancing",
    "Preference Respecting",
]

ARM_COLORS = {
    "LinUCB": "#e41a1c",
    "Coverage First": "#4daf4a",
    "Fatigue Aware": "#ff7f00",
    "Weekend Balancing": "#984ea3",
    "Preference Respecting": "#377eb8",
    "Oracle": "#000000",
}


# ─── single-policy evaluator ──────────────────────────────────────────────────

def _run_fixed_arm(
    arm: WeekArm,
    scenario: dict[str, Any],
    initial_history: dict[str, Any],
    weeks: list[dict[str, Any]],
) -> list[float]:
    """Always use ``arm``; return per-week rewards (penalty reduction)."""
    nurse_ids = [n["id"] for n in scenario["nurses"]]
    schedule = Schedule(num_weeks=len(weeks), nurse_ids=nurse_ids)
    current_history = initial_history
    rewards: list[float] = []

    for week_idx, week_data in enumerate(weeks):
        penalty_before = compute_penalty(schedule, scenario, weeks, initial_history).total
        assignments = arm.generate(scenario, current_history, week_data)
        for a in assignments:
            day_in_week = DAY_NAMES_FULL.index(a["day"])
            schedule.add_assignment(
                a["nurseId"], week_idx * 7 + day_in_week, a["shiftType"], a["skill"],
            )
        penalty_after = compute_penalty(schedule, scenario, weeks, initial_history).total
        rewards.append(float(penalty_before - penalty_after))
        if week_idx < len(weeks) - 1:
            current_history = _compute_week_history(schedule, week_idx, current_history, scenario)

    return rewards


def _run_linucb(
    bandit: LinUCB,
    arms: list[WeekArm],
    scenario: dict[str, Any],
    initial_history: dict[str, Any],
    weeks: list[dict[str, Any]],
) -> list[float]:
    """Run LinUCB in eval mode (no updates); return per-week rewards."""
    nurse_ids = [n["id"] for n in scenario["nurses"]]
    schedule = Schedule(num_weeks=len(weeks), nurse_ids=nurse_ids)
    current_history = initial_history
    rewards: list[float] = []
    total_weeks = len(weeks)

    for week_idx, week_data in enumerate(weeks):
        ctx, _ = build_context(scenario, current_history, week_data, week_idx, total_weeks)
        arm_idx = bandit.choose(ctx)  # eval only — no update
        arm = arms[arm_idx]

        penalty_before = compute_penalty(schedule, scenario, weeks, initial_history).total
        assignments = arm.generate(scenario, current_history, week_data)
        for a in assignments:
            day_in_week = DAY_NAMES_FULL.index(a["day"])
            schedule.add_assignment(
                a["nurseId"], week_idx * 7 + day_in_week, a["shiftType"], a["skill"],
            )
        penalty_after = compute_penalty(schedule, scenario, weeks, initial_history).total
        rewards.append(float(penalty_before - penalty_after))
        if week_idx < len(weeks) - 1:
            current_history = _compute_week_history(schedule, week_idx, current_history, scenario)

    return rewards


# ─── data collection ──────────────────────────────────────────────────────────

def collect_rewards(
    bandit: LinUCB,
    split: str,
    max_instances: int,
    seed: int,
) -> dict[str, list[float]]:
    """Return concatenated per-week rewards for LinUCB and each fixed arm."""
    fixed_arms = [
        CoverageFirstArm(),
        FatigueAwareArm(),
        WeekendBalancingArm(),
        PreferenceRespectingArm(),
    ]
    linucb_arms = [CoverageFirstArm(), FatigueAwareArm(), WeekendBalancingArm(), PreferenceRespectingArm()]

    all_rewards: dict[str, list[float]] = {
        "LinUCB": [],
        **{name: [] for name in FIXED_ARM_NAMES},
    }

    count = 0
    for inst in split_instances(split, seed=seed, shuffle=False, week_combos_per_scenario=3):
        if count >= max_instances:
            break
        try:
            # LinUCB
            r_lin = _run_linucb(bandit, linucb_arms, inst.scenario, inst.initial_history, inst.weeks)
            all_rewards["LinUCB"].extend(r_lin)

            # Each fixed arm
            for arm, name in zip(fixed_arms, FIXED_ARM_NAMES):
                r_fixed = _run_fixed_arm(arm, inst.scenario, inst.initial_history, inst.weeks)
                all_rewards[name].extend(r_fixed)

            print(
                f"  inst {count+1}: {inst.dataset_name}  "
                f"LinUCB={sum(r_lin):.0f}  "
                + "  ".join(
                    f"{name.split()[0]}={sum(all_rewards[name][-len(r_lin):]):.0f}"
                    for name in FIXED_ARM_NAMES
                )
            )
        except Exception as e:
            print(f"  [skip {getattr(inst, 'dataset_name', '?')}] {e}", file=sys.stderr)
            continue
        count += 1

    return all_rewards


# ─── plotting ─────────────────────────────────────────────────────────────────

def plot_cumulative_reward(
    all_rewards: dict[str, list[float]],
    out_path: str,
) -> None:
    """Plot cumulative reward over rounds for each policy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    lengths = [len(v) for v in all_rewards.values()]
    min_len = min(lengths)

    # Oracle = best fixed arm by total cumulative reward
    fixed_totals = {name: sum(all_rewards[name][:min_len]) for name in FIXED_ARM_NAMES}
    oracle_name = max(fixed_totals, key=fixed_totals.__getitem__)

    rounds = np.arange(1, min_len + 1)

    for name, rewards in all_rewards.items():
        r = np.array(rewards[:min_len])
        cum = np.cumsum(r)
        lw = 2.5 if name == "LinUCB" else 1.5
        ls = "-" if name in ("LinUCB", oracle_name) else "--"
        alpha = 1.0 if name in ("LinUCB", oracle_name) else 0.6
        color = ARM_COLORS.get(name, "gray")
        label = name + (" ← Oracle best-fixed" if name == oracle_name else "")
        ax.plot(rounds, cum, color=color, lw=lw, ls=ls, alpha=alpha, label=label)

    ax.set_xlabel("Cumulative rounds (weeks across all instances)", fontsize=11)
    ax.set_ylabel("Cumulative reward (penalty reduction)", fontsize=11)
    ax.set_title(
        "Week-Level LinUCB vs. Fixed Arms: Cumulative Reward\n"
        f"Oracle = '{oracle_name}' (best fixed arm in hindsight)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.8, ls=":")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cumulative_regret(
    all_rewards: dict[str, list[float]],
    out_path: str,
) -> None:
    """Plot cumulative regret of LinUCB vs. oracle best-fixed arm."""
    lengths = [len(v) for v in all_rewards.values()]
    min_len = min(lengths)

    fixed_totals = {name: sum(all_rewards[name][:min_len]) for name in FIXED_ARM_NAMES}
    oracle_name = max(fixed_totals, key=fixed_totals.__getitem__)

    linucb_r = np.array(all_rewards["LinUCB"][:min_len])
    oracle_r = np.array(all_rewards[oracle_name][:min_len])
    regret = np.cumsum(oracle_r - linucb_r)
    rounds = np.arange(1, min_len + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left panel: cumulative regret vs. oracle
    ax = axes[0]
    ax.plot(rounds, regret, color=ARM_COLORS["LinUCB"], lw=2.2, label="LinUCB regret vs. oracle")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    # Shade positive (LinUCB losing) vs negative (LinUCB winning)
    ax.fill_between(rounds, regret, 0, where=(regret > 0), alpha=0.15, color="red", label="LinUCB below oracle")
    ax.fill_between(rounds, regret, 0, where=(regret <= 0), alpha=0.15, color="green", label="LinUCB above oracle")
    ax.set_xlabel("Cumulative rounds (weeks)", fontsize=11)
    ax.set_ylabel("Cumulative regret (oracle − LinUCB)", fontsize=11)
    ax.set_title(f"Cumulative Regret\n(oracle = '{oracle_name}')", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: per-instance mean reward comparison (bar chart)
    ax2 = axes[1]
    policy_means = {name: float(np.mean(all_rewards[name][:min_len])) for name in all_rewards}
    names = list(policy_means.keys())
    vals = [policy_means[n] for n in names]
    colors = [ARM_COLORS.get(n, "gray") for n in names]
    bars = ax2.bar(range(len(names)), vals, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax2.set_ylabel("Mean per-week reward (penalty reduction)", fontsize=10)
    ax2.set_title("Mean Per-Round Reward by Policy", fontsize=11)
    ax2.axhline(0, color="gray", lw=0.8, ls=":")
    ax2.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(abs(v) for v in vals) * 0.02,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.suptitle(
        f"Week-Level LinUCB: Regret & Reward Analysis  ({min_len} rounds total)",
        fontsize=12, y=1.01,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Cumulative regret curves for week-level LinUCB.")
    p.add_argument("--week-checkpoint", default="runs/final_week.npz")
    p.add_argument("--split", default="dev")
    p.add_argument("--max-instances", type=int, default=30,
                   help="Dev instances to evaluate (default 30).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-reward", default="plots/cumulative_reward.png")
    p.add_argument("--out-regret", default="plots/cumulative_regret.png")
    args = p.parse_args()

    print(f"\n[regret] Loading checkpoint: {args.week_checkpoint}")
    bandit = LinUCB.load(args.week_checkpoint)
    print(f"  num_arms={bandit.num_arms}  context_dim={bandit.context_dim}  alpha={bandit.alpha}")

    print(f"[regret] Collecting rewards on split='{args.split}' (max {args.max_instances} instances)…")
    all_rewards = collect_rewards(bandit, args.split, args.max_instances, args.seed)

    lengths = {k: len(v) for k, v in all_rewards.items()}
    print(f"\nReward sequence lengths: {lengths}")

    # Summary
    for name, rewards in sorted(all_rewards.items()):
        r = np.array(rewards)
        print(f"  {name:28s}: mean={r.mean():.2f}  std={r.std():.2f}  total={r.sum():.0f}")

    fixed_totals = {name: sum(all_rewards[name]) for name in FIXED_ARM_NAMES}
    oracle = max(fixed_totals, key=fixed_totals.__getitem__)
    print(f"\nOracle best-fixed arm: '{oracle}' (total={fixed_totals[oracle]:.0f})")

    print("\n[regret] Generating cumulative reward plot…")
    plot_cumulative_reward(all_rewards, args.out_reward)

    print("[regret] Generating cumulative regret plot…")
    plot_cumulative_regret(all_rewards, args.out_regret)


if __name__ == "__main__":
    main()
