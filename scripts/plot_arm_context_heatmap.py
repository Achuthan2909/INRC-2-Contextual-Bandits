"""Per-arm selection frequency heatmap vs. context regime.

For week-level (4 arms × 5 regimes): loads the trained LinUCB, evaluates it on
dev instances in read-only mode (no parameter updates), records which arm was
picked and in which context, then classifies each context into a "regime" by
the feature that is most extreme relative to all observed contexts (percentile
argmax). Plots P(arm | regime) as a heatmap.

For repair-level (18 arms × 6 regimes): same approach using the dominant
penalty-share component (argmax of features 0-5, which are already fractional
shares summing to ~1).

Usage:
    PYTHONPATH=src python scripts/plot_arm_context_heatmap.py \\
        --week-checkpoint runs/final_week.npz \\
        --repair-checkpoint runs/final_repair.npz \\
        --split dev --max-instances 30 --repair-rounds 200 \\
        --out-week plots/arm_context_heatmap_week.png \\
        --out-repair plots/arm_context_heatmap_repair.png
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
import seaborn as sns

from bandit.linucb import LinUCB
from data.splits import split_instances
from repair_level.context import FEATURE_LABELS as REPAIR_FEATURE_LABELS, build_repair_context
from repair_level.init import generate_initial_schedule
from repair_level.linucb_selector import LinUCBRepairSelector
from repair_level.repairs import build_all_strategies
from repair_level.repairs.base import RepairStrategy
from repair_level.runner import run_repairs
from schedule.penalty import compute_penalty, _compute_week_history
from schedule.representation import DAY_NAMES_FULL, Schedule
from week_level.arms import (
    CoverageFirstArm,
    FatigueAwareArm,
    PreferenceRespectingArm,
    WeekendBalancingArm,
)
from week_level.context_builder import FEATURE_LABELS as WEEK_FEATURE_LABELS, build_context


# ─── labels ───────────────────────────────────────────────────────────────────

WEEK_ARM_LABELS = [
    "Coverage\nFirst",
    "Fatigue\nAware",
    "Weekend\nBalancing",
    "Preference\nRespecting",
]

# Regime = argmax of percentile-normalised features 0-4 (feature 5 = week_position, excluded)
WEEK_REGIME_LABELS = [
    "Coverage\ntight",
    "Fatigue\nloaded",
    "Weekend\nuneven",
    "Preference\nheavy",
    "Assignment\nsaturated",
]

# Repair regime = argmax of features 0-5 (feature 6 = progress_ratio, excluded)
REPAIR_REGIME_LABELS = [
    "S1\nCoverage",
    "S2\nConsec. work",
    "S3\nDays-off",
    "S4\nPrefs",
    "S5+S7\nWeekend",
    "S6\nTotal assign.",
]


# ─── week-level collection ────────────────────────────────────────────────────

def collect_week_records(
    bandit: LinUCB,
    split: str,
    max_instances: int,
    seed: int,
) -> tuple[list[int], list[np.ndarray]]:
    """Return (arm_indices, context_vectors) over all weeks in the eval set."""
    arms = [CoverageFirstArm(), FatigueAwareArm(), WeekendBalancingArm(), PreferenceRespectingArm()]

    arm_indices: list[int] = []
    contexts: list[np.ndarray] = []
    count = 0

    for inst in split_instances(split, seed=seed, shuffle=False, week_combos_per_scenario=3):
        if count >= max_instances:
            break
        try:
            nurse_ids = [n["id"] for n in inst.scenario["nurses"]]
            schedule = Schedule(num_weeks=len(inst.weeks), nurse_ids=nurse_ids)
            current_history = inst.initial_history
            total_weeks = len(inst.weeks)

            for week_idx, week_data in enumerate(inst.weeks):
                ctx, _ = build_context(
                    inst.scenario, current_history, week_data, week_idx, total_weeks,
                )
                arm_idx = bandit.choose(ctx)  # eval only — no update
                arm_indices.append(arm_idx)
                contexts.append(ctx.copy())

                # Advance schedule so history is realistic for subsequent weeks
                arm = arms[arm_idx]
                assignments = arm.generate(inst.scenario, current_history, week_data)
                for a in assignments:
                    day_in_week = DAY_NAMES_FULL.index(a["day"])
                    global_day = week_idx * 7 + day_in_week
                    schedule.add_assignment(
                        a["nurseId"], global_day, a["shiftType"], a["skill"],
                    )
                if week_idx < total_weeks - 1:
                    current_history = _compute_week_history(
                        schedule, week_idx, current_history, inst.scenario,
                    )
        except Exception as e:
            print(f"  [skip week-level {getattr(inst, 'dataset_name', '?')}] {e}", file=sys.stderr)
            continue
        count += 1

    return arm_indices, contexts


# ─── repair-level collection ──────────────────────────────────────────────────

class _RecordingRepairSelector(LinUCBRepairSelector):
    """LinUCBRepairSelector variant that records picks but never updates the bandit."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.arm_records: list[int] = []
        self.ctx_records: list[np.ndarray] = []

    def update(self, strategy_name: str, reward: float) -> None:
        # Intentionally frozen — no parameter updates during eval.
        if self._last_arm_idx is not None and self._last_ctx is not None:
            self.arm_records.append(self._last_arm_idx)
            self.ctx_records.append(self._last_ctx.copy())

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "records": len(self.arm_records)}


def collect_repair_records(
    linucb: LinUCB,
    strategy_names: list[str],
    split: str,
    max_instances: int,
    num_rounds: int,
    seed: int,
) -> tuple[list[int], list[np.ndarray]]:
    """Return (arm_indices, context_vectors) over all repair rounds in the eval set."""
    all_arm_indices: list[int] = []
    all_contexts: list[np.ndarray] = []
    count = 0

    for inst in split_instances(split, seed=seed, shuffle=False, week_combos_per_scenario=2):
        if count >= max_instances:
            break
        try:
            strategies = build_all_strategies(
                inst.scenario, inst.initial_history, inst.weeks, seed=seed,
            )
            names_here = [s.name for s in strategies]
            if names_here != strategy_names:
                continue

            selector = _RecordingRepairSelector(
                strategy_names=strategy_names,
                alpha=linucb.alpha,
                seed=seed,
                linucb=linucb,
            )
            schedule = generate_initial_schedule(
                inst.scenario, inst.initial_history, inst.weeks,
            )
            run_repairs(
                scenario=inst.scenario,
                history=inst.initial_history,
                week_data_list=inst.weeks,
                strategies=strategies,
                schedule=schedule,
                selector=selector,
                num_rounds=num_rounds,
                seed=seed + count,
            )
            all_arm_indices.extend(selector.arm_records)
            all_contexts.extend(selector.ctx_records)
        except Exception as e:
            print(f"  [skip repair-level {getattr(inst, 'dataset_name', '?')}] {e}", file=sys.stderr)
            continue
        count += 1

    return all_arm_indices, all_contexts


# ─── heatmap builder ──────────────────────────────────────────────────────────

def build_frequency_matrix(
    arm_indices: list[int],
    contexts: list[np.ndarray],
    n_arms: int,
    n_regimes: int,
    regime_feature_slice: slice,
    percentile_normalise: bool = True,
) -> np.ndarray:
    """Build count matrix (n_arms, n_regimes) of arm picks per context regime.

    Regime is determined by the argmax of the selected feature slice after
    optional percentile normalisation (so each feature has equal dynamic range).
    """
    if not contexts:
        return np.zeros((n_arms, n_regimes))

    ctx_array = np.array(contexts)  # (N, D)
    feat = ctx_array[:, regime_feature_slice]  # (N, K)

    if percentile_normalise:
        from scipy.stats import rankdata
        ranked = np.zeros_like(feat)
        for j in range(feat.shape[1]):
            ranked[:, j] = rankdata(feat[:, j]) / feat.shape[0]
        feat = ranked

    regimes = np.argmax(feat, axis=1)  # (N,)

    matrix = np.zeros((n_arms, n_regimes), dtype=float)
    for arm_idx, regime in zip(arm_indices, regimes):
        if 0 <= arm_idx < n_arms and 0 <= regime < n_regimes:
            matrix[arm_idx, regime] += 1

    # Normalise each column → P(arm | regime)
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    return matrix / col_sums


# ─── plot helpers ─────────────────────────────────────────────────────────────

def plot_heatmap(
    matrix: np.ndarray,
    arm_labels: list[str],
    regime_labels: list[str],
    title: str,
    out_path: str,
    fig_size: tuple[float, float] = (9, 5),
) -> None:
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=regime_labels,
        yticklabels=arm_labels,
        linewidths=0.4,
        linecolor="white",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": "P(arm | context regime)", "shrink": 0.8},
    )
    ax.set_xlabel("Context Regime (dominant feature)", fontsize=11, labelpad=8)
    ax.set_ylabel("Arm Selected", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)

    # Highlight the max in each column
    n_arms, n_regimes = matrix.shape
    for j in range(n_regimes):
        col = matrix[:, j]
        if col.sum() > 0:
            best = int(np.argmax(col))
            ax.add_patch(plt.Rectangle(
                (j, best), 1, 1,
                fill=False, edgecolor="crimson", lw=2.5, clip_on=False,
            ))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Arm-selection frequency heatmap vs. context regime."
    )
    p.add_argument("--week-checkpoint", default="runs/final_week.npz")
    p.add_argument("--repair-checkpoint", default="runs/final_repair.npz")
    p.add_argument("--split", default="dev")
    p.add_argument("--max-instances", type=int, default=30,
                   help="Dev instances to evaluate on (default 30).")
    p.add_argument("--repair-rounds", type=int, default=200,
                   help="Repair rounds per instance for repair-level eval.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-week", default="plots/arm_context_heatmap_week.png")
    p.add_argument("--out-repair", default="plots/arm_context_heatmap_repair.png")
    args = p.parse_args()

    # ── Week-level ────────────────────────────────────────────────────────────
    print(f"\n[week] Loading checkpoint: {args.week_checkpoint}")
    week_bandit = LinUCB.load(args.week_checkpoint)
    print(f"  num_arms={week_bandit.num_arms}  context_dim={week_bandit.context_dim}  alpha={week_bandit.alpha}")

    print(f"[week] Collecting arm picks on split='{args.split}' "
          f"(max {args.max_instances} instances)…")
    week_arm_idx, week_contexts = collect_week_records(
        week_bandit, args.split, args.max_instances, args.seed,
    )
    print(f"  collected {len(week_arm_idx)} week-level picks")

    if week_arm_idx:
        freq_counts = {i: week_arm_idx.count(i) for i in range(week_bandit.num_arms)}
        labels = ["coverage_first", "fatigue_aware", "weekend_balancing", "preference_respecting"]
        for i, name in enumerate(labels):
            print(f"  arm {i} ({name}): {freq_counts.get(i, 0)} picks "
                  f"({100*freq_counts.get(i,0)/len(week_arm_idx):.1f}%)")

    week_matrix = build_frequency_matrix(
        week_arm_idx,
        week_contexts,
        n_arms=week_bandit.num_arms,
        n_regimes=5,
        regime_feature_slice=slice(0, 5),  # features 0-4; exclude week_position (feature 5)
        percentile_normalise=True,
    )

    plot_heatmap(
        week_matrix,
        arm_labels=WEEK_ARM_LABELS,
        regime_labels=WEEK_REGIME_LABELS,
        title=(
            "Week-Level LinUCB: Arm Selection by Context Regime\n"
            f"P(arm | regime)  —  {len(week_arm_idx)} decisions, {args.max_instances} instances"
        ),
        out_path=args.out_week,
        fig_size=(9, 4),
    )

    # ── Repair-level ──────────────────────────────────────────────────────────
    print(f"\n[repair] Loading checkpoint: {args.repair_checkpoint}")
    repair_bandit = LinUCB.load(args.repair_checkpoint)
    print(f"  num_arms={repair_bandit.num_arms}  context_dim={repair_bandit.context_dim}  alpha={repair_bandit.alpha}")

    # Discover strategy names from the first available dev instance
    strategy_names: list[str] = []
    for inst in split_instances(args.split, seed=args.seed, shuffle=False, week_combos_per_scenario=1):
        try:
            strats = build_all_strategies(
                inst.scenario, inst.initial_history, inst.weeks, seed=args.seed,
            )
            strategy_names = [s.name for s in strats]
            break
        except Exception:
            continue

    if not strategy_names:
        print("[repair] Could not discover strategy names; skipping repair heatmap.")
    else:
        print(f"  {len(strategy_names)} repair strategies")
        print(f"[repair] Collecting arm picks on split='{args.split}' "
              f"(max {args.max_instances} instances, {args.repair_rounds} rounds each)…")
        repair_arm_idx, repair_contexts = collect_repair_records(
            repair_bandit,
            strategy_names,
            args.split,
            args.max_instances,
            args.repair_rounds,
            args.seed,
        )
        print(f"  collected {len(repair_arm_idx)} repair-level picks")

        repair_matrix = build_frequency_matrix(
            repair_arm_idx,
            repair_contexts,
            n_arms=repair_bandit.num_arms,
            n_regimes=6,
            regime_feature_slice=slice(0, 6),  # features 0-5; exclude progress_ratio (feature 6)
            percentile_normalise=False,  # shares already sum to ~1, comparable scale
        )

        # Short arm labels for the plot
        short_names = [
            n.replace("_", "\n", 1) if len(n) > 18 else n
            for n in strategy_names
        ]

        plot_heatmap(
            repair_matrix,
            arm_labels=short_names,
            regime_labels=REPAIR_REGIME_LABELS,
            title=(
                "Repair-Level LinUCB: Arm Selection by Dominant Penalty Component\n"
                f"P(arm | regime)  —  {len(repair_arm_idx)} decisions, {args.max_instances} instances"
            ),
            out_path=args.out_repair,
            fig_size=(10, 10),
        )


if __name__ == "__main__":
    main()
