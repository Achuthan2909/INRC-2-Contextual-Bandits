"""Cross-instance week-level LinUCB training (streams train split, checkpoints θ)."""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from bandit.linucb import LinUCB
from data.instances import enumerate_instances
from data.splits import SPLITS, split_instances
from schedule.penalty import compute_penalty, _compute_week_history
from schedule.representation import DAY_NAMES_FULL, Schedule
from week_level.arms.base import WeekArm
from week_level.context_builder import FEATURE_LABELS, build_context
from week_level.runner import run_week_level


ARM_REGISTRY: dict[str, type[WeekArm]] = {}


def _build_registry() -> dict[str, type[WeekArm]]:
    from week_level.arms.coverage_first import CoverageFirstArm
    from week_level.arms.fatigue_aware import FatigueAwareArm
    from week_level.arms.preference_respecting import PreferenceRespectingArm
    from week_level.arms.weekend_balancing import WeekendBalancingArm

    return {
        "coverage_first": CoverageFirstArm,
        "fatigue_aware": FatigueAwareArm,
        "weekend_balancing": WeekendBalancingArm,
        "preference_respecting": PreferenceRespectingArm,
    }

log = logging.getLogger(__name__)


def _run_instance_forced(
    inst: Any,
    arms: list[WeekArm],
    arm_indices: list[int],
    linucb: LinUCB,
    reward_scale: float,
) -> tuple[list[float], list[str]]:
    """Run one instance with a pre-chosen arm per week, updating the bandit."""
    nurse_ids = [n["id"] for n in inst.scenario["nurses"]]
    schedule = Schedule(num_weeks=len(inst.weeks), nurse_ids=nurse_ids)
    current_history = inst.initial_history
    picked: list[str] = []
    trajectory: list[float] = []

    for week_idx, week_data in enumerate(inst.weeks):
        arm_idx = arm_indices[week_idx % len(arm_indices)]
        arm = arms[arm_idx]
        ctx, _ = build_context(
            inst.scenario, current_history, week_data, week_idx, len(inst.weeks),
        )
        penalty_before = compute_penalty(
            schedule, inst.scenario, inst.weeks, inst.initial_history,
        ).total
        assignments = arm.generate(inst.scenario, current_history, week_data)
        for a in assignments:
            day = DAY_NAMES_FULL.index(a["day"])
            schedule.add_assignment(
                a["nurseId"], week_idx * 7 + day, a["shiftType"], a["skill"],
            )
        penalty_after = compute_penalty(
            schedule, inst.scenario, inst.weeks, inst.initial_history,
        ).total
        reward = float(penalty_before - penalty_after)
        trajectory.append(reward)
        picked.append(arm.name)
        rs = float(reward_scale) if reward_scale else 1.0
        linucb.update(arm_idx, ctx, reward / rs)
        if week_idx < len(inst.weeks) - 1:
            current_history = _compute_week_history(
                schedule, week_idx, current_history, inst.scenario,
            )
    return trajectory, picked


def train_linucb(
    arms: list[WeekArm],
    *,
    split: str = "train",
    dataset_root: str | None = None,
    alpha: float = 1.0,
    max_instances: int | None = None,
    week_combos_per_scenario: int = 20,
    seed: int = 0,
    reward_scale: float = 1000.0,
    checkpoint_path: str | Path = "runs/linucb_week_level.npz",
    log_every: int = 10,
    warm_start_rounds: int = 0,
    with_replacement: bool = False,
) -> LinUCB:
    """Stream training instances, update one shared LinUCB, save checkpoint.

    Uses :func:`data.instances.enumerate_instances` on ``dataset_root`` with the
    train-style defaults (same as ``train_instances`` when ``dataset_root`` is
    ``Dataset/datasets_json``).

    Per-round reward scaling is applied inside :func:`run_week_level` (bandit
    sees scaled updates; returned trajectories stay unscaled).
    """
    if not arms:
        raise ValueError("arms must be non-empty")

    rs = float(reward_scale) if reward_scale else 1.0

    if dataset_root is not None:
        stream = enumerate_instances(
            dataset_root,
            seed=seed,
            shuffle=True,
            week_combos_per_scenario=week_combos_per_scenario,
            with_replacement=with_replacement,
        )
    else:
        stream = split_instances(
            split,
            seed=seed,
            shuffle=True,
            week_combos_per_scenario=week_combos_per_scenario,
            with_replacement=with_replacement,
        )

    arm_names = [a.name for a in arms]
    linucb: LinUCB | None = None
    total_rounds = 0
    pick_counts: Counter[str] = Counter()
    scaled_reward_values: list[float] = []
    pick_trajectory: list[str] = []
    theta_snapshots: list[list[list[float]]] = []
    snapshot_rounds: list[int] = []
    instance_count = 0
    warm_rounds_done = 0
    t_train0 = time.perf_counter()

    for inst in stream:
        if max_instances is not None and instance_count >= max_instances:
            break
        try:
            if linucb is None:
                ctx0, _ = build_context(
                    inst.scenario,
                    inst.initial_history,
                    inst.weeks[0],
                    0,
                    len(inst.weeks),
                )
                linucb = LinUCB(
                    num_arms=len(arms),
                    context_dim=int(ctx0.shape[0]),
                    alpha=alpha,
                )

            t0 = time.perf_counter()
            assert linucb is not None

            if warm_rounds_done < warm_start_rounds:
                start = warm_rounds_done
                idxs = [(start + i) % len(arms) for i in range(len(inst.weeks))]
                traj, picked = _run_instance_forced(
                    inst, arms, idxs, linucb, rs,
                )
                arms_picked_names = picked
                warm_rounds_done += len(inst.weeks)
            else:
                out = run_week_level(
                    scenario=inst.scenario,
                    initial_history=inst.initial_history,
                    week_data_list=inst.weeks,
                    arms=arms,
                    bandit=linucb,
                    reward_scale=rs,
                )
                traj = out.get("linucb_reward_trajectory", [])
                arms_picked_names = out["arms_picked"]
            dt = time.perf_counter() - t0
            for r in traj:
                scaled_reward_values.append(float(r) / rs)
            for name in arms_picked_names:
                pick_counts[name] += 1
                pick_trajectory.append(name)
            total_rounds += len(inst.weeks)
            instance_count += 1
            if log_every > 0 and instance_count % log_every == 0:
                theta_snapshots.append(
                    [linucb.theta(i).tolist() for i in range(len(arms))]
                )
                snapshot_rounds.append(total_rounds)
            log.info(
                "[%d] %s | %.2fs | weeks=%d | mean_unscaled_traj=%.2f",
                instance_count,
                inst.dataset_name,
                dt,
                len(inst.weeks),
                float(np.mean(traj)) if traj else 0.0,
            )

            if log_every > 0 and instance_count % log_every == 0:
                mr = (
                    float(np.mean(scaled_reward_values))
                    if scaled_reward_values
                    else 0.0
                )
                tail = scaled_reward_values[-200:]
                hist = (
                    np.histogram(tail, bins=8)[0].tolist()
                    if tail
                    else []
                )
                log.info(
                    "[%d] mean_scaled_reward=%.6f arm_usage=%s hist(last≤200)=%s",
                    instance_count,
                    mr,
                    dict(pick_counts),
                    hist,
                )
        except Exception as e:
            log.warning("Skipping instance %s: %s", getattr(inst, "dataset_name", "?"), e)
            continue

    if linucb is None:
        raise RuntimeError("No instance completed training (empty stream or all skipped)")

    training_config: dict[str, Any] = {
        "split": split,
        "dataset_root": dataset_root,
        "alpha": alpha,
        "max_instances": max_instances,
        "week_combos_per_scenario": week_combos_per_scenario,
        "seed": seed,
        "reward_scale": rs,
        "log_every": log_every,
        "warm_start_rounds": warm_start_rounds,
        "instances_run": instance_count,
        "total_bandit_rounds": total_rounds,
        "wall_clock_s": time.perf_counter() - t_train0,
        "arm_names": arm_names,
    }

    metadata = {
        "feature_labels": list(FEATURE_LABELS),
        "reward_scale": rs,
        "training_config": training_config,
    }

    linucb.save(
        checkpoint_path,
        metadata=metadata,
    )
    sidecar_path = Path(checkpoint_path).with_suffix(".trajectory.json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps({
        "arm_names": arm_names,
        "feature_labels": list(FEATURE_LABELS),
        "pick_trajectory": pick_trajectory,
        "reward_trajectory_scaled": scaled_reward_values,
        "theta_snapshots": theta_snapshots,
        "snapshot_rounds": snapshot_rounds,
        "warm_start_rounds": warm_start_rounds,
        "training_config": training_config,
    }))
    log.info(
        "Saved checkpoint to %s (%.1fs, %d instances, %d rounds)",
        checkpoint_path,
        training_config["wall_clock_s"],
        instance_count,
        total_rounds,
    )
    return linucb


def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train week-level LinUCB (cross-instance).")
    p.add_argument(
        "--split", choices=SPLITS, default="train",
        help="Named split (train/dev/val/test). Ignored if --dataset-root is given.",
    )
    p.add_argument(
        "--dataset-root", default=None,
        help="Explicit dataset root; overrides --split.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-instances", type=int, default=None)
    p.add_argument("--week-combos-per-scenario", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-scale", type=float, default=1000.0)
    p.add_argument("--checkpoint", default="runs/linucb_week_level.npz")
    p.add_argument("--log-every", type=int, default=10)
    registry = _build_registry()
    p.add_argument(
        "--arms",
        nargs="+",
        choices=list(registry.keys()),
        default=list(registry.keys()),
        help="Arms to register (default: all four).",
    )
    p.add_argument(
        "--warm-start-rounds",
        type=int,
        default=0,
        help="Force round-robin arm selection for the first N bandit rounds before letting LinUCB choose.",
    )
    p.add_argument(
        "--with-replacement",
        action="store_true",
        help="Allow the same WD file to appear multiple times in a week-combo (e.g. (0, 0, 0, 0)).",
    )
    args = p.parse_args()

    arms = [registry[name]() for name in args.arms]

    train_linucb(
        arms,
        split=args.split,
        dataset_root=args.dataset_root,
        alpha=args.alpha,
        max_instances=args.max_instances,
        week_combos_per_scenario=args.week_combos_per_scenario,
        seed=args.seed,
        reward_scale=args.reward_scale,
        checkpoint_path=args.checkpoint,
        log_every=args.log_every,
        warm_start_rounds=args.warm_start_rounds,
        with_replacement=args.with_replacement,
    )


if __name__ == "__main__":
    _cli()
