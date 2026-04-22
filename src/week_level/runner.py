"""Week-level runner — one bandit round per week, one arm generates the week."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from bandit.linucb import LinUCB
from schedule.representation import Schedule, DAY_NAMES_FULL
from schedule.penalty import compute_penalty, _compute_week_history

from week_level.arms.base import WeekArm
from week_level.context_builder import build_context


ArmSelector = Callable[[list[WeekArm], int], WeekArm]


def first_arm_selector(arms: list[WeekArm], week_idx: int) -> WeekArm:
    return arms[0]


def run_week_level(
    scenario: dict[str, Any],
    initial_history: dict[str, Any],
    week_data_list: list[dict[str, Any]],
    arms: list[WeekArm],
    selector: ArmSelector = first_arm_selector,
    bandit: LinUCB | None = None,
    reward_scale: float = 1.0,
) -> dict[str, Any]:
    """Drive the week-level bandit loop and score with the shared penalty.

    For each week, an arm generates assignments against the current history;
    those assignments are folded into a shared ``Schedule``, and history is
    advanced for the next week. The full multi-week schedule is then scored
    with ``compute_penalty``.

    **Selector modes**

    * If ``bandit`` is ``None`` (default), ``selector(arms, week_idx)`` picks
      the arm (e.g. :func:`first_arm_selector`).
    * If ``bandit`` is a :class:`bandit.linucb.LinUCB` instance, context is built
      with :func:`week_level.context_builder.build_context` from the *current*
      rolling history, ``bandit.choose`` picks the arm index, and after the week
      is applied we update with a scalar reward (see below). ``selector`` is
      ignored in this mode.

    **Per-week reward (LinUCB only)**

    Let ``P`` be the total *soft* penalty (sum of S1–S7, same as
    :attr:`schedule.penalty.PenaltyResult.total`). Before adding week ``w``,
    ``penalty_before = P`` on the schedule with weeks ``0..w-1`` filled; after
    merging that week's assignments, ``penalty_after = P``. Reward is::

        reward = penalty_before - penalty_after

    so positive values mean this step reduced aggregate soft penalty. (Hard
    counts are not included in ``P``.)

    **Reward scaling (LinUCB only)**

    If ``reward_scale != 1.0``, the value passed to ``bandit.update`` is
    ``(penalty_before - penalty_after) / reward_scale``. The list
    ``linucb_reward_trajectory`` still records **unscaled** deltas for
    interpretability.
    """
    if not arms:
        raise ValueError("arms must be non-empty")

    nurse_ids = [n["id"] for n in scenario["nurses"]]
    schedule = Schedule(num_weeks=len(week_data_list), nurse_ids=nurse_ids)

    current_history = initial_history
    arms_picked: list[str] = []
    reward_trajectory: list[float] = []

    total_weeks = len(week_data_list)

    if bandit is not None:
        if bandit.num_arms != len(arms):
            raise ValueError(
                f"bandit.num_arms ({bandit.num_arms}) must match len(arms) ({len(arms)})"
            )

    for week_idx, week_data in enumerate(week_data_list):
        if bandit is not None:
            context, _ = build_context(
                scenario, current_history, week_data, week_idx, total_weeks,
            )
            if context.shape[0] != bandit.context_dim:
                raise ValueError(
                    f"context dim {context.shape[0]} != bandit.context_dim "
                    f"({bandit.context_dim})"
                )
            arm_idx = bandit.choose(context)
            arm = arms[arm_idx]
        else:
            context = None
            arm_idx = -1
            arm = selector(arms, week_idx)

        arms_picked.append(arm.name)

        if bandit is not None:
            penalty_before_soft = compute_penalty(
                schedule, scenario, week_data_list, initial_history,
            ).total

        assignments = arm.generate(scenario, current_history, week_data)
        for a in assignments:
            day_in_week = DAY_NAMES_FULL.index(a["day"])
            global_day = week_idx * 7 + day_in_week
            schedule.add_assignment(
                a["nurseId"], global_day, a["shiftType"], a["skill"],
            )

        if bandit is not None:
            penalty_after_soft = compute_penalty(
                schedule, scenario, week_data_list, initial_history,
            ).total
            assert context is not None
            reward = float(penalty_before_soft - penalty_after_soft)
            reward_trajectory.append(reward)
            rs = float(reward_scale)
            scaled = reward / rs if rs != 0.0 else reward
            bandit.update(arm_idx, context, scaled)

        if week_idx < len(week_data_list) - 1:
            current_history = _compute_week_history(
                schedule, week_idx, current_history, scenario,
            )

    result = compute_penalty(schedule, scenario, week_data_list, initial_history)

    out: dict[str, Any] = {
        "schedule": schedule,
        "arms_picked": arms_picked,
        "total_penalty": result.total,
        "hard_violations": result.hard,
        "soft_breakdown": {
            "s1_optimal_coverage": result.s1_optimal_coverage,
            "s2_consecutive": result.s2_consecutive,
            "s3_days_off": result.s3_days_off,
            "s4_preferences": result.s4_preferences,
            "s5_complete_weekends": result.s5_complete_weekends,
            "s6_total_assignments": result.s6_total_assignments,
            "s7_working_weekends": result.s7_working_weekends,
        },
    }

    if bandit is not None:
        out["linucb_reward_trajectory"] = reward_trajectory
        out["linucb_thetas"] = [bandit.theta(i) for i in range(len(arms))]
        out["linucb_context_dim"] = bandit.context_dim
        out["linucb_alpha"] = bandit.alpha

    return out


def _fingerprint(assignments: list[dict]) -> set[tuple[str, str, str, str]]:
    return {
        (a["nurseId"], a["day"], a["shiftType"], a["skill"]) for a in assignments
    }


if __name__ == "__main__":
    from collections import Counter
    from itertools import combinations

    from instance_loader import load_instance
    from week_level.arms import (
        CoverageFirstArm,
        FatigueAwareArm,
        PreferenceRespectingArm,
        WeekendBalancingArm,
    )

    instance = load_instance(
        dataset_root="Dataset/datasets_json",
        dataset_name="n030w4",
        history_idx=0,
        week_indices=[0, 1, 2, 3],
    )

    arms = [
        CoverageFirstArm(),
        FatigueAwareArm(),
        WeekendBalancingArm(),
        PreferenceRespectingArm(),
    ]

    # Same week: at least two arms should differ in who fills which slots.
    w0 = instance.weeks[0]
    fps = [ _fingerprint(a.generate(instance.scenario, instance.initial_history, w0)) for a in arms ]
    assert any(fps[i] != fps[j] for i, j in combinations(range(len(arms)), 2)), (
        "tie-breaks produced identical assignments for all arms"
    )

    print("=== Baseline: first_arm_selector (always arms[0]) ===")
    out_base = run_week_level(
        scenario=instance.scenario,
        initial_history=instance.initial_history,
        week_data_list=instance.weeks,
        arms=arms,
        selector=first_arm_selector,
    )
    print("Arms picked per week:", out_base["arms_picked"])
    print("Hard violations:", out_base["hard_violations"])
    print("Soft breakdown:", out_base["soft_breakdown"])
    print("Total soft penalty:", out_base["total_penalty"])

    print()
    print("=== LinUCB (4 arms) ===")
    _ctx0, _ = build_context(
        instance.scenario,
        instance.initial_history,
        instance.weeks[0],
        0,
        len(instance.weeks),
    )
    lin = LinUCB(
        num_arms=len(arms),
        context_dim=int(_ctx0.shape[0]),
        alpha=1.0,
        seed=42,
    )
    out_lb = run_week_level(
        scenario=instance.scenario,
        initial_history=instance.initial_history,
        week_data_list=instance.weeks,
        arms=arms,
        bandit=lin,
    )
    print("Arms picked per week:", out_lb["arms_picked"])
    print("LinUCB reward trajectory:", out_lb["linucb_reward_trajectory"])
    print("arm pick counts:", dict(Counter(out_lb["arms_picked"])))
    for i, a in enumerate(arms):
        print(f"theta[{a.name}]:", np.round(out_lb["linucb_thetas"][i], 4))
    print("Hard violations:", out_lb["hard_violations"])
    print("Total soft penalty:", out_lb["total_penalty"])
