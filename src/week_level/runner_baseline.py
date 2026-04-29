"""Week-level runner for non-contextual baselines (uniform / UCB1 / eps-greedy / Thompson).

Mirrors :func:`week_level.runner.run_week_level` but drives a
:class:`bandit.base.BanditSelector` (which doesn't consume a context). At each
week we call ``selector._pick`` over all arm names, generate the week, compute
penalty-reduction reward, and feed it back via ``selector.update``.

Kept separate from ``run_week_level`` so the contextual LinUCB path doesn't
grow extra branches. Both runners return the same result shape (minus the
``linucb_*`` fields, which are replaced with baseline-appropriate analogs).
"""
from __future__ import annotations

from typing import Any

from bandit.base import BanditSelector
from schedule.representation import Schedule, DAY_NAMES_FULL
from schedule.penalty import compute_penalty, _compute_week_history

from week_level.arms.base import WeekArm


def run_week_level_baseline(
    scenario: dict[str, Any],
    initial_history: dict[str, Any],
    week_data_list: list[dict[str, Any]],
    arms: list[WeekArm],
    selector: BanditSelector,
    reward_scale: float = 1.0,
) -> dict[str, Any]:
    """Non-contextual week-level driver.

    At each week:
      1. ``selector._pick(all_arm_names)`` picks an arm by name.
      2. The arm generates that week's assignments; we compute
         ``reward = penalty_before - penalty_after`` on the aggregate schedule.
      3. ``selector.update(arm_name, reward / reward_scale)`` feeds the reward back.
    """
    if not arms:
        raise ValueError("arms must be non-empty")
    arm_names = [a.name for a in arms]
    arm_by_name = {a.name: a for a in arms}

    nurse_ids = [n["id"] for n in scenario["nurses"]]
    schedule = Schedule(num_weeks=len(week_data_list), nurse_ids=nurse_ids)

    current_history = initial_history
    arms_picked: list[str] = []
    reward_trajectory: list[float] = []
    rs = float(reward_scale) if reward_scale else 1.0

    for week_idx, week_data in enumerate(week_data_list):
        name = selector._pick(arm_names)
        arm = arm_by_name[name]
        arms_picked.append(name)

        penalty_before = compute_penalty(
            schedule, scenario, week_data_list, initial_history,
        ).total

        assignments = arm.generate(scenario, current_history, week_data)
        for a in assignments:
            day_in_week = DAY_NAMES_FULL.index(a["day"])
            global_day = week_idx * 7 + day_in_week
            schedule.add_assignment(
                a["nurseId"], global_day, a["shiftType"], a["skill"],
            )

        penalty_after = compute_penalty(
            schedule, scenario, week_data_list, initial_history,
        ).total
        reward = float(penalty_before - penalty_after)
        reward_trajectory.append(reward)
        selector.update(name, reward / rs)

        if week_idx < len(week_data_list) - 1:
            current_history = _compute_week_history(
                schedule, week_idx, current_history, scenario,
            )

    result = compute_penalty(schedule, scenario, week_data_list, initial_history)

    return {
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
        "baseline_reward_trajectory": reward_trajectory,
        "baseline_stats": selector.stats(),
    }


__all__ = ["run_week_level_baseline"]
