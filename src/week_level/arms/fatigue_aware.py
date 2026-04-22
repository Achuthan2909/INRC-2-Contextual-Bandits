"""Prefer nurses with lower projected consecutive working streak (load rotation)."""
from __future__ import annotations

from typing import Any

from week_level.arms._common import (
    Ctx,
    extract_minimum_requirements,
    greedy_minimum_schedule,
    projected_consecutive_if_assigned_work,
    shift_off_penalty,
)
from week_level.arms.base import WeekArm


def _key_fatigue(nurse_id: str, ctx: Ctx) -> tuple[int, int]:
    pc = projected_consecutive_if_assigned_work(
        nurse_id,
        ctx["day_idx"],
        ctx["assigned_shift_by_nurse_day"],
        ctx["history_lookup"],
    )
    so = shift_off_penalty(
        nurse_id, ctx["day"], ctx["shift_type"], ctx["shift_off_by_nurse"],
    )
    return (pc, so)


def generate_schedule_fatigue_aware(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reqs = extract_minimum_requirements(week_data)
    return greedy_minimum_schedule(
        scenario, history, week_data, reqs, _key_fatigue,
    )


class FatigueAwareArm(WeekArm):
    name = "fatigue_aware"

    def generate(
        self,
        scenario: dict[str, Any],
        history: dict[str, Any],
        week_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assignments, _ = generate_schedule_fatigue_aware(
            scenario=scenario, history=history, week_data=week_data,
        )
        return assignments
