"""On weekends, prefer nurses with fewer working weekends; weekdays = shift-off tie."""
from __future__ import annotations

from typing import Any

from week_level.arms._common import (
    Ctx,
    extract_minimum_requirements,
    greedy_minimum_schedule,
    shift_off_penalty,
)
from week_level.arms.base import WeekArm


def _weekend_extra_this_week(
    nurse_id: str,
    assigned_shift_by_nurse_day: dict[tuple[str, int], str],
) -> int:
    sat = assigned_shift_by_nurse_day.get((nurse_id, 5))
    sun = assigned_shift_by_nurse_day.get((nurse_id, 6))

    def ok(x: str | None) -> bool:
        return x is not None and x not in ("None", "")

    return 1 if (ok(sat) or ok(sun)) else 0


def _key_weekend(nurse_id: str, ctx: Ctx) -> tuple[int, int]:
    so = shift_off_penalty(
        nurse_id, ctx["day"], ctx["shift_type"], ctx["shift_off_by_nurse"],
    )
    day = ctx["day"]
    if day not in ("Saturday", "Sunday"):
        return (0, so)
    h = ctx["history_lookup"][nurse_id]
    base = int(h.get("numberOfWorkingWeekends", 0))
    bonus = _weekend_extra_this_week(nurse_id, ctx["assigned_shift_by_nurse_day"])
    return (base + bonus, so)


def generate_schedule_weekend_balancing(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reqs = extract_minimum_requirements(week_data)
    return greedy_minimum_schedule(
        scenario, history, week_data, reqs, _key_weekend,
    )


class WeekendBalancingArm(WeekArm):
    name = "weekend_balancing"

    def generate(
        self,
        scenario: dict[str, Any],
        history: dict[str, Any],
        week_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assignments, _ = generate_schedule_weekend_balancing(
            scenario=scenario, history=history, week_data=week_data,
        )
        return assignments
