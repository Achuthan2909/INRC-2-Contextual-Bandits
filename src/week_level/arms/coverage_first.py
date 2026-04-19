"""Coverage-first week-level arm — tie-break by shift-off penalty only."""
from __future__ import annotations

from typing import Any

from week_level.arms._common import (
    Ctx,
    extract_minimum_requirements,
    greedy_minimum_schedule,
    shift_off_penalty,
)
from week_level.arms.base import WeekArm


def _key_coverage_first(nurse_id: str, ctx: Ctx) -> tuple[int,]:
    return (
        shift_off_penalty(
            nurse_id, ctx["day"], ctx["shift_type"], ctx["shift_off_by_nurse"],
        ),
    )


def generate_schedule_coverage_first(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reqs = extract_minimum_requirements(week_data)
    return greedy_minimum_schedule(
        scenario, history, week_data, reqs, _key_coverage_first,
    )


class CoverageFirstArm(WeekArm):
    name = "coverage_first"

    def generate(
        self,
        scenario: dict[str, Any],
        history: dict[str, Any],
        week_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assignments, _ = generate_schedule_coverage_first(
            scenario=scenario, history=history, week_data=week_data,
        )
        return assignments
