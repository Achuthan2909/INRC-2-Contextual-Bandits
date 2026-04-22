"""Prioritize slots with many shift-off requests; tie-break by preference then load."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from week_level.arms._common import (
    Ctx,
    extract_minimum_requirements,
    greedy_minimum_schedule,
    shift_off_penalty,
)
from week_level.arms.base import WeekArm


def _order_requirements_by_requests(
    week_data: dict[str, Any],
    flat_reqs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shifts_by_day: dict[str, set[str]] = defaultdict(set)
    for fr in flat_reqs:
        shifts_by_day[fr["day"]].add(fr["shiftType"])

    cnt: Counter[tuple[str, str]] = Counter()
    for r in week_data.get("shiftOffRequests", []):
        d, st = r["day"], r["shiftType"]
        if st == "Any":
            for st2 in shifts_by_day.get(d, ()):
                cnt[(d, st2)] += 1
        else:
            cnt[(d, st)] += 1

    return sorted(
        flat_reqs,
        key=lambda x: (-cnt[(x["day"], x["shiftType"])], x["day"], x["shiftType"], x["skill"]),
    )


def _key_preference(nurse_id: str, ctx: Ctx) -> tuple[int, int]:
    so = shift_off_penalty(
        nurse_id, ctx["day"], ctx["shift_type"], ctx["shift_off_by_nurse"],
    )
    n = int(ctx["history_lookup"][nurse_id]["numberOfAssignments"])
    return (so, n)


def generate_schedule_preference_respecting(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flat = extract_minimum_requirements(week_data)
    reqs = _order_requirements_by_requests(week_data, flat)
    return greedy_minimum_schedule(
        scenario, history, week_data, reqs, _key_preference,
    )


class PreferenceRespectingArm(WeekArm):
    name = "preference_respecting"

    def generate(
        self,
        scenario: dict[str, Any],
        history: dict[str, Any],
        week_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assignments, _ = generate_schedule_preference_respecting(
            scenario=scenario, history=history, week_data=week_data,
        )
        return assignments
