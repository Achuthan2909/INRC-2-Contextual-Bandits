"""
Preference-first scheduling strategy.

Prioritises nurse satisfaction: slots with the most shift-off requests are
filled last (so nurses get their preferences wherever coverage allows).
Within each slot, candidates are sorted so nurses with NO request penalty
are assigned first — the opposite priority order to coverage_first, which
assigns the lowest-penalty nurse but still picks greedily slot-by-slot.

Same signature as generate_schedule_coverage_first so it can be swapped in
directly by a bandit or main.py.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from coverage_first import (
    DAY_NAMES,
    build_nurse_lookup,
    build_history_lookup,
    build_forbidden_map,
    extract_minimum_requirements,
    nurse_skill,
    already_worked,
    violates_forbidden,
    score_candidate_basic,
)


def _slot_request_load(slot: dict, week_data: dict[str, Any]) -> int:
    """Count how many shift-off requests touch this (day, shiftType) slot."""
    day, shift = slot["day"], slot["shiftType"]
    return sum(
        1 for r in week_data.get("shiftOffRequests", [])
        if r["day"] == day and r["shiftType"] in (shift, "Any")
    )


def get_feasible_candidates_pref(
    day: str,
    day_idx: int,
    shift_type: str,
    skill: str,
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
    assignments_by_day: dict[str, set[str]],
    assigned_shift_by_nurse_day: dict[tuple[str, int], str],
) -> list[str]:
    nurse_lookup   = build_nurse_lookup(scenario)
    history_lookup = build_history_lookup(history)
    forbidden_map  = build_forbidden_map(scenario)

    feasible = [
        nid for nid, info in nurse_lookup.items()
        if nurse_skill(info, skill)
        and not already_worked(nid, day, assignments_by_day)
        and not violates_forbidden(
            nid, shift_type, day_idx,
            assigned_shift_by_nurse_day, history_lookup, forbidden_map
        )
    ]

    # Sort ascending by penalty — nurses with no request (penalty=0) come first,
    # preserving their preferences.  Ties broken by nurse id for determinism.
    feasible.sort(key=lambda nid: (
        score_candidate_basic(nid, day, shift_type, week_data),
        nid,
    ))
    return feasible


def generate_schedule_preference_first(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requirements = extract_minimum_requirements(week_data)

    # Sort slots so those with fewer request conflicts are filled first —
    # giving nurses in high-conflict slots a better chance of being free
    # when we finally assign them.
    requirements.sort(key=lambda s: _slot_request_load(s, week_data))

    assignments: list[dict[str, Any]] = []
    assignments_by_day: dict[str, set[str]] = defaultdict(set)
    assigned_shift_by_nurse_day: dict[tuple[str, int], str] = {}
    uncovered: list[dict[str, Any]] = []

    for req in requirements:
        day       = req["day"]
        day_idx   = DAY_NAMES.index(day)
        shift_type = req["shiftType"]
        skill     = req["skill"]
        minimum   = req["minimum"]

        candidates = get_feasible_candidates_pref(
            day, day_idx, shift_type, skill,
            scenario, history, week_data,
            assignments_by_day, assigned_shift_by_nurse_day,
        )

        assigned_count = 0
        for nid in candidates:
            if assigned_count >= minimum:
                break
            if nid in assignments_by_day[day]:
                continue
            assignments.append({"nurseId": nid, "day": day,
                                 "shiftType": shift_type, "skill": skill})
            assignments_by_day[day].add(nid)
            assigned_shift_by_nurse_day[(nid, day_idx)] = shift_type
            assigned_count += 1

        if assigned_count < minimum:
            uncovered.append({
                "day": day, "shiftType": shift_type, "skill": skill,
                "required": minimum, "assigned": assigned_count,
                "shortage": minimum - assigned_count,
            })

    return assignments, uncovered
