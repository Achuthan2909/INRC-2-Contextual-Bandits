"""
Max-coverage scheduling strategy.

Fills each slot up to its MAXIMUM staffing level (not just minimum), reducing
the risk of being short-staffed if nurses call in sick or constraints become
tighter in later slots.  Candidate ordering is the same as coverage_first
(lowest penalty first).

Same signature as generate_schedule_coverage_first.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

DAY_KEY_MAP = {
    "Monday":    "requirementOnMonday",
    "Tuesday":   "requirementOnTuesday",
    "Wednesday": "requirementOnWednesday",
    "Thursday":  "requirementOnThursday",
    "Friday":    "requirementOnFriday",
    "Saturday":  "requirementOnSaturday",
    "Sunday":    "requirementOnSunday",
}

from coverage_first import (
    build_nurse_lookup,
    build_history_lookup,
    build_forbidden_map,
    nurse_skill,
    already_worked,
    violates_forbidden,
    score_candidate_basic,
)


def extract_max_requirements(week_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Like extract_minimum_requirements but also captures the maximum field."""
    flat = []
    for req in week_data["requirements"]:
        shift_type = req["shiftType"]
        skill      = req["skill"]
        for day in DAY_NAMES:
            day_key = DAY_KEY_MAP[day]
            minimum = req[day_key]["minimum"]
            maximum = req[day_key]["maximum"]
            if maximum > 0:
                flat.append({
                    "day":       day,
                    "shiftType": shift_type,
                    "skill":     skill,
                    "minimum":   minimum,
                    "maximum":   maximum,
                })
    return flat


def get_feasible_candidates_max(
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

    feasible.sort(key=lambda nid: (
        score_candidate_basic(nid, day, shift_type, week_data),
        nid,
    ))
    return feasible


def generate_schedule_max_coverage(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requirements = extract_max_requirements(week_data)

    assignments: list[dict[str, Any]] = []
    assignments_by_day: dict[str, set[str]] = defaultdict(set)
    assigned_shift_by_nurse_day: dict[tuple[str, int], str] = {}
    uncovered: list[dict[str, Any]] = []

    for req in requirements:
        day        = req["day"]
        day_idx    = DAY_NAMES.index(day)
        shift_type = req["shiftType"]
        skill      = req["skill"]
        minimum    = req["minimum"]
        maximum    = req["maximum"]

        candidates = get_feasible_candidates_max(
            day, day_idx, shift_type, skill,
            scenario, history, week_data,
            assignments_by_day, assigned_shift_by_nurse_day,
        )

        assigned_count = 0
        for nid in candidates:
            if assigned_count >= maximum:   # fill to maximum, not minimum
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
