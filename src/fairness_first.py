"""
Fairness-first scheduling strategy.

Assigns nurses with the fewest total shifts worked (history + current week so
far) first, spreading workload as evenly as possible.  Coverage minimums are
still respected — fairness only changes the candidate ordering, not whether
a slot gets filled.

Same signature as generate_schedule_coverage_first.
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
)


def _total_assignments(
    nurse_id: str,
    history_lookup: dict[str, dict[str, Any]],
    assignments_so_far: list[dict[str, Any]],
) -> int:
    """Shifts worked historically + shifts assigned so far this week."""
    hist = history_lookup.get(nurse_id, {})
    # INRC-2 history tracks numberOfAssignments
    historical = hist.get("numberOfAssignments", 0)
    current    = sum(1 for a in assignments_so_far if a["nurseId"] == nurse_id)
    return historical + current


def get_feasible_candidates_fair(
    day: str,
    day_idx: int,
    shift_type: str,
    skill: str,
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
    assignments_by_day: dict[str, set[str]],
    assigned_shift_by_nurse_day: dict[tuple[str, int], str],
    assignments_so_far: list[dict[str, Any]],
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

    # Least-worked nurse first; ties broken by id for determinism
    feasible.sort(key=lambda nid: (
        _total_assignments(nid, history_lookup, assignments_so_far),
        nid,
    ))
    return feasible


def generate_schedule_fairness_first(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requirements = extract_minimum_requirements(week_data)

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

        candidates = get_feasible_candidates_fair(
            day, day_idx, shift_type, skill,
            scenario, history, week_data,
            assignments_by_day, assigned_shift_by_nurse_day,
            assignments_so_far=assignments,
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
