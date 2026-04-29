"""
Random baseline scheduling strategy.

Selects nurses uniformly at random from the feasible candidate pool for each
slot.  Hard constraints (skill match, no double-booking, forbidden successions)
are still respected — only the ordering is random.

Used as a lower-bound baseline: any learned strategy should outperform this.
Set `seed` for reproducible runs.

Same signature as generate_schedule_coverage_first (plus optional seed).
"""
from __future__ import annotations

import random
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


def get_feasible_candidates_random(
    day: str,
    day_idx: int,
    shift_type: str,
    skill: str,
    scenario: dict[str, Any],
    history: dict[str, Any],
    assignments_by_day: dict[str, set[str]],
    assigned_shift_by_nurse_day: dict[tuple[str, int], str],
    rng: random.Random,
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

    rng.shuffle(feasible)
    return feasible


def generate_schedule_random(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
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

        candidates = get_feasible_candidates_random(
            day, day_idx, shift_type, skill,
            scenario, history,
            assignments_by_day, assigned_shift_by_nurse_day,
            rng,
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
