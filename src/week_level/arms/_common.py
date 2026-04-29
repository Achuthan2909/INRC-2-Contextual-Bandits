"""Shared helpers for week-level greedy arms (feasibility + minimum-coverage loop)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

DAY_KEY_MAP = {
    "Monday": "requirementOnMonday",
    "Tuesday": "requirementOnTuesday",
    "Wednesday": "requirementOnWednesday",
    "Thursday": "requirementOnThursday",
    "Friday": "requirementOnFriday",
    "Saturday": "requirementOnSaturday",
    "Sunday": "requirementOnSunday",
}


def build_forbidden_map(scenario: dict[str, Any]) -> dict[str, set[str]]:
    return {
        row["precedingShiftType"]: set(row["succeedingShiftTypes"])
        for row in scenario["forbiddenShiftTypeSuccessions"]
    }


def extract_minimum_requirements(week_data: dict[str, Any]) -> list[dict[str, Any]]:
    flat_reqs: list[dict[str, Any]] = []
    for req in week_data["requirements"]:
        shift_type = req["shiftType"]
        skill = req["skill"]
        for day in DAY_NAMES:
            minimum = req[DAY_KEY_MAP[day]]["minimum"]
            if minimum > 0:
                flat_reqs.append({
                    "day": day,
                    "shiftType": shift_type,
                    "skill": skill,
                    "minimum": minimum,
                })
    return flat_reqs


def shift_off_penalty(
    nurse_id: str,
    day: str,
    shift_type: str,
    shift_off_by_nurse: dict[str, list[dict[str, Any]]],
) -> int:
    penalty = 0
    for req in shift_off_by_nurse.get(nurse_id, []):
        if req["day"] != day:
            continue
        if req["shiftType"] == "Any":
            penalty += 100
        elif req["shiftType"] == shift_type:
            penalty += 150
    return penalty


def violates_forbidden(
    nurse_id: str,
    candidate_shift: str,
    day_idx: int,
    assigned_shift_by_nurse_day: dict[tuple[str, int], str],
    history_lookup: dict[str, dict[str, Any]],
    forbidden_map: dict[str, set[str]],
) -> bool:
    if day_idx == 0:
        prev_shift = history_lookup[nurse_id]["lastAssignedShiftType"]
    else:
        prev_shift = assigned_shift_by_nurse_day.get((nurse_id, day_idx - 1))
    if prev_shift in (None, "None"):
        return False
    return candidate_shift in forbidden_map.get(prev_shift, set())


def projected_consecutive_if_assigned_work(
    nurse_id: str,
    day_idx: int,
    assigned_shift_by_nurse_day: dict[tuple[str, int], str],
    history_lookup: dict[str, dict[str, Any]],
) -> int:
    """Consecutive working days ending the day we assign this shift (ascending = less fatigued)."""
    streak = 0
    d = day_idx - 1
    while d >= 0:
        sh = assigned_shift_by_nurse_day.get((nurse_id, d))
        if sh is None or sh in ("None", ""):
            break
        streak += 1
        d -= 1
    if streak == 0:
        if day_idx == 0:
            h = history_lookup[nurse_id]
            last = h.get("lastAssignedShiftType")
            if last in (None, "None", ""):
                return 1
            return int(h["numberOfConsecutiveWorkingDays"]) + 1
        return 1
    return streak + 1


Ctx = dict[str, Any]

GreedyKey = Callable[[str, Ctx], tuple[int | float, ...]]


def greedy_minimum_schedule(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
    requirements: list[dict[str, Any]],
    feasible_key: GreedyKey,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fill minimum requirements in order; sort feasible nurses with ``feasible_key`` (ascending)."""
    nurse_lookup = {n["id"]: n for n in scenario["nurses"]}
    history_lookup = {h["nurse"]: h for h in history["nurseHistory"]}
    forbidden_map = build_forbidden_map(scenario)

    shift_off_by_nurse: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for req in week_data.get("shiftOffRequests", []):
        shift_off_by_nurse[req["nurse"]].append(req)

    assignments: list[dict[str, Any]] = []
    assignments_by_day: dict[str, set[str]] = defaultdict(set)
    assigned_shift_by_nurse_day: dict[tuple[str, int], str] = {}
    uncovered: list[dict[str, Any]] = []

    for req in requirements:
        day = req["day"]
        day_idx = DAY_NAMES.index(day)
        shift_type = req["shiftType"]
        skill = req["skill"]
        minimum = req["minimum"]

        feasible: list[str] = []
        for nurse_id in nurse_lookup:
            info = nurse_lookup[nurse_id]
            if skill not in info["skills"]:
                continue
            if nurse_id in assignments_by_day[day]:
                continue
            if violates_forbidden(
                nurse_id, shift_type, day_idx,
                assigned_shift_by_nurse_day, history_lookup, forbidden_map,
            ):
                continue
            feasible.append(nurse_id)

        ctx: Ctx = {
            "day": day,
            "day_idx": day_idx,
            "shift_type": shift_type,
            "skill": skill,
            "minimum": minimum,
            "nurse_lookup": nurse_lookup,
            "history_lookup": history_lookup,
            "shift_off_by_nurse": shift_off_by_nurse,
            "assigned_shift_by_nurse_day": assigned_shift_by_nurse_day,
            "assignments_by_day": assignments_by_day,
            "week_data": week_data,
        }
        feasible.sort(key=lambda nid: feasible_key(nid, ctx))

        assigned_count = 0
        for nurse_id in feasible:
            if assigned_count >= minimum:
                break
            assignments.append({
                "nurseId": nurse_id,
                "day": day,
                "shiftType": shift_type,
                "skill": skill,
            })
            assignments_by_day[day].add(nurse_id)
            assigned_shift_by_nurse_day[(nurse_id, day_idx)] = shift_type
            assigned_count += 1

        if assigned_count < minimum:
            uncovered.append({
                "day": day,
                "shiftType": shift_type,
                "skill": skill,
                "required": minimum,
                "assigned": assigned_count,
                "shortage": minimum - assigned_count,
            })

    return assignments, uncovered
