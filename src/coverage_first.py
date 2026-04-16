from __future__ import annotations

from collections import defaultdict
from typing import Any

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  

DAY_KEY_MAP = {
    "Monday": "requirementOnMonday",
    "Tuesday": "requirementOnTuesday",
    "Wednesday": "requirementOnWednesday",
    "Thursday": "requirementOnThursday",
    "Friday": "requirementOnFriday",
    "Saturday": "requirementOnSaturday",
    "Sunday": "requirementOnSunday"
}

def build_nurse_lookup (scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n["id"]: n for n in scenario["nurses"]}

def build_history_lookup (history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {h["nurse"]: h for h in history["nurseHistory"]}

def build_forbidden_map (scenario: dict[str, Any]) -> dict[str, set[str]]:
    forbidden = {}
    for row in scenario["forbiddenShiftTypeSuccessions"]:
        forbidden[row["precedingShiftType"]] = set(row["succeedingShiftTypes"])
    return forbidden

def extract_minimum_requirements (week_data: dict[str, Any]) -> list[dict[str, int]]:
    flat_reqs = []

    for req in week_data["requirements"]:
        shift_type = req["shiftType"]
        skill = req["skill"]
        
        for day in DAY_NAMES:
            day_key = DAY_KEY_MAP[day]
            minimum = req[day_key]["minimum"]

            if minimum > 0:
                flat_reqs.append({
                    "day": day,
                    "shiftType": shift_type,
                    "skill": skill,
                    "minimum": minimum
                })

    return flat_reqs

def nurse_skill (nurse_info: dict[str, Any], skill: str) -> bool:
    return skill in nurse_info["skills"]

def already_worked (
        nurse_id: str,
        day: str,
        assignments_by_day: dict[str, set[str]]
) -> bool:
    return nurse_id in assignments_by_day[day]

def violates_forbidden (
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
        prev_shift = assigned_shift_by_nurse_day.get((nurse_id, day_idx - 1), None)
    
    if prev_shift is None:
        return False
    
    forbidden_next = forbidden_map.get(prev_shift, set())
    return candidate_shift in forbidden_next

def score_candidate_basic (
        nurse_id: str,
        day: str,
        shift_type: str,
        week_data: dict[str, Any]
) -> int:
    penalty = 0

    for req in week_data.get("shiftOffRequests", []):
        if req["nurse"] != nurse_id:
            continue
        if req["day"] != day:
            continue

        if req["shiftType"] == "Any":
            penalty += 100
        elif req["shiftType"] == shift_type:
            penalty += 150

    return penalty

def get_feasible_candidates (
        day: str,
        day_idx: int,
        shift_type: str,
        skill: str,
        scenario: dict[str, Any],
        history: dict[str, Any],
        week_data: dict[str, Any],
        assignments_by_day: dict[str, set[str]],
        assigned_shift_by_nurse_day: dict[tuple[str, int], str]
) -> list[str]:
    nurse_lookup = build_nurse_lookup(scenario)
    history_lookup = build_history_lookup(history)
    forbidden_map = build_forbidden_map(scenario)

    feasible = []

    for nurse_id, nurse_info in nurse_lookup.items():
        if not nurse_skill(nurse_info, skill):
            continue
        if already_worked(nurse_id, day, assignments_by_day):
            continue
        if violates_forbidden(nurse_id, shift_type, day_idx, assigned_shift_by_nurse_day, history_lookup, forbidden_map):
            continue
        
        feasible.append(nurse_id)

    feasible.sort(
        key=lambda nurse_id: score_candidate_basic(
            nurse_id=nurse_id,
            day=day,
            shift_type=shift_type,
            week_data=week_data
        )
    )

    return feasible


def generate_schedule_coverage_first(
        scenario: dict[str, Any],
        history: dict[str, Any],
        week_data: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requirements = extract_minimum_requirements(week_data)
    assignments: list[dict[str, Any]] = []

    # Track whether a nurse is already assigned on a day
    assignments_by_day: dict[str, set[str]] = defaultdict(set)

    # Track shift assigned to nurse on each day index for succession checks
    assigned_shift_by_nurse_day: dict[tuple[str, int], str] = {}
    
    uncovered: list[dict[str, Any]] = []

    for req in requirements:
        day = req["day"]
        day_idx = DAY_NAMES.index(day)
        shift_type = req["shiftType"]
        skill = req["skill"]
        minimum = req["minimum"]

        assigned_count = 0

        candidates = get_feasible_candidates(
            day=day,
            day_idx=day_idx,
            shift_type=shift_type,
            skill=skill,
            scenario=scenario,
            history=history,
            week_data=week_data,
            assignments_by_day=assignments_by_day,
            assigned_shift_by_nurse_day=assigned_shift_by_nurse_day
        )

        for nurse_id in candidates:
            if assigned_count >= minimum:
                break

            # safety check: maybe candidate became  unavailable
            if nurse_id in assignments_by_day[day]:
                continue

            assignments.append({
                "nurseId": nurse_id,
                "day": day,
                "shiftType": shift_type,
                "skill": skill
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
                "shortage": minimum - assigned_count
            })
            
    return assignments, uncovered
